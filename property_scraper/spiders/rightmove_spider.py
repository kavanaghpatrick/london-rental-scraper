"""
Rightmove Spider - Parallel property listing scraper

Uses Scrapy's built-in async engine for efficient parallel scraping.
Extracts data from Next.js __NEXT_DATA__ JSON embedded in pages.

Usage:
    cd scrapy_project
    scrapy crawl rightmove -o output/listings.json
    scrapy crawl rightmove -a areas=Belgravia,Chelsea -a max_pages=5
    scrapy crawl rightmove -a fetch_details=true   # Fetch sqft from detail pages
    scrapy crawl rightmove -a fetch_floorplans=true  # Enable inline OCR

When fetch_floorplans=true, downloads floorplan images and runs OCR to extract:
- size_sqft (if not found in page)
- floor_count, property_levels (single_floor, duplex, etc.)
- has_basement, has_ground, has_first_floor, etc.
Requires pytesseract: pip install pytesseract; brew install tesseract
"""

import scrapy
import json
import re
import time
import requests
from datetime import datetime
from urllib.parse import urljoin
from property_scraper.items import PropertyItem
from property_scraper.utils.url_validation import validate_floorplan_url

# OCR support for inline floorplan extraction
try:
    from property_scraper.utils.floorplan_extractor import FloorplanExtractor
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class RightmoveSpider(scrapy.Spider):
    """Spider for Rightmove rental listings with comprehensive logging."""

    name = 'rightmove'
    allowed_domains = ['rightmove.co.uk']

    # Default areas to scrape
    # Full Prime Central London coverage - matches registry.py
    DEFAULT_AREAS = [
        'Belgravia', 'Chelsea', 'Kensington', 'South-Kensington',
        'Knightsbridge', 'Notting-Hill', 'Earls-Court', 'Fulham',
        'Hampstead', 'St-Johns-Wood', 'Mayfair', 'Marylebone'
    ]

    def __init__(self, areas=None, max_pages=None, fetch_details=True, fetch_floorplans=False,
                 listing_type='rent', *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Listing vertical: 'rent' (default — UNCHANGED rental behaviour) or 'sale'
        # (the separate FOR-SALE vertical, pointed at the site's /property-for-sale/
        # section). Any value other than the explicit 'sale' falls back to 'rent' so a
        # typo can never silently switch the rental scrape to sale. This is the ONLY
        # knob that changes which part of the site is scraped; the rest of the spider
        # is shared infra. See parse_for_sale_property / start_url_for below.
        self.listing_type = 'sale' if str(listing_type).lower() == 'sale' else 'rent'

        # Parse areas argument
        if areas:
            self.areas = [a.strip() for a in areas.split(',')]
        else:
            self.areas = self.DEFAULT_AREAS

        # Safe parsing of max_pages (None = unlimited)
        if max_pages is None or str(max_pages).lower() in ('none', '0', ''):
            self.max_pages = None  # Unlimited
        else:
            try:
                self.max_pages = int(max_pages)
            except (ValueError, TypeError):
                self.logger.warning(f"[CONFIG] Invalid max_pages '{max_pages}', using unlimited")
                self.max_pages = None

        # Enable detail page fetching for sqft and descriptions
        self.fetch_details = str(fetch_details).lower() in ('true', '1', 'yes')

        # Enable floorplan extraction (implies fetch_details)
        self.fetch_floorplans = str(fetch_floorplans).lower() in ('true', '1', 'yes')
        if self.fetch_floorplans:
            self.fetch_details = True  # Need detail page for floorplans

        # Initialize floorplan OCR extractor
        if self.fetch_floorplans and OCR_AVAILABLE:
            self.floorplan_extractor = FloorplanExtractor()
        else:
            self.floorplan_extractor = None

        # Enhanced stats tracking
        self.stats = {
            'total': 0,
            'by_area': {},
            'prices': [],
            'missing_fields': {},
            'let_agreed_count': 0,
            'requests_made': 0,
            'requests_failed': 0,
            'bytes_downloaded': 0,
            'sqft_found': 0,
            'floorplans_found': 0,
            'floorplans_ocr_success': 0,
            'details_fetched': 0,
            'start_time': time.time(),
        }

        self.logger.info("=" * 70)
        self.logger.info("RIGHTMOVE SPIDER INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"[CONFIG] Areas to scrape: {', '.join(self.areas)}")
        self.logger.info(f"[CONFIG] Max pages per area: {self.max_pages or 'unlimited'}")
        self.logger.info(f"[CONFIG] Fetch details: {self.fetch_details}")
        self.logger.info(f"[CONFIG] Fetch floorplans: {self.fetch_floorplans}")
        self.logger.info(f"[CONFIG] Inline OCR: {self.floorplan_extractor is not None}")
        if self.fetch_floorplans and not OCR_AVAILABLE:
            self.logger.warning("[CONFIG] OCR requested but pytesseract not available!")
        if self.max_pages:
            self.logger.info(f"[CONFIG] Estimated max listings: ~{len(self.areas) * self.max_pages * 24}")
        else:
            self.logger.info(f"[CONFIG] Scraping all available pages")
        self.logger.info(f"[CONFIG] Listing type: {self.listing_type}")
        self.logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # FOR-SALE vertical (additive — default rental path is untouched)    #
    # ------------------------------------------------------------------ #
    def start_url_for(self, area: str) -> str:
        """Build the search URL for `area` in the configured vertical.

        Rental (default): the SAME /property-to-rent/ URL the spider has always used.
        Sale: the site's /property-for-sale/ section — the only change is which part of
        Rightmove we point the (otherwise identical) spider at. Pure/standalone so the
        for-sale parse tests can assert the URL without a crawl.
        """
        section = 'property-for-sale' if self.listing_type == 'sale' else 'property-to-rent'
        return f'https://www.rightmove.co.uk/{section}/{area}.html'

    def parse_for_sale_property(self, prop: dict, area: str):
        """Parse a single FOR-SALE property from the __NEXT_DATA__ search results.

        DELEGATES to for_sale.listing_parse.parse_rightmove_for_sale — the pure,
        independently-tested SINGLE SOURCE OF TRUTH for the for-sale parse (so the spider
        mode and the unit tests never drift on a duplicated copy of the logic). Returns a
        plain dict for the spider/data-layer path (asking_price in sale magnitude, never
        price_pcm/price_pw); a POA/amount-0 row yields asking_price None. None if no id.
        """
        # Import lazily so the for-sale package is only required when sale mode is used —
        # a pure rental crawl never imports it.
        from for_sale.listing_parse import parse_rightmove_for_sale

        sale_item = parse_rightmove_for_sale(prop, area)
        if sale_item is None:
            self.logger.warning(f"[FOR-SALE][VALIDATION] property missing ID in {area}")
            return None

        item = dict(sale_item)  # SaleListingItem -> plain dict for the yield/persist path
        # listing_parse stores amount 0 for POA; normalise to None at the spider boundary
        # so unpriced comps are explicitly skippable downstream (and never look like £0).
        if not item.get('asking_price'):
            item['asking_price'] = None
        return item

    def start_requests(self):
        """Generate initial requests for all target areas."""
        self.logger.info(
            f"[START] Launching {len(self.areas)} parallel area scrapers "
            f"(listing_type={self.listing_type})..."
        )

        for i, area in enumerate(self.areas):
            self.stats['by_area'][area] = {'count': 0, 'total_available': 0, 'pages': 0}
            url = self.start_url_for(area)

            self.logger.info(f"[REQUEST] [{i+1}/{len(self.areas)}] Starting: {area} -> {url}")

            yield scrapy.Request(
                url,
                callback=self.parse_search,
                meta={
                    'area': area,
                    'page': 0,
                    'request_start': time.time()
                },
                dont_filter=True,
                errback=self.handle_error
            )

    def handle_error(self, failure):
        """Handle request failures with detailed logging."""
        request = failure.request
        area = request.meta.get('area', 'unknown')

        self.stats['requests_failed'] += 1
        self.logger.error(f"[ERROR] Request failed for {area}: {failure.value}")
        self.logger.error(f"[ERROR] URL: {request.url}")
        self.logger.error(f"[ERROR] Type: {failure.type.__name__}")

    def parse_search(self, response):
        """Parse search results page with detailed logging."""
        area = response.meta['area']
        page = response.meta['page']
        request_time = time.time() - response.meta.get('request_start', time.time())

        self.stats['requests_made'] += 1
        self.stats['bytes_downloaded'] += len(response.body)

        # Log response details
        self.logger.info(
            f"[RESPONSE] {area} p{page+1} | "
            f"Status: {response.status} | "
            f"Size: {len(response.body)/1024:.1f}KB | "
            f"Time: {request_time:.2f}s"
        )

        # Issue #17 FIX: Retry on rate limiting with exponential backoff
        if response.status == 429:
            retry_count = response.meta.get('retry_count', 0)
            if retry_count < 3:
                # Get retry delay from header or use exponential backoff
                retry_after = response.headers.get('Retry-After', b'').decode('utf-8', errors='ignore')
                try:
                    wait_time = int(retry_after) if retry_after else (30 * (retry_count + 1))
                except ValueError:
                    wait_time = 30 * (retry_count + 1)

                self.logger.warning(
                    f"[RATE-LIMIT] Got 429 for {area}, retrying in {wait_time}s ({retry_count + 1}/3)"
                )
                # Note: Scrapy doesn't support async sleep in callbacks, so we use download_delay
                # The retry will be delayed by DOWNLOAD_DELAY setting
                yield response.request.replace(
                    meta={**response.meta, 'retry_count': retry_count + 1},
                    dont_filter=True
                )
                return
            else:
                self.logger.error(f"[RATE-LIMIT] Max retries exceeded for {area}, skipping page")
                return

        if response.status != 200:
            self.logger.warning(f"[HTTP-ERROR] {area} returned status {response.status}")
            return

        # Extract __NEXT_DATA__ JSON
        script = response.css('script#__NEXT_DATA__::text').get()

        if not script:
            self.logger.error(f"[PARSE-ERROR] No __NEXT_DATA__ found for {area}")
            self.logger.debug(f"[DEBUG] Response preview: {response.text[:500]}")
            return

        self.logger.debug(f"[JSON] Found __NEXT_DATA__ ({len(script)/1024:.1f}KB)")

        try:
            data = json.loads(script)
        except json.JSONDecodeError as e:
            self.logger.error(f"[JSON-ERROR] Failed to parse JSON for {area}: {e}")
            return

        # Navigate to properties with validation
        props = data.get('props', {})
        if not props:
            self.logger.warning(f"[STRUCTURE] No 'props' in data for {area}")
            return

        page_props = props.get('pageProps', {})
        if not page_props:
            self.logger.warning(f"[STRUCTURE] No 'pageProps' in data for {area}")
            return

        search_results = page_props.get('searchResults', {})
        properties = search_results.get('properties', [])

        # Log first page stats
        if page == 0:
            total_raw = search_results.get('resultCount', 0)
            # Handle string (may have commas) or int
            try:
                if isinstance(total_raw, str):
                    total = int(total_raw.replace(',', ''))
                else:
                    total = int(total_raw) if total_raw else 0
            except (ValueError, TypeError):
                total = 0
                self.logger.warning(f"[PARSE] Could not parse resultCount: {total_raw}")

            self.stats['by_area'][area]['total_available'] = total
            pages_available = (total // 24) + 1

            self.logger.info(f"[DISCOVERY] {area}: {total} listings across ~{pages_available} pages")

            # Log result structure for debugging
            result_keys = list(search_results.keys())
            self.logger.debug(f"[STRUCTURE] searchResults keys: {result_keys}")

        # Parse each property
        parsed_count = 0
        price_sum = 0

        for prop in properties:
            if self.listing_type == 'sale':
                # FOR-SALE vertical: yield the sale dict directly (no rental detail
                # fetch / floorplan path) so the sale magnitude never touches the rental
                # PropertyItem/pipeline. The sale data layer owns persistence.
                item = self.parse_for_sale_property(prop, area)
                if item:
                    parsed_count += 1
                    self.stats['total'] += 1
                    self.stats['by_area'][area]['count'] += 1
                    if item.get('asking_price'):
                        price_sum += item['asking_price']
                        self.stats['prices'].append(item['asking_price'])
                    yield item
                continue

            item = self.parse_property(prop, area)
            if item:
                parsed_count += 1
                self.stats['total'] += 1
                self.stats['by_area'][area]['count'] += 1

                if item.get('price_pcm'):
                    price_sum += item['price_pcm']
                    self.stats['prices'].append(item['price_pcm'])

                if item.get('let_agreed'):
                    self.stats['let_agreed_count'] += 1

                # Optionally fetch detail page for sqft and description
                if self.fetch_details and item.get('url'):
                    yield scrapy.Request(
                        item['url'],
                        callback=self.parse_detail,
                        meta={
                            'item': dict(item),
                            'request_start': time.time()
                        },
                        dont_filter=True,
                        errback=self.handle_error
                    )
                else:
                    yield item

        self.stats['by_area'][area]['pages'] += 1

        # Log page summary
        avg_price = price_sum // parsed_count if parsed_count else 0
        self.logger.info(
            f"[PAGE] {area} p{page+1}: "
            f"{parsed_count}/{len(properties)} parsed | "
            f"Avg: £{avg_price:,}/pcm | "
            f"Running total: {self.stats['by_area'][area]['count']}"
        )

        # Follow pagination
        pagination = search_results.get('pagination', {})
        next_index = pagination.get('next')

        # Follow pagination if there's a next page and we haven't hit max_pages limit
        should_continue = next_index and (self.max_pages is None or page < self.max_pages - 1)

        if should_continue:
            next_url = f'{self.start_url_for(area)}?index={next_index}'

            self.logger.debug(f"[PAGINATION] {area}: Following to page {page+2} (index={next_index})")

            yield scrapy.Request(
                next_url,
                callback=self.parse_search,
                meta={
                    'area': area,
                    'page': page + 1,
                    'request_start': time.time()
                },
                dont_filter=True,
                errback=self.handle_error
            )
        else:
            if self.max_pages and page >= self.max_pages - 1:
                self.logger.info(f"[COMPLETE] {area}: Reached max pages ({self.max_pages})")
            else:
                self.logger.info(f"[COMPLETE] {area}: No more pages (stopped at {page+1})")

    def parse_property(self, prop: dict, area: str) -> PropertyItem:
        """Parse a single property from search results with validation logging."""
        item = PropertyItem()

        prop_id = prop.get('id', '')
        if not prop_id:
            self.logger.warning(f"[VALIDATION] Property missing ID in {area}")
            return None

        # Basic info
        item['source'] = 'rightmove'
        item['property_id'] = str(prop_id)
        item['url'] = f"https://www.rightmove.co.uk{prop.get('propertyUrl', '')}"
        item['area'] = area

        # Price with validation
        price_data = prop.get('price', {})
        price = price_data.get('amount', 0)
        frequency = price_data.get('frequency', 'monthly')

        if not price:
            self._track_missing('price', area)
            self.logger.debug(f"[VALIDATION] Property {prop_id} missing price")

        if frequency == 'weekly':
            item['price_pw'] = price
            item['price_pcm'] = int(price * 52 / 12) if price else 0
            item['price_period'] = 'pw'
        else:
            item['price_pcm'] = price
            item['price_pw'] = int(price * 12 / 52) if price else 0
            item['price_period'] = 'pcm'

        item['price'] = price

        # Location
        item['address'] = prop.get('displayAddress', '')
        if not item['address']:
            self._track_missing('address', area)

        location = prop.get('location', {})
        item['latitude'] = location.get('latitude')
        item['longitude'] = location.get('longitude')

        if not item['latitude'] or not item['longitude']:
            self._track_missing('coordinates', area)

        # Extract postcode from address - try full postcode first, then outcode only
        address_upper = item['address'].upper()

        # Try full postcode (e.g., "SW1X 8AB")
        full_postcode_match = re.search(
            r'([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})',
            address_upper
        )
        if full_postcode_match:
            item['postcode'] = full_postcode_match.group(1).replace(' ', '')
        else:
            # Fallback: try outcode only (e.g., "SW1X" at end of address)
            # Look for pattern at end of address or before comma
            outcode_match = re.search(
                r'\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(?:,|$)',
                address_upper
            )
            item['postcode'] = outcode_match.group(1) if outcode_match else None

        if not item['postcode']:
            self._track_missing('postcode', area)

        # Property details
        item['bedrooms'] = prop.get('bedrooms')
        item['bathrooms'] = prop.get('bathrooms')
        item['property_type'] = prop.get('propertySubType', prop.get('propertyType', ''))

        if item['bedrooms'] is None:
            self._track_missing('bedrooms', area)

        # Size
        size = prop.get('displaySize', '')
        if size:
            # R3: tolerate periods/spaces in the real search format "675 sq. ft."
            # (the old ([\d,]+)\s*sq\s*ft was period-blind and silently dropped sqft).
            sqft_match = re.search(r'([\d,]+)\s*sq[\s.]*ft', size, re.I)
            item['size_sqft'] = int(sqft_match.group(1).replace(',', '')) if sqft_match else None
        else:
            item['size_sqft'] = None
            self._track_missing('size_sqft', area)

        # Status
        status = prop.get('displayStatus', '')
        item['let_agreed'] = 'let agreed' in status.lower() if status else False

        # Agent
        customer = prop.get('customer', {})
        item['agent_name'] = customer.get('branchDisplayName', '')
        item['agent_phone'] = customer.get('contactTelephone', '')

        # Dates
        item['added_date'] = prop.get('addedOrReduced', '')
        item['scraped_at'] = datetime.utcnow().isoformat()

        # Additional fields
        item['summary'] = prop.get('summary', '')
        item['features'] = prop.get('featureList', [])

        return item

    def _track_missing(self, field: str, area: str):
        """Track missing fields for data quality reporting."""
        if field not in self.stats['missing_fields']:
            self.stats['missing_fields'][field] = 0
        self.stats['missing_fields'][field] += 1

    def parse_detail(self, response):
        """Parse property detail page for square footage and description."""
        item = response.meta.get('item', {})
        prop_id = item.get('property_id', 'unknown')
        request_time = time.time() - response.meta.get('request_start', time.time())

        self.stats['details_fetched'] += 1

        if response.status != 200:
            self.logger.warning(f"[DETAIL-ERROR] {prop_id}: status {response.status}")
            yield PropertyItem(**item)
            return

        # Try to extract __NEXT_DATA__ JSON
        script = response.css('script#__NEXT_DATA__::text').get()
        sqft = None
        description = None

        if script:
            try:
                data = json.loads(script)
                props = data.get('props', {})
                page_props = props.get('pageProps', {})
                property_data = page_props.get('propertyData', {})

                # Check sizings for sqft
                sizings = property_data.get('sizings', [])
                for sizing in sizings:
                    if sizing.get('unit') == 'sqft':
                        sqft = sizing.get('minimumSize') or sizing.get('maximumSize')
                        break

                # Also try sizes object
                if not sqft:
                    sizes = property_data.get('sizes', {})
                    sqft = sizes.get('totalFloorAreaSqft')

                # Try text content for sqft
                if not sqft:
                    text = property_data.get('text', {})
                    desc_text = text.get('description', '') + ' ' + text.get('propertyPhrase', '')
                    sqft_match = re.search(r'(\d{3,5})\s*(?:sq\.?\s*ft|square\s*feet)', desc_text, re.I)
                    if sqft_match:
                        sqft = int(sqft_match.group(1))

                # Extract description
                text_data = property_data.get('text', {})
                description = text_data.get('description', '')

                # Extract floorplan URL if enabled
                if self.fetch_floorplans:
                    floorplan_url = self._extract_floorplan_url(property_data, response.text)
                    if floorplan_url:
                        item['floorplan_url'] = floorplan_url
                        self.stats['floorplans_found'] += 1
                        self.logger.debug(f"[FLOORPLAN] {prop_id}: {floorplan_url[:60]}...")

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.logger.debug(f"[DETAIL-PARSE] {prop_id}: JSON parse issue - {e}")

        # Fallback: search in page text for sqft
        if not sqft:
            page_text = response.text
            sqft_patterns = [
                r'(\d{3,5})\s*sq\.?\s*ft',
                r'(\d{3,5})\s*square\s*feet',
                r'(\d{3,5})\s*sqft',
            ]
            for pattern in sqft_patterns:
                match = re.search(pattern, page_text, re.I)
                if match:
                    sqft = int(match.group(1))
                    break

        # Extract floorplan URL (works even without JSON)
        if self.fetch_floorplans and not item.get('floorplan_url'):
            floorplan_url = self._extract_floorplan_url({}, response.text)
            if floorplan_url:
                item['floorplan_url'] = floorplan_url
                self.stats['floorplans_found'] += 1
                self.logger.debug(f"[FLOORPLAN] {prop_id}: {floorplan_url[:60]}...")

        # Run inline OCR on floorplan if we have one and didn't find sqft yet
        if self.floorplan_extractor and item.get('floorplan_url') and not sqft:
            ocr_result = self._extract_floorplan_ocr(item['floorplan_url'])
            if ocr_result:
                if ocr_result.get('size_sqft'):
                    sqft = ocr_result['size_sqft']
                    self.logger.debug(f"[OCR] {prop_id}: extracted {sqft} sqft from floorplan")
                # Copy floor data fields
                for field in ['floor_count', 'property_levels', 'has_basement',
                              'has_lower_ground', 'has_ground', 'has_mezzanine',
                              'has_first_floor', 'has_second_floor', 'has_third_floor',
                              'has_fourth_plus', 'has_roof_terrace']:
                    if ocr_result.get(field):
                        item[field] = ocr_result[field]

        # Update item with enriched data
        if sqft and 100 < sqft < 50000:  # Sanity check
            item['size_sqft'] = sqft
            self.stats['sqft_found'] += 1
            self.logger.debug(
                f"[DETAIL] {prop_id}: {sqft} sqft | "
                f"{item.get('bedrooms')}bed @ £{item.get('price_pcm', 0):,}"
            )

        if description and len(description) > 50:
            # Clean and truncate description
            description = re.sub(r'\s+', ' ', description).strip()
            if len(description) > 5000:
                description = description[:5000]
            item['summary'] = description

        # Progress logging
        if self.stats['details_fetched'] % 50 == 0:
            sqft_pct = (self.stats['sqft_found'] / self.stats['details_fetched'] * 100) if self.stats['details_fetched'] else 0
            self.logger.info(
                f"[DETAILS-PROGRESS] {self.stats['details_fetched']} fetched | "
                f"Sqft found: {self.stats['sqft_found']} ({sqft_pct:.0f}%)"
            )

        yield PropertyItem(**item)

    def _extract_floorplan_ocr(self, floorplan_url: str) -> dict | None:
        """Download floorplan and run OCR to extract floor data and sqft."""
        if not self.floorplan_extractor or not floorplan_url:
            return None

        try:
            # Download floorplan image
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'image/*,*/*;q=0.8',
            }

            img_response = requests.get(floorplan_url, headers=headers, timeout=15)

            if img_response.status_code != 200 or len(img_response.content) < 1000:
                return None

            # Run OCR
            floorplan_data = self.floorplan_extractor.extract_from_bytes(img_response.content)

            if not floorplan_data:
                return None

            result = {}

            # Extract sqft
            if floorplan_data.total_sqft and floorplan_data.total_sqft > 100:
                result['size_sqft'] = floorplan_data.total_sqft

            # Extract floor data
            if floorplan_data.floor_data:
                fd = floorplan_data.floor_data
                result['has_basement'] = fd.has_basement
                result['has_lower_ground'] = fd.has_lower_ground
                result['has_ground'] = fd.has_ground
                result['has_mezzanine'] = fd.has_mezzanine
                result['has_first_floor'] = fd.has_first_floor
                result['has_second_floor'] = fd.has_second_floor
                result['has_third_floor'] = fd.has_third_floor
                result['has_fourth_plus'] = fd.has_fourth_plus
                result['has_roof_terrace'] = fd.has_roof_terrace
                result['floor_count'] = fd.floor_count
                result['property_levels'] = fd.property_levels

            if result:
                self.stats['floorplans_ocr_success'] += 1

            return result if result else None

        except Exception as e:
            self.logger.debug(f"[OCR] Error processing {floorplan_url[:50]}: {e}")
            return None

    def _extract_floorplan_url(self, property_data: dict, response_text: str = None) -> str | None:
        """Extract floorplan URL from Rightmove property data.

        Rightmove floorplan URLs contain '_FLP_' in the filename:
        https://media.rightmove.co.uk/XXXk/XXXXXX/ID/XXXXX_FLP_00_0000.jpeg

        Note: Detail pages don't use __NEXT_DATA__, so we also search HTML directly.

        Issue #25 FIX: All URLs are validated before returning.
        """
        # Strategy 1: Search HTML for FLP URLs directly (most reliable for detail pages)
        if response_text:
            flp_pattern = r'https://media\.rightmove\.co\.uk/[^"\'<>\s]*_FLP_[^"\'<>\s]*'
            matches = re.findall(flp_pattern, response_text, re.I)
            if matches:
                # Filter out thumbnails (prefer full size images)
                full_size = [m for m in matches if '_max_' not in m]
                candidate = full_size[0] if full_size else matches[0]
                validated = validate_floorplan_url(candidate)
                if validated:
                    return validated

        # Strategy 2: Check JSON data (works for search results)
        images = property_data.get('images', [])
        for img in images:
            url = img.get('url', '') or img.get('srcUrl', '')
            if '_FLP_' in url or '_flp_' in url.lower():
                validated = validate_floorplan_url(url)
                if validated:
                    return validated

        # Strategy 3: Check floorplans array directly
        floorplans = property_data.get('floorplans', [])
        if floorplans:
            for fp in floorplans:
                url = fp.get('url', '') or fp.get('srcUrl', '')
                if url:
                    validated = validate_floorplan_url(url)
                    if validated:
                        return validated

        # Strategy 4: Check media array
        media = property_data.get('media', [])
        for item in media:
            if item.get('type', '').lower() == 'floorplan':
                url = item.get('url', '') or item.get('srcUrl', '')
                if url:
                    validated = validate_floorplan_url(url)
                    if validated:
                        return validated

        return None

    def closed(self, reason):
        """Log comprehensive summary when spider closes."""
        elapsed = time.time() - self.stats['start_time']
        rate = self.stats['requests_made'] / elapsed if elapsed > 0 else 0

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("SCRAPING COMPLETE")
        self.logger.info("=" * 70)

        # Overall stats
        self.logger.info(f"[SUMMARY] Reason: {reason}")
        self.logger.info(f"[SUMMARY] Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        self.logger.info(f"[SUMMARY] Total listings scraped: {self.stats['total']}")
        self.logger.info(f"[SUMMARY] Let agreed: {self.stats['let_agreed_count']}")
        self.logger.info(f"[SUMMARY] Active listings: {self.stats['total'] - self.stats['let_agreed_count']}")

        # Report sqft stats if detail fetching was enabled
        if self.fetch_details and self.stats['details_fetched'] > 0:
            sqft_pct = (self.stats['sqft_found'] / self.stats['details_fetched'] * 100)
            self.logger.info(f"[SUMMARY] Details fetched: {self.stats['details_fetched']}")
            self.logger.info(f"[SUMMARY] Sqft found: {self.stats['sqft_found']} ({sqft_pct:.1f}%)")

        # Report floorplan/OCR stats if enabled
        if self.fetch_floorplans and self.stats['floorplans_found'] > 0:
            floorplan_pct = (self.stats['floorplans_found'] / self.stats['details_fetched'] * 100) if self.stats['details_fetched'] else 0
            ocr_pct = (self.stats['floorplans_ocr_success'] / self.stats['floorplans_found'] * 100) if self.stats['floorplans_found'] else 0
            self.logger.info(f"[SUMMARY] Floorplans found: {self.stats['floorplans_found']} ({floorplan_pct:.1f}%)")
            self.logger.info(f"[SUMMARY] OCR success: {self.stats['floorplans_ocr_success']} ({ocr_pct:.1f}%)")

        # Request stats
        self.logger.info("")
        self.logger.info(f"[REQUESTS] Made: {self.stats['requests_made']}")
        self.logger.info(f"[REQUESTS] Failed: {self.stats['requests_failed']}")
        self.logger.info(f"[REQUESTS] Rate: {rate:.2f} req/s")
        self.logger.info(f"[REQUESTS] Data: {self.stats['bytes_downloaded']/1024/1024:.2f} MB")

        # Price stats
        if self.stats['prices']:
            prices = sorted(self.stats['prices'])
            avg = sum(prices) // len(prices)
            median = prices[len(prices) // 2]
            p10 = prices[int(len(prices) * 0.1)]
            p90 = prices[int(len(prices) * 0.9)]

            self.logger.info("")
            self.logger.info(f"[PRICES] Average: £{avg:,}/pcm")
            self.logger.info(f"[PRICES] Median: £{median:,}/pcm")
            self.logger.info(f"[PRICES] Range: £{prices[0]:,} - £{prices[-1]:,}/pcm")
            self.logger.info(f"[PRICES] P10-P90: £{p10:,} - £{p90:,}/pcm")

        # Per-area breakdown
        self.logger.info("")
        self.logger.info("[BY AREA]")
        for area, data in sorted(self.stats['by_area'].items()):
            coverage = (data['count'] / data['total_available'] * 100) if data['total_available'] else 0
            self.logger.info(
                f"  {area}: {data['count']}/{data['total_available']} "
                f"({coverage:.0f}%) across {data['pages']} pages"
            )

        # Data quality
        if self.stats['missing_fields']:
            self.logger.info("")
            self.logger.info("[DATA QUALITY] Missing fields:")
            for field, count in sorted(self.stats['missing_fields'].items(), key=lambda x: -x[1]):
                pct = count / self.stats['total'] * 100 if self.stats['total'] else 0
                self.logger.info(f"  {field}: {count} ({pct:.1f}%)")

        self.logger.info("=" * 70)
        self.logger.info("")
