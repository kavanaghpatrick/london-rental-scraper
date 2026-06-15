"""
Foxtons Spider - Scrapes rental listings from Foxtons.co.uk

Uses __NEXT_DATA__ JSON embedded in pages for efficient data extraction.
Foxtons returns up to 100 properties per page.

Usage:
    scrapy crawl foxtons -o output/foxtons.json
    scrapy crawl foxtons -a areas=Kensington,Chelsea -a max_pages=5
    scrapy crawl foxtons -a fetch_floorplans=true  # Enable inline OCR

When fetch_floorplans=true, downloads floorplan images and runs OCR to extract:
- size_sqft (if not already available)
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
from property_scraper.items import PropertyItem

# OCR support for inline floorplan extraction
try:
    from property_scraper.utils.floorplan_extractor import FloorplanExtractor
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class FoxtonsSpider(scrapy.Spider):
    """Spider for Foxtons rental listings."""

    name = 'foxtons'
    allowed_domains = ['foxtons.co.uk']

    # Areas from central registry - full Prime Central London coverage
    DEFAULT_AREAS = [
        'Belgravia', 'Chelsea', 'Kensington', 'South-Kensington',
        'Knightsbridge', 'Notting-Hill', 'Earls-Court', 'Fulham',
        'Hampstead', 'St-Johns-Wood', 'Mayfair', 'Marylebone'
    ]

    # Map display names to URL slugs
    AREA_SLUGS = {
        'Belgravia': 'belgravia',
        'Chelsea': 'chelsea',
        'Kensington': 'kensington',
        'South-Kensington': 'south-kensington',
        'Knightsbridge': 'knightsbridge',
        'Notting-Hill': 'notting-hill',
        'Earls-Court': 'earls-court',
        'Fulham': 'fulham',
        'Hampstead': 'hampstead',
        'St-Johns-Wood': 'st-johns-wood',
        'Mayfair': 'mayfair',
        'Marylebone': 'marylebone',
    }

    def __init__(self, areas=None, max_pages=None, fetch_floorplans=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if areas:
            self.areas = [a.strip().lower() for a in areas.split(',')]
        else:
            self.areas = self.DEFAULT_AREAS

        # Parse max_pages (None = unlimited)
        if max_pages is None or str(max_pages).lower() in ('none', '0', ''):
            self.max_pages = None
        else:
            try:
                self.max_pages = int(max_pages)
            except (ValueError, TypeError):
                self.logger.warning(f"[CONFIG] Invalid max_pages '{max_pages}', using unlimited")
                self.max_pages = None

        # Inline OCR for floorplans
        self.fetch_floorplans = str(fetch_floorplans).lower() in ('true', '1', 'yes')
        if self.fetch_floorplans and OCR_AVAILABLE:
            self.floorplan_extractor = FloorplanExtractor()
        else:
            self.floorplan_extractor = None

        # Stats tracking
        self.stats = {
            'total': 0,
            'by_area': {},
            'prices': [],
            'start_time': time.time(),
            'requests_made': 0,
            'sqft_found': 0,
            'floorplans_found': 0,
            'floorplans_ocr_success': 0,
        }

        self.logger.info("=" * 70)
        self.logger.info("FOXTONS SPIDER INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"[CONFIG] Areas: {', '.join(self.areas)}")
        self.logger.info(f"[CONFIG] Max pages per area: {self.max_pages or 'unlimited'}")
        self.logger.info(f"[CONFIG] Fetch floorplans (inline OCR): {self.fetch_floorplans}")
        if self.fetch_floorplans and not OCR_AVAILABLE:
            self.logger.warning("[CONFIG] OCR requested but pytesseract not available!")
        self.logger.info("=" * 70)

    def _extract_floorplan_ocr(self, floorplan_url, response):
        """Download floorplan and run OCR to extract floor data.

        Uses the response's cookies/session for authenticated CDN access.
        """
        if not self.floorplan_extractor or not floorplan_url:
            return None

        try:
            # Use cookies from the scrapy response for CDN auth
            cookies = {c.name: c.value for c in response.headers.getlist('Set-Cookie')} if response else {}

            # Download floorplan image
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Referer': 'https://www.foxtons.co.uk/',
                'Accept': 'image/*,*/*;q=0.8',
            }

            img_response = requests.get(floorplan_url, headers=headers, cookies=cookies, timeout=15)

            if img_response.status_code != 200 or len(img_response.content) < 1000:
                return None

            # Check for XML error response (Access Denied)
            if img_response.content[:5] == b'<?xml':
                self.logger.debug(f"[OCR] CDN access denied for {floorplan_url[:50]}")
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

    def start_requests(self):
        """Generate initial requests for all target areas."""
        self.logger.info(f"[START] Launching {len(self.areas)} area scrapers...")

        for i, area in enumerate(self.areas):
            self.stats['by_area'][area] = {'count': 0, 'pages': 0}
            # Use slug for URL, keep display name for data
            slug = self.AREA_SLUGS.get(area, area.lower())
            url = f'https://www.foxtons.co.uk/properties-to-rent/{slug}/'

            self.logger.info(f"[REQUEST] [{i+1}/{len(self.areas)}] Starting: {area}")

            yield scrapy.Request(
                url,
                callback=self.parse_search,
                meta={
                    'area': area,
                    'page': 1,
                    'request_start': time.time()
                },
                dont_filter=True,
                errback=self.handle_error
            )

    def handle_error(self, failure):
        """Handle request failures."""
        request = failure.request
        area = request.meta.get('area', 'unknown')
        self.logger.error(f"[ERROR] Request failed for {area}: {failure.value}")

    def parse_search(self, response):
        """Parse search results page."""
        area = response.meta['area']
        page = response.meta['page']
        request_time = time.time() - response.meta.get('request_start', time.time())

        self.stats['requests_made'] += 1

        self.logger.info(
            f"[RESPONSE] {area} p{page} | "
            f"Status: {response.status} | "
            f"Size: {len(response.body)/1024:.1f}KB | "
            f"Time: {request_time:.2f}s"
        )

        if response.status != 200:
            self.logger.warning(f"[HTTP-ERROR] {area} returned status {response.status}")
            return

        # Extract __NEXT_DATA__ JSON
        script = response.css('script#__NEXT_DATA__::text').get()

        if not script:
            self.logger.error(f"[PARSE-ERROR] No __NEXT_DATA__ found for {area}")
            return

        try:
            data = json.loads(script)
        except json.JSONDecodeError as e:
            self.logger.error(f"[JSON-ERROR] Failed to parse JSON for {area}: {e}")
            return

        # Navigate to properties
        # Path: props.pageProps.pageData.data.data
        try:
            page_data = data['props']['pageProps']['pageData']['data']
            properties = page_data.get('data', [])
        except (KeyError, TypeError) as e:
            self.logger.error(f"[STRUCTURE-ERROR] Unexpected data structure for {area}: {e}")
            return

        total_count = len(properties)
        self.logger.info(f"[DISCOVERY] {area} p{page}: {total_count} properties found")

        # Parse each property
        parsed_count = 0
        for prop in properties:
            item = self.parse_property(prop, area, response)
            if item:
                parsed_count += 1
                self.stats['total'] += 1
                self.stats['by_area'][area]['count'] += 1

                if item.get('price_pcm'):
                    self.stats['prices'].append(item['price_pcm'])

                yield item

        self.stats['by_area'][area]['pages'] += 1

        self.logger.info(
            f"[PAGE] {area} p{page}: {parsed_count}/{total_count} parsed | "
            f"Running total: {self.stats['by_area'][area]['count']}"
        )

        # Issue #15 FIX: More robust pagination detection
        # Check for explicit pagination metadata in JSON, with fallback to page count heuristic
        # Foxtons typically returns 100 per page, but we shouldn't assume this
        try:
            # Foxtons exposes pagination directly on page_data: total, page, pageSize, totalPages
            total_pages_meta = page_data.get('totalPages', None)
            pagination_meta = page_data.get('pagination', {}) or page_data.get('meta', {})
            has_next_explicit = pagination_meta.get('hasNext', None) or pagination_meta.get('has_next', None)
            total_results = (page_data.get('total', None)
                             or pagination_meta.get('total', None)
                             or pagination_meta.get('totalCount', None))
            page_size = page_data.get('pageSize', None)

            if total_pages_meta is not None:
                # Most reliable: explicit total page count from Foxtons
                should_continue = page < int(total_pages_meta)
            elif has_next_explicit is not None:
                # Use explicit flag if available
                should_continue = has_next_explicit
            elif total_results is not None:
                # Calculate from total (use reported pageSize, fall back to 100)
                per_page = int(page_size) if page_size else 100
                total_pages = (int(total_results) + per_page - 1) // per_page
                should_continue = page < total_pages
            else:
                # Fallback: continue if we got any results (more conservative than >= 100)
                # This handles cases where Foxtons changes their page size
                should_continue = total_count > 0
        except (KeyError, TypeError, ValueError):
            # Safe fallback: continue if we got results
            should_continue = total_count > 0

        # Apply max_pages limit
        if self.max_pages is not None and page >= self.max_pages:
            should_continue = False

        if should_continue:
            next_page = page + 1
            # Use the same URL slug as the initial request for consistency
            slug = self.AREA_SLUGS.get(area, area.lower())
            next_url = f'https://www.foxtons.co.uk/properties-to-rent/{slug}/?page={next_page}'

            self.logger.debug(f"[PAGINATION] {area}: Following to page {next_page}")

            yield scrapy.Request(
                next_url,
                callback=self.parse_search,
                meta={
                    'area': area,
                    'page': next_page,
                    'request_start': time.time()
                },
                dont_filter=True,
                errback=self.handle_error
            )
        else:
            reason = "max pages reached" if self.max_pages and page >= self.max_pages else "no more results"
            self.logger.info(f"[COMPLETE] {area}: Stopped at page {page} ({reason})")

    def parse_property(self, prop: dict, area: str, response=None) -> PropertyItem:
        """Parse a single property from Foxtons data."""
        item = PropertyItem()

        prop_ref = prop.get('propertyReference', '')
        if not prop_ref:
            self.logger.warning(f"[VALIDATION] Property missing reference in {area}")
            return None

        # Basic info
        item['source'] = 'foxtons'
        item['property_id'] = prop_ref
        item['area'] = area

        # Extract postcode district for URL (Foxtons uses /properties-to-rent/{postcode}/{ref})
        # Prefer the reliable postcodeShort field; streetName rarely contains a postcode.
        address = prop.get('streetName', '')
        postcode_short = prop.get('postcodeShort')
        if postcode_short:
            postcode_district = postcode_short.upper()
        else:
            postcode_match = re.search(r'([A-Z]{1,2}\d{1,2}[A-Z]?)', address.upper())
            postcode_district = postcode_match.group(1) if postcode_match else None

        if postcode_district:
            item['url'] = f"https://www.foxtons.co.uk/properties-to-rent/{postcode_district}/{prop_ref}"
        else:
            # Fallback: try to use area as postcode hint
            item['url'] = f"https://www.foxtons.co.uk/properties-to-rent/{area.lower()}/{prop_ref}"

        # Price - Foxtons provides pricePcm as string
        price_pcm_str = prop.get('pricePcm', '0')
        try:
            price_pcm = int(float(price_pcm_str))
        except (ValueError, TypeError):
            price_pcm = 0

        item['price_pcm'] = price_pcm
        item['price_pw'] = int(price_pcm * 12 / 52) if price_pcm else 0
        item['price'] = price_pcm
        item['price_period'] = 'pcm'

        # Location
        item['address'] = prop.get('streetName', '')
        location = prop.get('location', {})
        item['latitude'] = location.get('lat')
        item['longitude'] = location.get('lon')

        # Extract postcode - prefer postcodeShort from JSON, fallback to address parsing
        postcode_short = prop.get('postcodeShort')
        if postcode_short:
            item['postcode'] = postcode_short.upper()
        else:
            # Fallback: try to extract from address
            postcode_match = re.search(
                r'([A-Z]{1,2}\d{1,2}[A-Z]?)',
                item['address'].upper()
            )
            item['postcode'] = postcode_match.group(1) if postcode_match else None

        # Property details
        item['bedrooms'] = prop.get('bedrooms')
        item['bathrooms'] = prop.get('bathrooms')
        item['property_type'] = prop.get('typeGroup', '')

        # Size from propertyBlob
        prop_blob = prop.get('propertyBlob', {}) or {}
        floor_area = prop_blob.get('floorArea')
        if floor_area:
            try:
                item['size_sqft'] = int(float(floor_area))
                self.stats['sqft_found'] += 1
            except (ValueError, TypeError):
                item['size_sqft'] = None
        else:
            item['size_sqft'] = None

        # Extract floorplan URL from assets
        asset_info = prop_blob.get('assetInfo', {}) or {}
        assets = asset_info.get('assets', {}) or {}
        floorplan_data = assets.get('floorplan', {}) or {}

        floorplan_url = None
        # Prefer large PNG, then small PNG
        if floorplan_data.get('large') and floorplan_data['large'].get('filename'):
            filename = floorplan_data['large']['filename']
            floorplan_url = f"https://assets.foxtons.co.uk/{filename}"
        elif floorplan_data.get('small') and floorplan_data['small'].get('filename'):
            filename = floorplan_data['small']['filename']
            floorplan_url = f"https://assets.foxtons.co.uk/{filename}"

        if floorplan_url:
            item['floorplan_url'] = floorplan_url
            self.stats['floorplans_found'] += 1

            # Run inline OCR if enabled and sqft not already found
            if self.fetch_floorplans and not item.get('size_sqft'):
                ocr_result = self._extract_floorplan_ocr(floorplan_url, response)
                if ocr_result:
                    if ocr_result.get('size_sqft') and not item.get('size_sqft'):
                        item['size_sqft'] = ocr_result['size_sqft']
                        self.stats['sqft_found'] += 1
                    # Copy floor data fields
                    for field in ['floor_count', 'property_levels', 'has_basement',
                                  'has_lower_ground', 'has_ground', 'has_mezzanine',
                                  'has_first_floor', 'has_second_floor', 'has_third_floor',
                                  'has_fourth_plus', 'has_roof_terrace']:
                        if ocr_result.get(field):
                            item[field] = ocr_result[field]

        # Agent info
        item['agent_name'] = prop.get('officeName', 'Foxtons')
        item['agent_phone'] = ''  # Not available in search results

        # Status
        item['let_agreed'] = False  # Foxtons filters these out

        # Dates
        item['added_date'] = ''
        item['scraped_at'] = datetime.utcnow().isoformat()

        # Extract structured amenity flags from Foxtons data
        amenities = {}
        amenities['has_garden'] = bool(prop.get('hasGarden'))
        amenities['has_patio'] = bool(prop.get('hasPatio'))
        amenities['has_balcony'] = bool(prop.get('hasBalcony'))
        amenities['has_roof_terrace'] = bool(prop.get('hasRoofTerrace'))
        amenities['has_outdoor_space'] = any([
            amenities['has_garden'], amenities['has_patio'],
            amenities['has_balcony'], amenities['has_roof_terrace']
        ])

        # Build summary from bullet points if available
        prop_blob = prop.get('propertyBlob', {}) or {}
        bullet_points = prop_blob.get('bulletPoints', []) or []
        description = prop_blob.get('description', '') or ''
        description_short = prop_blob.get('descriptionShort', '') or ''

        # Use available text for summary
        if bullet_points:
            item['summary'] = ' | '.join(bullet_points)
        elif description_short:
            item['summary'] = description_short
        elif description:
            item['summary'] = description[:1000]
        else:
            item['summary'] = ''

        # Store features as JSON-serializable dict (json already imported at top)
        item['features'] = json.dumps({k: v for k, v in amenities.items() if v})

        return item

    def closed(self, reason):
        """Log summary when spider closes."""
        elapsed = time.time() - self.stats['start_time']
        sqft_pct = (self.stats['sqft_found'] / self.stats['total'] * 100) if self.stats['total'] else 0
        floorplan_pct = (self.stats['floorplans_found'] / self.stats['total'] * 100) if self.stats['total'] else 0

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("FOXTONS SCRAPING COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"[SUMMARY] Reason: {reason}")
        self.logger.info(f"[SUMMARY] Duration: {elapsed:.1f}s")
        self.logger.info(f"[SUMMARY] Total listings: {self.stats['total']}")
        self.logger.info(f"[SUMMARY] With sqft: {self.stats['sqft_found']} ({sqft_pct:.0f}%)")
        self.logger.info(f"[SUMMARY] With floorplan: {self.stats['floorplans_found']} ({floorplan_pct:.0f}%)")
        if self.fetch_floorplans:
            ocr_success_pct = (self.stats['floorplans_ocr_success'] / self.stats['floorplans_found'] * 100) if self.stats['floorplans_found'] else 0
            self.logger.info(f"[SUMMARY] OCR success: {self.stats['floorplans_ocr_success']} ({ocr_success_pct:.0f}%)")
        self.logger.info(f"[SUMMARY] Requests made: {self.stats['requests_made']}")

        if self.stats['prices']:
            prices = sorted(self.stats['prices'])
            avg = sum(prices) // len(prices)
            median = prices[len(prices) // 2]
            self.logger.info(f"[PRICES] Average: £{avg:,}/pcm")
            self.logger.info(f"[PRICES] Median: £{median:,}/pcm")
            self.logger.info(f"[PRICES] Range: £{prices[0]:,} - £{prices[-1]:,}/pcm")

        self.logger.info("[BY AREA]")
        for area, data in sorted(self.stats['by_area'].items()):
            self.logger.info(f"  {area}: {data['count']} across {data['pages']} pages")

        self.logger.info("=" * 70)
