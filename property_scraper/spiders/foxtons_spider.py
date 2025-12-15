"""
Foxtons Spider - Scrapes rental listings from Foxtons.co.uk

Uses __NEXT_DATA__ JSON embedded in pages for efficient data extraction.
Foxtons returns up to 100 properties per page.

Usage:
    scrapy crawl foxtons -o output/foxtons.json
    scrapy crawl foxtons -a areas=Kensington,Chelsea -a max_pages=5
"""

import scrapy
import json
import re
import time
from datetime import datetime
from property_scraper.items import PropertyItem


class FoxtonsSpider(scrapy.Spider):
    """Spider for Foxtons rental listings."""

    name = 'foxtons'
    allowed_domains = ['foxtons.co.uk']

    # Areas matching Rightmove format (Title-Case)
    DEFAULT_AREAS = [
        'Belgravia', 'Chelsea', 'Kensington', 'South-Kensington',
        'Knightsbridge', 'Notting-Hill'
    ]

    # Map display names to URL slugs
    AREA_SLUGS = {
        'Belgravia': 'belgravia',
        'Chelsea': 'chelsea',
        'Kensington': 'kensington',
        'South-Kensington': 'south-kensington',
        'Knightsbridge': 'knightsbridge',
        'Notting-Hill': 'notting-hill',
    }

    def __init__(self, areas=None, max_pages=None, *args, **kwargs):
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

        # Stats tracking
        self.stats = {
            'total': 0,
            'by_area': {},
            'prices': [],
            'start_time': time.time(),
            'requests_made': 0,
            'sqft_found': 0,
            'floorplans_found': 0,
        }

        self.logger.info("=" * 70)
        self.logger.info("FOXTONS SPIDER INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"[CONFIG] Areas: {', '.join(self.areas)}")
        self.logger.info(f"[CONFIG] Max pages per area: {self.max_pages or 'unlimited'}")
        self.logger.info("=" * 70)

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
            item = self.parse_property(prop, area)
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
            pagination_meta = page_data.get('pagination', {}) or page_data.get('meta', {})
            has_next_explicit = pagination_meta.get('hasNext', None) or pagination_meta.get('has_next', None)
            total_results = pagination_meta.get('total', None) or pagination_meta.get('totalCount', None)

            if has_next_explicit is not None:
                # Use explicit flag if available
                should_continue = has_next_explicit
            elif total_results is not None:
                # Calculate from total if available (assume 100 per page)
                total_pages = (int(total_results) + 99) // 100
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
            next_url = f'https://www.foxtons.co.uk/properties-to-rent/{area}/?page={next_page}'

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

    def parse_property(self, prop: dict, area: str) -> PropertyItem:
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
        address = prop.get('streetName', '')
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
