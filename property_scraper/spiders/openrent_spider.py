"""
OpenRent Spider - Scrapes rental listings from OpenRent.co.uk

OpenRent is heavily JavaScript-rendered, so this spider uses scrapy-playwright
for browser-based scraping.

Requires:
    pip install scrapy-playwright
    playwright install chromium

Usage:
    scrapy crawl openrent -o output/openrent.json
    scrapy crawl openrent -a areas=Kensington,Chelsea -a max_pages=5

Note: Enable Playwright in settings.py by uncommenting DOWNLOAD_HANDLERS
"""

import scrapy
import json
import re
import time
from datetime import datetime
from urllib.parse import urlencode
from property_scraper.items import PropertyItem


class OpenRentSpider(scrapy.Spider):
    """Spider for OpenRent rental listings using Playwright."""

    name = 'openrent'
    allowed_domains = ['openrent.co.uk']

    # OpenRent area search terms
    DEFAULT_AREAS = [
        'Belgravia, London', 'Chelsea, London', 'Kensington, London',
        'South Kensington, London', 'Knightsbridge, London', 'Notting Hill, London'
    ]

    # OpenRent shows 20 properties per page
    ITEMS_PER_PAGE = 20

    def __init__(self, areas=None, max_pages=10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if areas:
            self.areas = [a.strip() for a in areas.split(',')]
        else:
            self.areas = self.DEFAULT_AREAS

        try:
            self.max_pages = int(max_pages)
        except (ValueError, TypeError):
            self.logger.warning(f"[CONFIG] Invalid max_pages '{max_pages}', using default 10")
            self.max_pages = 10

        # Stats
        self.stats = {
            'total': 0,
            'by_area': {},
            'prices': [],
            'start_time': time.time(),
            'requests_made': 0,
        }

        self.logger.info("=" * 70)
        self.logger.info("OPENRENT SPIDER INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"[CONFIG] Areas: {', '.join(self.areas)}")
        self.logger.info(f"[CONFIG] Max pages per area: {self.max_pages}")
        self.logger.info("[CONFIG] Using Playwright for JavaScript rendering")
        self.logger.info("=" * 70)

    def start_requests(self):
        """Generate initial requests for all target areas."""
        self.logger.info(f"[START] Launching {len(self.areas)} area scrapers...")

        for i, area in enumerate(self.areas):
            area_key = area.replace(', London', '').replace(' ', '-').lower()
            self.stats['by_area'][area_key] = {'count': 0, 'pages': 0}

            # Build OpenRent search URL
            params = {
                'term': area,
                'viewType': 'LIST',
            }
            url = f'https://www.openrent.co.uk/properties-to-rent?{urlencode(params)}'

            self.logger.info(f"[REQUEST] [{i+1}/{len(self.areas)}] Starting: {area}")

            # Use Playwright for JavaScript rendering
            yield scrapy.Request(
                url,
                callback=self.parse_search,
                meta={
                    'playwright': True,
                    'playwright_include_page': True,
                    'playwright_page_methods': [
                        {'method': 'wait_for_selector', 'args': ['[data-listing-id]'], 'kwargs': {'timeout': 10000}},
                    ],
                    'area': area,
                    'area_key': area_key,
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

    async def parse_search(self, response):
        """Parse search results page using Playwright."""
        area = response.meta['area']
        area_key = response.meta['area_key']
        page = response.meta['page']
        request_time = time.time() - response.meta.get('request_start', time.time())

        self.stats['requests_made'] += 1

        # Get the Playwright page object for interaction
        playwright_page = response.meta.get('playwright_page')

        self.logger.info(
            f"[RESPONSE] {area} p{page} | "
            f"Status: {response.status} | "
            f"Size: {len(response.body)/1024:.1f}KB | "
            f"Time: {request_time:.2f}s"
        )

        if response.status != 200:
            self.logger.warning(f"[HTTP-ERROR] {area} returned status {response.status}")
            if playwright_page:
                await playwright_page.close()
            return

        # Extract listing IDs from the rendered page
        listing_elements = response.css('[data-listing-id]')
        self.logger.info(f"[DISCOVERY] {area} p{page}: {len(listing_elements)} listing elements found")

        # Parse each property card
        parsed_count = 0
        for element in listing_elements:
            listing_id = element.attrib.get('data-listing-id')
            if not listing_id:
                continue

            # Extract data from the card
            item = self.parse_property_card(element, listing_id, area_key)
            if item:
                parsed_count += 1
                self.stats['total'] += 1
                self.stats['by_area'][area_key]['count'] += 1

                if item.get('price_pcm'):
                    self.stats['prices'].append(item['price_pcm'])

                yield item

        self.stats['by_area'][area_key]['pages'] += 1

        self.logger.info(
            f"[PAGE] {area} p{page}: {parsed_count}/{len(listing_elements)} parsed | "
            f"Running total: {self.stats['by_area'][area_key]['count']}"
        )

        # Close the Playwright page
        if playwright_page:
            await playwright_page.close()

        # Check for pagination - OpenRent uses skip parameter
        if len(listing_elements) >= self.ITEMS_PER_PAGE and page < self.max_pages:
            skip = page * self.ITEMS_PER_PAGE
            params = {
                'term': area,
                'viewType': 'LIST',
                'skip': skip,
            }
            next_url = f'https://www.openrent.co.uk/properties-to-rent?{urlencode(params)}'

            self.logger.debug(f"[PAGINATION] {area}: Following to page {page + 1} (skip={skip})")

            yield scrapy.Request(
                next_url,
                callback=self.parse_search,
                meta={
                    'playwright': True,
                    'playwright_include_page': True,
                    'playwright_page_methods': [
                        {'method': 'wait_for_selector', 'args': ['[data-listing-id]'], 'kwargs': {'timeout': 10000}},
                    ],
                    'area': area,
                    'area_key': area_key,
                    'page': page + 1,
                    'request_start': time.time()
                },
                dont_filter=True,
                errback=self.handle_error
            )
        else:
            reason = "max pages reached" if page >= self.max_pages else "no more results"
            self.logger.info(f"[COMPLETE] {area}: Stopped at page {page} ({reason})")

    def parse_property_card(self, element, listing_id: str, area: str) -> PropertyItem:
        """Parse a property card element from search results."""
        item = PropertyItem()

        item['source'] = 'openrent'
        item['property_id'] = listing_id
        item['url'] = f"https://www.openrent.co.uk/property-to-rent/{listing_id}"
        item['area'] = area

        # Try to extract price from the card
        # OpenRent shows price in format "£X,XXX pcm" or "£XXX pw"
        price_text = element.css('.price-link::text, .price::text').get()
        if price_text:
            price_match = re.search(r'£([\d,]+)\s*(pcm|pw|per\s*month|per\s*week)?', price_text, re.I)
            if price_match:
                price = int(price_match.group(1).replace(',', ''))
                period = price_match.group(2) or 'pcm'

                if 'pw' in period.lower() or 'week' in period.lower():
                    item['price_pw'] = price
                    item['price_pcm'] = int(price * 52 / 12)
                    item['price_period'] = 'pw'
                else:
                    item['price_pcm'] = price
                    item['price_pw'] = int(price * 12 / 52)
                    item['price_period'] = 'pcm'

                item['price'] = price
        else:
            item['price'] = 0
            item['price_pcm'] = 0
            item['price_pw'] = 0
            item['price_period'] = 'pcm'

        # Address - usually in the title or description
        address_text = element.css('a::text, .title::text, h2::text').get()
        item['address'] = address_text.strip() if address_text else ''

        # Extract postcode
        postcode_match = re.search(
            r'([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d?[A-Z]{0,2}',
            item['address'].upper()
        )
        item['postcode'] = postcode_match.group(1) if postcode_match else None

        # Bedrooms - look for "X bed" pattern
        beds_text = element.css('.beds::text, .property-beds::text').get()
        if beds_text:
            beds_match = re.search(r'(\d+)', beds_text)
            item['bedrooms'] = int(beds_match.group(1)) if beds_match else None
        else:
            # Check in the full text
            full_text = element.get()
            beds_match = re.search(r'(\d+)\s*(?:bed|bedroom)', full_text, re.I)
            item['bedrooms'] = int(beds_match.group(1)) if beds_match else None

        # Bathrooms - OpenRent shows this
        baths_text = element.css('.baths::text, .property-baths::text').get()
        if baths_text:
            baths_match = re.search(r'(\d+)', baths_text)
            item['bathrooms'] = int(baths_match.group(1)) if baths_match else None
        else:
            item['bathrooms'] = None

        # Property type
        item['property_type'] = ''  # Not easily visible in cards

        # Coordinates - usually not in cards
        item['latitude'] = None
        item['longitude'] = None

        # Size
        item['size_sqft'] = None

        # Agent - OpenRent is landlord-direct but sometimes shows agent
        item['agent_name'] = 'OpenRent'
        item['agent_phone'] = ''

        # Status
        item['let_agreed'] = False

        # Dates
        item['added_date'] = ''
        item['scraped_at'] = datetime.utcnow().isoformat()

        # Additional
        item['summary'] = ''
        item['features'] = []

        return item

    def closed(self, reason):
        """Log summary when spider closes."""
        elapsed = time.time() - self.stats['start_time']

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("OPENRENT SCRAPING COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"[SUMMARY] Reason: {reason}")
        self.logger.info(f"[SUMMARY] Duration: {elapsed:.1f}s")
        self.logger.info(f"[SUMMARY] Total listings: {self.stats['total']}")
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
