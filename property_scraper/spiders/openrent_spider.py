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
from scrapy_playwright.page import PageMethod
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
                        PageMethod('wait_for_selector', '[data-listing-id]', timeout=15000),
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

        # Extract structured card data from the rendered page.
        # OpenRent's [data-listing-id] element is just the image swiper; the price,
        # title/address and bed/bath counts live in the surrounding card container.
        # We compute innerText/alt in the browser context, which the raw-HTML
        # selectors used previously could not do.
        cards = []
        if playwright_page:
            try:
                cards = await playwright_page.evaluate(self._CARD_EXTRACTION_JS)
            except Exception as e:
                self.logger.warning(f"[EXTRACT] {area} p{page}: card evaluate failed - {e}")
        else:
            self.logger.warning(f"[EXTRACT] {area} p{page}: no Playwright page available")

        self.logger.info(f"[DISCOVERY] {area} p{page}: {len(cards)} listing cards found")

        # Parse each property card
        parsed_count = 0
        for card in cards:
            listing_id = card.get('id')
            if not listing_id:
                continue

            item = self.parse_property_card(card, listing_id, area_key)
            if item:
                parsed_count += 1
                self.stats['total'] += 1
                self.stats['by_area'][area_key]['count'] += 1

                if item.get('price_pcm'):
                    self.stats['prices'].append(item['price_pcm'])

                yield item

        self.stats['by_area'][area_key]['pages'] += 1

        self.logger.info(
            f"[PAGE] {area} p{page}: {parsed_count}/{len(cards)} parsed | "
            f"Running total: {self.stats['by_area'][area_key]['count']}"
        )

        # Close the Playwright page
        if playwright_page:
            await playwright_page.close()

        # Check for pagination - OpenRent uses skip parameter
        if len(cards) >= self.ITEMS_PER_PAGE and page < self.max_pages:
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
                        PageMethod('wait_for_selector', '[data-listing-id]', timeout=15000),
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

    # JS run in the browser context to pull structured data out of each card.
    # OpenRent's [data-listing-id] node is only the photo swiper, so we walk up to
    # the nearest ancestor that contains a price, then read its innerText plus the
    # property image's alt text (which reliably encodes "<beds> Bed <type>, <street>, <district>").
    _CARD_EXTRACTION_JS = r'''() => {
        const swipers = [...document.querySelectorAll('[data-listing-id]')];
        const results = [];
        const seen = new Set();
        for (const sw of swipers) {
            const id = sw.getAttribute('data-listing-id');
            if (!id || seen.has(id)) continue;
            seen.add(id);
            // Walk up to the card container that holds the price text
            let node = sw, card = null;
            for (let i = 0; i < 6; i++) {
                node = node.parentElement;
                if (!node) break;
                if (node.innerText && node.innerText.includes('£')) { card = node; break; }
            }
            if (!card) { results.push({ id }); continue; }
            const img = card.querySelector('img.propertyPic, img[alt]');
            results.push({
                id,
                alt: img ? (img.getAttribute('alt') || '') : '',
                fullText: card.innerText || '',
            });
        }
        return results;
    }'''

    def parse_property_card(self, card: dict, listing_id: str, area: str) -> PropertyItem:
        """Parse a structured property card dict (from _CARD_EXTRACTION_JS)."""
        item = PropertyItem()

        alt = (card.get('alt') or '').strip()
        full_text = card.get('fullText') or ''

        item['source'] = 'openrent'
        item['property_id'] = listing_id
        item['url'] = f"https://www.openrent.co.uk/property-to-rent/{listing_id}"
        item['area'] = area

        # Price - OpenRent shows "£X,XXX" followed by "/month" or "/week" (or "pcm"/"pw")
        price_match = re.search(r'£\s*([\d,]+)', full_text)
        item['price'] = 0
        item['price_pcm'] = 0
        item['price_pw'] = 0
        item['price_period'] = 'pcm'
        if price_match:
            price = int(price_match.group(1).replace(',', ''))
            # Look at the text right after the price for the period
            tail = full_text[price_match.end():price_match.end() + 20].lower()
            is_weekly = ('week' in tail) or ('/wk' in tail) or re.search(r'\bpw\b', tail) is not None
            if is_weekly:
                item['price_pw'] = price
                item['price_pcm'] = int(price * 52 / 12)
                item['price_period'] = 'pw'
            else:
                item['price_pcm'] = price
                item['price_pw'] = int(price * 12 / 52)
                item['price_period'] = 'pcm'
            item['price'] = price

        # Address - the image alt encodes "<type>, <street>, <district>", e.g.
        # "2 Bed Flat, Elm Park Gardens, SW10". Fall back to the title line in text.
        address = alt
        if not address:
            # First non-price line that contains a comma + postcode-like token
            for line in full_text.split('\n'):
                line = line.strip()
                if ',' in line and re.search(r'[A-Z]{1,2}\d', line):
                    address = line
                    break
        item['address'] = address

        # Postcode district (e.g. SW10) - from alt/address
        postcode_match = re.search(r'\b([A-Z]{1,2}\d{1,2}[A-Z]?)\b', address.upper())
        item['postcode'] = postcode_match.group(1) if postcode_match else None

        # Property type and bedrooms from the alt/title ("2 Bed Flat" / "Studio Flat")
        type_source = alt or address
        item['property_type'] = ''
        type_match = re.search(r'\b(flat|apartment|house|studio|maisonette|bungalow|room)\b', type_source, re.I)
        if type_match:
            item['property_type'] = type_match.group(1).lower()

        # Bedrooms: "Studio" => 0, "N Bed" => N
        if re.search(r'\bstudio\b', type_source, re.I):
            item['bedrooms'] = 0
        else:
            beds_match = re.search(r'(\d+)\s*bed', type_source, re.I)
            item['bedrooms'] = int(beds_match.group(1)) if beds_match else None

        # Bathrooms: "N Bath" in the card text
        baths_match = re.search(r'(\d+)\s*bath', full_text, re.I)
        item['bathrooms'] = int(baths_match.group(1)) if baths_match else None

        # Furnished status
        if re.search(r'\bunfurnished\b', full_text, re.I):
            item['furnished'] = 'unfurnished'
        elif re.search(r'\bpart[\s-]*furnished\b', full_text, re.I):
            item['furnished'] = 'part_furnished'
        elif re.search(r'\bfurnished\b', full_text, re.I):
            item['furnished'] = 'furnished'

        # Coordinates - not in cards
        item['latitude'] = None
        item['longitude'] = None

        # Size - not in cards
        item['size_sqft'] = None

        # Agent - OpenRent is landlord-direct
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
