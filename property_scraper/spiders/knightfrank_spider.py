"""
Knight Frank Spider - Scrapes rental listings from KnightFrank.co.uk

Uses Playwright for JavaScript rendering.
Knight Frank is a premium agent with excellent sqft data on all properties.

Usage:
    scrapy crawl knightfrank -a max_properties=500
    scrapy crawl knightfrank -a fetch_details=true  # Also fetch descriptions

Requires Playwright settings enabled in settings.py
"""

import scrapy
import re
import time
import json
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from property_scraper.items import PropertyItem

# OCR support for floorplan extraction
try:
    from property_scraper.utils.floorplan_extractor import FloorplanExtractor
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class KnightFrankSpider(scrapy.Spider):
    """Spider for Knight Frank rental listings using Playwright."""

    name = 'knightfrank'
    allowed_domains = ['knightfrank.co.uk']

    # Prime London areas we want
    TARGET_POSTCODES = [
        'SW1', 'SW3', 'SW5', 'SW7', 'SW10', 'SW11',  # Chelsea, South Ken, Knightsbridge
        'W8', 'W11', 'W2',  # Kensington, Notting Hill
        'NW1', 'NW3', 'NW8',  # St John's Wood, Hampstead
        'W1',  # Mayfair
    ]

    def __init__(self, max_properties=500, fetch_details=False, fetch_floorplans=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.max_properties = int(max_properties)
        except (ValueError, TypeError):
            self.max_properties = 500

        # Enable detail page fetching for descriptions
        self.fetch_details = str(fetch_details).lower() in ('true', '1', 'yes')

        # Enable floorplan extraction (implies fetch_details)
        self.fetch_floorplans = str(fetch_floorplans).lower() in ('true', '1', 'yes')
        if self.fetch_floorplans:
            self.fetch_details = True  # Need detail page for floorplans

        self.stats = {
            'total': 0,
            'prices': [],
            'sqft_found': 0,
            'sqft_from_ocr': 0,
            'floorplans_found': 0,
            'descriptions_found': 0,
            'start_time': time.time(),
        }

        # Thread pool for OCR (reused across requests)
        if self.fetch_floorplans and OCR_AVAILABLE:
            self.executor = ThreadPoolExecutor(max_workers=4)
        else:
            self.executor = None

        self.logger.info("=" * 70)
        self.logger.info("KNIGHT FRANK SPIDER INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"[CONFIG] Max properties: {self.max_properties}")
        self.logger.info(f"[CONFIG] Target postcodes: {', '.join(self.TARGET_POSTCODES)}")
        self.logger.info(f"[CONFIG] Fetch details: {self.fetch_details}")
        self.logger.info(f"[CONFIG] Fetch floorplans: {self.fetch_floorplans}")
        self.logger.info(f"[CONFIG] OCR available: {OCR_AVAILABLE}")
        self.logger.info("[CONFIG] Using Playwright for JS rendering")
        self.logger.info("=" * 70)

    def start_requests(self):
        """Generate requests for each page using offset parameter."""
        base_url = 'https://www.knightfrank.co.uk/properties/residential/to-let/uk/all-types/all-beds'

        # Calculate how many pages we need (48 per page)
        pages_needed = (self.max_properties // 48) + 1
        pages_needed = min(pages_needed, 30)  # Cap at 30 pages

        self.logger.info(f"[REQUEST] Will fetch {pages_needed} pages")

        for page_num in range(1, pages_needed + 1):
            offset = (page_num - 1) * 48
            if offset == 0:
                url = base_url
            else:
                url = f"{base_url};offset={offset}"

            self.logger.info(f"[REQUEST] Queueing page {page_num}: {url}")

            yield scrapy.Request(
                url,
                callback=self.parse_search,
                meta={
                    'playwright': True,
                    'playwright_include_page': True,
                    'page': page_num,
                    'request_start': time.time()
                },
                dont_filter=True,
                errback=self.handle_error
            )

    def handle_error(self, failure):
        """Handle request failures."""
        self.logger.error(f"[ERROR] Request failed: {failure.value}")

    async def parse_search(self, response):
        """Parse a single search results page."""
        page_num = response.meta['page']
        request_time = time.time() - response.meta.get('request_start', time.time())

        playwright_page = response.meta.get('playwright_page')

        self.logger.info(
            f"[RESPONSE] Page {page_num} | "
            f"Status: {response.status} | "
            f"Size: {len(response.body)/1024:.1f}KB | "
            f"Time: {request_time:.2f}s"
        )

        if not playwright_page:
            self.logger.error("[ERROR] No Playwright page available")
            return

        # Wait for page to fully load
        try:
            await playwright_page.wait_for_selector('.property-features', timeout=30000)
            await playwright_page.wait_for_timeout(2000)
        except Exception as e:
            self.logger.warning(f"[WARNING] Timeout waiting for cards on page {page_num}: {e}")
            await playwright_page.close()
            return

        # Extract cards from this page
        cards_data = await playwright_page.evaluate('''() => {
            const features = document.querySelectorAll('.property-features');
            return Array.from(features).map(f => {
                const card = f.closest('a') || f.parentElement?.parentElement;
                return {
                    text: card?.innerText || '',
                    href: card?.href || card?.querySelector('a')?.href || ''
                };
            });
        }''')

        self.logger.info(f"[PAGE {page_num}] Found {len(cards_data)} cards")

        await playwright_page.close()

        if not cards_data:
            self.logger.warning("[WARNING] No property cards found")
            return

        # Parse each card
        parsed_count = 0
        target_count = 0
        for card_data in cards_data:
            item = self.parse_card_data(card_data)
            if item:
                if self.is_target_area(item):
                    target_count += 1
                    self.stats['total'] += 1

                    if item.get('price_pcm'):
                        self.stats['prices'].append(item['price_pcm'])
                    if item.get('size_sqft'):
                        self.stats['sqft_found'] += 1

                    # Optionally fetch detail page for description
                    if self.fetch_details and item.get('url'):
                        yield scrapy.Request(
                            item['url'],
                            callback=self.parse_detail,
                            meta={
                                'item': dict(item),
                                'playwright': True,
                                'playwright_include_page': True,
                            },
                            dont_filter=True,
                            errback=self.handle_error
                        )
                    else:
                        yield item
                parsed_count += 1

        self.logger.info(
            f"[COMPLETE] Parsed {parsed_count} cards, {target_count} in target areas"
        )

    def is_target_area(self, item) -> bool:
        """Check if property is in a target London area."""
        postcode = item.get('postcode', '')
        if not postcode:
            return True

        for target in self.TARGET_POSTCODES:
            if postcode.upper().startswith(target):
                return True
        return False

    def parse_card_data(self, card_data: dict) -> PropertyItem:
        """Parse card data extracted from Playwright."""
        item = PropertyItem()

        href = card_data.get('href', '')
        text = card_data.get('text', '')

        if not href or not text:
            return None

        # Extract property ID from URL
        # Format: /properties/residential/to-let/{location}/{id}
        id_match = re.search(r'/([a-z]{3}\d+)$', href, re.I)
        prop_id = id_match.group(1) if id_match else f"kf_{hash(href)}"

        item['source'] = 'knightfrank'
        item['property_id'] = prop_id
        item['url'] = href

        # Parse text content
        # Format: "1/13\nShort Let\nAddress\nType\n, sqft\nbeds\nbaths\nrecs\nprice\nperiod"
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        # Find address - usually contains postcode pattern
        address = ''
        for line in lines:
            if re.search(r'[A-Z]{1,2}\d{1,2}', line):
                address = line
                break

        item['address'] = address

        # Extract postcode
        postcode_match = re.search(r'([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d?[A-Z]{0,2}$', address.upper())
        item['postcode'] = postcode_match.group(1) if postcode_match else None

        # Determine area
        if item['postcode']:
            item['area'] = self.postcode_to_area(item['postcode'])
        else:
            item['area'] = ''

        # Price - look for £ amount
        price = 0
        period = 'pcm'
        price_match = re.search(r'£([\d,]+)', text)
        if price_match:
            price = int(price_match.group(1).replace(',', ''))
            if 'weekly' in text.lower():
                period = 'pw'
            elif 'monthly' in text.lower():
                period = 'pcm'

        if period == 'pw':
            item['price_pw'] = price
            item['price_pcm'] = int(price * 52 / 12)
            item['price_period'] = 'pw'
        else:
            item['price_pcm'] = price
            item['price_pw'] = int(price * 12 / 52)
            item['price_period'] = 'pcm'
        item['price'] = price

        # Square footage - look for "XXX sqft" or "X,XXX sqft"
        sqft_match = re.search(r'([\d,]+)\s*sqft', text, re.I)
        if sqft_match:
            item['size_sqft'] = int(sqft_match.group(1).replace(',', ''))
        else:
            item['size_sqft'] = None

        # Bedrooms, bathrooms - single digits in sequence
        nums = re.findall(r'\b(\d)\b', text)
        # Skip first few which might be image count
        if len(nums) >= 5:
            item['bedrooms'] = int(nums[-3])
            item['bathrooms'] = int(nums[-2])
        elif len(nums) >= 3:
            item['bedrooms'] = int(nums[0])
            item['bathrooms'] = int(nums[1])
        else:
            item['bedrooms'] = None
            item['bathrooms'] = None

        # Property type
        prop_type = ''
        for line in lines:
            line_lower = line.lower()
            if line_lower in ['flat', 'house', 'apartment', 'maisonette', 'studio']:
                prop_type = line_lower
                break
        item['property_type'] = prop_type

        # Let type
        if 'short let' in text.lower():
            item['property_type'] = 'short let'

        # No coordinates
        item['latitude'] = None
        item['longitude'] = None

        # Agent info
        item['agent_name'] = 'Knight Frank'
        item['agent_phone'] = ''

        # Status
        item['let_agreed'] = 'agreed' in text.lower() or 'let' in text.lower() and 'under' in text.lower()

        # Dates
        item['added_date'] = ''
        item['scraped_at'] = datetime.utcnow().isoformat()

        # Additional
        item['summary'] = ''
        item['features'] = []

        return item

    def postcode_to_area(self, postcode: str) -> str:
        """Convert postcode to area name."""
        mapping = {
            'SW1': 'Belgravia',
            'SW3': 'Chelsea',
            'SW5': 'Earls Court',
            'SW7': 'South Kensington',
            'SW10': 'Chelsea',
            'SW11': 'Battersea',
            'W1': 'Mayfair',
            'W8': 'Kensington',
            'W11': 'Notting Hill',
            'W2': 'Bayswater',
            'NW1': "St John's Wood",
            'NW3': 'Hampstead',
            'NW8': "St John's Wood",
        }
        for prefix, area in mapping.items():
            if postcode.upper().startswith(prefix):
                return area
        return postcode

    async def parse_detail(self, response):
        """Parse property detail page to extract description, features, and floorplans."""
        item = response.meta.get('item', {})
        playwright_page = response.meta.get('playwright_page')

        if not playwright_page:
            self.logger.warning(f"[DETAIL] No Playwright page for {item.get('property_id')}")
            yield PropertyItem(**item)
            return

        try:
            # Wait for description content to load
            await playwright_page.wait_for_timeout(2000)

            # Extract description from multiple possible selectors
            description = await playwright_page.evaluate('''() => {
                const selectors = [
                    '.property-description',
                    '.kf-property-description',
                    '[class*="Description"]',
                    '.summary',
                    '.overview-text',
                    'article.property-details',
                ];
                for (const sel of selectors) {
                    const el = document.querySelector(sel);
                    if (el && el.innerText.length > 50) {
                        return el.innerText.trim();
                    }
                }
                // Fallback: look for main content paragraphs
                const paras = document.querySelectorAll('main p, .content p');
                let text = '';
                for (const p of paras) {
                    if (p.innerText.length > 30) {
                        text += p.innerText + ' ';
                    }
                }
                return text.trim() || null;
            }''')

            # Extract key features/highlights
            features = await playwright_page.evaluate('''() => {
                const selectors = [
                    'ul.highlights li',
                    'ul.key-features li',
                    '.property-features li',
                ];
                for (const sel of selectors) {
                    const els = document.querySelectorAll(sel);
                    if (els.length > 0) {
                        return Array.from(els).map(el => el.innerText.trim()).filter(t => t);
                    }
                }
                return [];
            }''')

            # Extract floorplan if enabled
            if self.fetch_floorplans:
                floorplan_data = await self._extract_floorplan_data(playwright_page, item)
                if floorplan_data:
                    item.update(floorplan_data)

            await playwright_page.close()

            # Update item with extracted data
            if description and len(description) > 50:
                # Clean description
                description = re.sub(r'\s+', ' ', description).strip()
                if len(description) > 5000:
                    description = description[:5000]
                item['summary'] = description
                self.stats['descriptions_found'] += 1
                self.logger.info(f"[DETAIL] {item.get('property_id')}: {len(description)} chars")

            if features:
                # Extract amenities from features list
                features_text = ' '.join(features).lower()
                amenities = {}
                amenities['has_balcony'] = 'balcon' in features_text
                amenities['has_terrace'] = 'terrace' in features_text
                amenities['has_garden'] = 'garden' in features_text
                amenities['has_porter'] = 'porter' in features_text or 'concierge' in features_text
                amenities['has_gym'] = 'gym' in features_text or 'fitness' in features_text
                amenities['has_pool'] = 'pool' in features_text or 'swimming' in features_text
                amenities['has_parking'] = 'parking' in features_text or 'garage' in features_text
                amenities['has_lift'] = 'lift' in features_text or 'elevator' in features_text
                amenities['has_ac'] = 'air con' in features_text or 'conditioning' in features_text
                item['features'] = json.dumps({k: v for k, v in amenities.items() if v})

        except Exception as e:
            self.logger.error(f"[DETAIL-ERROR] {item.get('property_id')}: {e}")
            if playwright_page:
                await playwright_page.close()

        yield PropertyItem(**item)

    async def _extract_floorplan_data(self, page, item):
        """Extract floorplan URL and run OCR if sqft missing."""
        data = {}

        try:
            # Find floorplan links in the page
            # Knight Frank uses: content.knightfrank.com/property/{id}/floorplans/en/{id}-en-floorplan-{guid}.jpg
            floorplan_url = await page.evaluate('''() => {
                // Look for links with "Floorplan" text
                const links = document.querySelectorAll('a');
                for (const link of links) {
                    const href = link.href || '';
                    const text = link.innerText || '';
                    if (href.includes('/floorplans/') && text.toLowerCase().includes('floorplan')) {
                        return href;
                    }
                }

                // Look for images with floorplan in src
                const imgs = document.querySelectorAll('img');
                for (const img of imgs) {
                    const src = img.src || img.getAttribute('data-src') || '';
                    if (src.includes('/floorplans/') || src.includes('floorplan')) {
                        return src;
                    }
                }

                // Check for hidden/lazy images
                const html = document.documentElement.innerHTML;
                const match = html.match(/https:\\/\\/content\\.knightfrank\\.com\\/property\\/[^\\/]+\\/floorplans\\/[^"'\\s]+/);
                return match ? match[0] : null;
            }''')

            if floorplan_url:
                # Ensure URL ends with image extension
                if not any(floorplan_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    floorplan_url = floorplan_url + '.jpg' if '.' not in floorplan_url.split('/')[-1] else floorplan_url

                floorplan_url = floorplan_url.rstrip('\\').strip()
                data['floorplan_url'] = floorplan_url
                self.stats['floorplans_found'] += 1
                self.logger.debug(f"[FLOORPLAN] Found: {floorplan_url}")

                # Run OCR if sqft is missing
                if not item.get('size_sqft') and OCR_AVAILABLE and self.executor:
                    ocr_data = await self._extract_sqft_via_ocr(floorplan_url)
                    if ocr_data:
                        data.update(ocr_data)
                        if ocr_data.get('size_sqft'):
                            self.stats['sqft_from_ocr'] += 1
                            self.logger.info(f"[OCR] Extracted {ocr_data['size_sqft']} sqft from floorplan")

        except Exception as e:
            self.logger.debug(f"[FLOORPLAN] Error extracting: {e}")

        return data

    async def _extract_sqft_via_ocr(self, floorplan_url):
        """Download floorplan image and extract sqft using OCR."""
        try:
            import requests

            def download_and_extract(url):
                """Blocking function to run in thread pool."""
                resp = requests.get(url, timeout=15, headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                })
                if resp.status_code != 200:
                    return None

                extractor = FloorplanExtractor()
                return extractor.extract_from_bytes(resp.content)

            loop = asyncio.get_event_loop()
            floorplan_data = await asyncio.wait_for(
                loop.run_in_executor(self.executor, download_and_extract, floorplan_url),
                timeout=25.0
            )

            if not floorplan_data:
                return None

            result = {}

            # Extract sqft
            if floorplan_data.total_sqft:
                result['size_sqft'] = floorplan_data.total_sqft
            if floorplan_data.total_sqm:
                result['size_sqm'] = floorplan_data.total_sqm

            # Extract floor data for ML training
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

            # Extract room details
            if floorplan_data.rooms:
                room_details = []
                for room in floorplan_data.rooms[:10]:
                    room_info = {'type': room.type, 'name': room.name}
                    if room.dimensions_imperial:
                        room_info['dimensions'] = room.dimensions_imperial
                    if room.sqft:
                        room_info['sqft'] = room.sqft
                    room_details.append(room_info)
                if room_details:
                    result['room_details'] = room_details

            return result if result else None

        except asyncio.TimeoutError:
            self.logger.debug(f"[OCR] Timeout processing floorplan")
            return None
        except Exception as e:
            self.logger.debug(f"[OCR] Error: {e}")
            return None

    def closed(self, reason):
        """Log summary when spider closes."""
        # Cleanup executor
        if self.executor:
            self.executor.shutdown(wait=False)

        elapsed = time.time() - self.stats['start_time']
        sqft_pct = (self.stats['sqft_found'] / self.stats['total'] * 100) if self.stats['total'] else 0
        desc_pct = (self.stats['descriptions_found'] / self.stats['total'] * 100) if self.stats['total'] else 0
        floorplan_pct = (self.stats['floorplans_found'] / self.stats['total'] * 100) if self.stats['total'] else 0

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("KNIGHT FRANK SCRAPING COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"[SUMMARY] Duration: {elapsed:.1f}s")
        self.logger.info(f"[SUMMARY] Total listings: {self.stats['total']}")
        self.logger.info(f"[SUMMARY] With sqft: {self.stats['sqft_found']} ({sqft_pct:.0f}%)")
        if self.fetch_floorplans:
            self.logger.info(f"[SUMMARY] Floorplans found: {self.stats['floorplans_found']} ({floorplan_pct:.0f}%)")
            self.logger.info(f"[SUMMARY] Sqft from OCR: {self.stats['sqft_from_ocr']}")
        if self.fetch_details:
            self.logger.info(f"[SUMMARY] With descriptions: {self.stats['descriptions_found']} ({desc_pct:.0f}%)")

        if self.stats['prices']:
            prices = sorted(self.stats['prices'])
            self.logger.info(f"[PRICES] Average: £{sum(prices)//len(prices):,}/pcm")
            self.logger.info(f"[PRICES] Median: £{prices[len(prices)//2]:,}/pcm")

        self.logger.info(f"[CLOSE] Reason: {reason}")
        self.logger.info("=" * 70)
