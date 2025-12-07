"""
Chestertons Spider - Scrapes rental listings from Chestertons.co.uk

Uses Playwright to bypass Cloudflare protection and render JavaScript.
Chestertons is a premium London estate agent with excellent sqft data.

Usage:
    scrapy crawl chestertons -a max_properties=500
    scrapy crawl chestertons -a fetch_details=true  # Also fetch descriptions
    scrapy crawl chestertons -a fetch_floorplans=true  # Extract floorplan URLs + OCR

Requires Playwright settings enabled in settings.py
"""

import scrapy
import json
import re
import time
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


class ChestertonsSpider(scrapy.Spider):
    """Spider for Chestertons rental listings using Playwright."""

    name = 'chestertons'
    allowed_domains = ['chestertons.co.uk']

    # Prime London areas we want
    TARGET_POSTCODES = [
        'SW1', 'SW3', 'SW5', 'SW7', 'SW10',  # Chelsea, South Ken, Knightsbridge
        'W8', 'W11', 'W2',  # Kensington, Notting Hill
        'NW1', 'NW3', 'NW8',  # St John's Wood, Hampstead
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
        self.logger.info("CHESTERTONS SPIDER INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"[CONFIG] Max properties: {self.max_properties}")
        self.logger.info(f"[CONFIG] Target postcodes: {', '.join(self.TARGET_POSTCODES)}")
        self.logger.info(f"[CONFIG] Fetch details: {self.fetch_details}")
        self.logger.info(f"[CONFIG] Fetch floorplans: {self.fetch_floorplans}")
        self.logger.info(f"[CONFIG] OCR available: {OCR_AVAILABLE}")
        self.logger.info("[CONFIG] Using Playwright for Cloudflare bypass")
        self.logger.info("=" * 70)

    def start_requests(self):
        """Generate initial requests using Playwright."""
        url = 'https://www.chestertons.co.uk/properties/lettings'

        self.logger.info(f"[REQUEST] Starting: {url}")

        yield scrapy.Request(
            url,
            callback=self.parse_search,
            meta={
                'playwright': True,
                'playwright_include_page': True,
                'playwright_page_methods': [
                    # Wait for property cards to load
                    {'method': 'wait_for_selector', 'args': ['.pegasus-property-card'],
                     'kwargs': {'timeout': 30000}},
                    # Extra wait for dynamic content
                    {'method': 'wait_for_timeout', 'args': [3000]},
                ],
                'request_start': time.time()
            },
            dont_filter=True,
            errback=self.handle_error
        )

    def handle_error(self, failure):
        """Handle request failures."""
        self.logger.error(f"[ERROR] Request failed: {failure.value}")

    async def parse_search(self, response):
        """Parse search results page with Load More pagination."""
        request_time = time.time() - response.meta.get('request_start', time.time())

        playwright_page = response.meta.get('playwright_page')

        self.logger.info(
            f"[RESPONSE] Initial page | "
            f"Status: {response.status} | "
            f"Size: {len(response.body)/1024:.1f}KB | "
            f"Time: {request_time:.2f}s"
        )

        if not playwright_page:
            self.logger.error("[ERROR] No Playwright page available")
            return

        # Calculate how many Load More clicks we need
        # Each click loads 12 more properties, starting with 12
        clicks_needed = (self.max_properties - 12) // 12
        clicks_needed = max(0, min(clicks_needed, 150))  # Cap at 150 clicks (1,812 properties)

        self.logger.info(f"[PAGINATION] Will click Load More up to {clicks_needed} times")

        # Click Load More button repeatedly using JavaScript
        click_count = 0

        for i in range(clicks_needed):
            try:
                # Use JavaScript to find and click the button by text content
                clicked = await playwright_page.evaluate('''() => {
                    const allLinks = document.querySelectorAll('a');
                    for (const link of allLinks) {
                        if (link.innerText.includes('Load More')) {
                            link.scrollIntoView();
                            link.click();
                            return true;
                        }
                    }
                    return false;
                }''')

                if clicked:
                    click_count += 1
                    await playwright_page.wait_for_timeout(2000)

                    if click_count % 10 == 0:
                        cards = await playwright_page.query_selector_all('.pegasus-property-card')
                        self.logger.info(f"[LOADING] Click {click_count}: {len(cards)} cards loaded")
                else:
                    self.logger.info(f"[PAGINATION] No more Load More button after {click_count} clicks")
                    break
            except Exception as e:
                self.logger.info(f"[PAGINATION] Stopped at click {click_count}: {e}")
                break

        # Extract all cards
        cards_data = await playwright_page.evaluate('''() => {
            const cards = document.querySelectorAll('.pegasus-property-card');
            return Array.from(cards).map(card => {
                // Get link
                const link = card.querySelector('a[href*="/properties/"]');
                const href = link ? link.getAttribute('href') : '';

                // Get full text content
                const textContent = card.innerText;

                // Get address from specific element
                const addressEl = card.querySelector('a.text-base');
                const address = addressEl ? addressEl.innerText.trim() : '';

                // Get property type (Long Let / Short Let)
                const typeEl = card.querySelector('.bg-primary');
                const letType = typeEl ? typeEl.innerText.trim() : '';

                return {
                    href,
                    address,
                    letType,
                    textContent
                };
            });
        }''')

        self.logger.info(f"[DISCOVERY] Total property cards found: {len(cards_data)}")

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
            return True  # Include if we can't determine

        # Check against target postcodes
        for target in self.TARGET_POSTCODES:
            if postcode.upper().startswith(target):
                return True
        return False

    def parse_card_data(self, card_data: dict) -> PropertyItem:
        """Parse card data extracted from Playwright."""
        item = PropertyItem()

        href = card_data.get('href', '')
        address = card_data.get('address', '')
        text = card_data.get('textContent', '')
        let_type = card_data.get('letType', '')

        # Extract property ID from URL
        # Format: /properties/21263453/lettings/SVL210100
        id_match = re.search(r'/properties/(\d+)/lettings/(\w+)', href)
        if id_match:
            prop_id = f"{id_match.group(1)}_{id_match.group(2)}"
        else:
            prop_id = f"chestertons_{hash(href)}"

        item['source'] = 'chestertons'
        item['property_id'] = prop_id

        # URL
        if href:
            item['url'] = f"https://www.chestertons.co.uk{href}" if href.startswith('/') else href
        else:
            item['url'] = ''

        # Address
        item['address'] = address

        # Extract postcode from address
        postcode_match = re.search(r'([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d?[A-Z]{0,2}$', address.upper())
        item['postcode'] = postcode_match.group(1) if postcode_match else None

        # Determine area from postcode
        if item['postcode']:
            item['area'] = self.postcode_to_area(item['postcode'])
        else:
            item['area'] = ''

        # Parse text content to extract details
        # Text format: "Long Let\nAddress\nbeds\nbaths\nreceptions\nsqft\n2\n£price\n(pcm)\nFees apply"

        # Price - look for £ amount
        price_match = re.search(r'£([\d,]+)', text)
        if price_match:
            price = int(price_match.group(1).replace(',', ''))
            # Check if weekly
            if 'pw' in text.lower() or 'week' in text.lower():
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

        # Square footage - look for "XXX ft" pattern
        sqft_match = re.search(r'(\d{3,5})\s*ft', text)
        item['size_sqft'] = int(sqft_match.group(1)) if sqft_match else None

        # Bedrooms, bathrooms, receptions - they appear as single digits after address
        # Try to find the pattern: single digits in sequence after address
        nums = re.findall(r'\b(\d)\b', text)
        if len(nums) >= 3:
            # First 3 single digits after address are typically beds, baths, receptions
            item['bedrooms'] = int(nums[0])
            item['bathrooms'] = int(nums[1])
        elif len(nums) >= 1:
            item['bedrooms'] = int(nums[0])
            item['bathrooms'] = None
        else:
            item['bedrooms'] = None
            item['bathrooms'] = None

        # Property type - infer from let type
        if 'short' in let_type.lower():
            item['property_type'] = 'short let'
        else:
            item['property_type'] = 'long let'

        # No coordinates in card view
        item['latitude'] = None
        item['longitude'] = None

        # Agent info
        item['agent_name'] = 'Chestertons'
        item['agent_phone'] = ''

        # Status
        item['let_agreed'] = 'agreed' in text.lower()

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
        """Parse property detail page to extract description, features, and floorplan."""
        item = response.meta.get('item', {})
        playwright_page = response.meta.get('playwright_page')

        if not playwright_page:
            self.logger.warning(f"[DETAIL] No Playwright page for {item.get('property_id')}")
            yield PropertyItem(**item)
            return

        try:
            # Wait for content to load
            await playwright_page.wait_for_timeout(2000)

            # Extract description from Chestertons detail page
            description = await playwright_page.evaluate('''() => {
                const selectors = [
                    '.property-description',
                    '.description',
                    '[class*="description"]',
                    '.overview',
                    '.property-details-text',
                    'article p',
                ];
                for (const sel of selectors) {
                    const el = document.querySelector(sel);
                    if (el && el.innerText.length > 50) {
                        return el.innerText.trim();
                    }
                }
                // Fallback: look for main content paragraphs
                const paras = document.querySelectorAll('main p, .content p, section p');
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
                    'ul.features li',
                    'ul.key-features li',
                    '.property-features li',
                    '.highlights li',
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
        """Click Floorplans tab, extract URL, and run OCR if sqft missing."""
        data = {}

        try:
            # Click Floorplans tab
            clicked = False
            for label in ['Floorplans', 'Floor Plans', 'Floorplan', 'Floor Plan']:
                try:
                    await page.click(f'text="{label}"', timeout=3000)
                    clicked = True
                    self.logger.debug(f"[FLOORPLAN] Clicked '{label}' tab")
                    await page.wait_for_timeout(2000)
                    break
                except:
                    pass

            if not clicked:
                self.logger.debug(f"[FLOORPLAN] No floorplan tab found for {item.get('property_id')}")
                return data

            # Extract floorplan image URL
            # Pattern: https://mr0.homeflow-assets.co.uk/files/floorplan/image/{id}/{id2}/_x_/{ref}.jpg
            floorplan_url = await page.evaluate('''() => {
                // Look for floorplan images (distinct from property photos)
                const imgs = document.querySelectorAll('img');
                for (const img of imgs) {
                    const src = img.src || img.getAttribute('data-src') || '';
                    if (src.includes('/floorplan/') || src.includes('floorplan')) {
                        return src;
                    }
                }
                // Fallback: check page source for floorplan URLs
                const html = document.documentElement.innerHTML;
                const match = html.match(/https:\\/\\/[^"\\s]+\\/files\\/floorplan\\/[^"\\s]+/);
                return match ? match[0] : null;
            }''')

            if floorplan_url:
                # Clean up URL (remove trailing backslash/escape chars)
                floorplan_url = floorplan_url.rstrip('\\').strip()
                data['floorplan_url'] = floorplan_url
                self.stats['floorplans_found'] += 1
                self.logger.info(f"[FLOORPLAN] Found URL for {item.get('property_id')}: {floorplan_url[:60]}...")

                # If no sqft from card, try OCR extraction
                if not item.get('size_sqft') and OCR_AVAILABLE and self.executor:
                    ocr_data = await self._extract_sqft_via_ocr(floorplan_url)
                    if ocr_data:
                        data.update(ocr_data)
                        if ocr_data.get('size_sqft'):
                            self.stats['sqft_from_ocr'] += 1
                            self.logger.info(f"[OCR] Extracted {ocr_data.get('size_sqft')} sqft for {item.get('property_id')}")

        except Exception as e:
            self.logger.debug(f"[FLOORPLAN] Error extracting: {e}")

        return data

    async def _extract_sqft_via_ocr(self, floorplan_url):
        """Download floorplan image and extract sqft using OCR."""
        try:
            import requests

            def download_and_extract(url):
                """Download image and run OCR extraction."""
                resp = requests.get(url, timeout=15)
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

        except Exception as e:
            self.logger.debug(f"[OCR] Error: {e}")
            return None

    def closed(self, reason):
        """Log summary when spider closes."""
        # Clean up the thread pool executor
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)

        elapsed = time.time() - self.stats['start_time']
        total = self.stats['total'] or 1  # Avoid division by zero
        sqft_pct = (self.stats['sqft_found'] / total * 100)
        desc_pct = (self.stats['descriptions_found'] / total * 100)
        floorplan_pct = (self.stats['floorplans_found'] / total * 100)

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("CHESTERTONS SCRAPING COMPLETE")
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
