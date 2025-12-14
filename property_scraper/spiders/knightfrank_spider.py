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
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from property_scraper.items import PropertyItem

# Default timeouts for Playwright operations (in seconds)
EVALUATE_TIMEOUT = 30.0  # For page.evaluate() calls
CLICK_TIMEOUT = 10.0     # For page.click() calls
DETAIL_PAGE_TIMEOUT = 60.0  # Max time for entire detail page processing


async def safe_evaluate(page, script, timeout=EVALUATE_TIMEOUT, default=None):
    """Execute page.evaluate() with a timeout to prevent hanging.

    Args:
        page: Playwright page object
        script: JavaScript code to execute
        timeout: Maximum seconds to wait (default 30s)
        default: Value to return if timeout or error occurs

    Returns:
        Result of evaluate() or default on timeout/error
    """
    try:
        return await asyncio.wait_for(
            page.evaluate(script),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return default
    except Exception:
        return default


async def safe_click(page, selector, timeout=CLICK_TIMEOUT):
    """Execute page.click() with a timeout.

    Args:
        page: Playwright page object
        selector: CSS selector or text selector to click
        timeout: Maximum seconds to wait (default 10s)

    Returns:
        True if clicked successfully, False otherwise
    """
    try:
        await page.click(selector, timeout=timeout * 1000)
        return True
    except Exception:
        return False


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

    def __init__(self, max_properties=None, fetch_details=False, fetch_floorplans=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parse max_properties (None = unlimited, use high cap)
        if max_properties is None or str(max_properties).lower() in ('none', '0', ''):
            self.max_properties = None
        else:
            try:
                self.max_properties = int(max_properties)
            except (ValueError, TypeError):
                self.max_properties = None

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
            # Issue #26 FIX: OCR metrics tracking
            'ocr_attempts': 0,
            'ocr_successes': 0,
            'ocr_failures': 0,
            'ocr_timeouts': 0,
        }

        # === Issue #6 FIX: Track empty pages to stop pagination dynamically ===
        self.consecutive_empty_pages = 0
        self.max_consecutive_empty = 3  # Stop after 3 empty pages in a row
        self.stop_pagination = False

        # Thread pool for OCR (reused across requests)
        if self.fetch_floorplans and OCR_AVAILABLE:
            self.executor = ThreadPoolExecutor(max_workers=4)
        else:
            self.executor = None

        self.logger.info("=" * 70)
        self.logger.info("KNIGHT FRANK SPIDER INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"[CONFIG] Max properties: {self.max_properties or 'unlimited'}")
        self.logger.info(f"[CONFIG] Target postcodes: {', '.join(self.TARGET_POSTCODES)}")
        self.logger.info(f"[CONFIG] Fetch details: {self.fetch_details}")
        self.logger.info(f"[CONFIG] Fetch floorplans: {self.fetch_floorplans}")
        self.logger.info(f"[CONFIG] OCR available: {OCR_AVAILABLE}")
        self.logger.info("[CONFIG] Using Playwright for JS rendering")
        self.logger.info("=" * 70)

    def start_requests(self):
        """Generate requests for each page using offset parameter.

        Issue #6 FIX: Start with reasonable estimate (20 pages ~960 properties),
        actual stopping handled dynamically in parse_search via empty page detection.
        """
        base_url = 'https://www.knightfrank.co.uk/properties/residential/to-let/uk/all-types/all-beds'

        # Calculate how many pages we need (48 per page)
        # === Issue #6 FIX: Use conservative estimate, rely on empty page detection ===
        if self.max_properties is None:
            pages_needed = 20  # Start with ~960 properties, stop dynamically if empty
        else:
            pages_needed = (self.max_properties // 48) + 1
            pages_needed = min(pages_needed, 30)  # Cap at 30 pages

        self.logger.info(f"[REQUEST] Will fetch up to {pages_needed} pages (stops early on empty pages)")

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
        """Parse a single search results page.

        Issue #6 FIX: Tracks consecutive empty pages and stops pagination early.
        Issue #11 FIX: Detects empty pages quickly before waiting for selectors.
        Issue #10 FIX: Always closes playwright_page in all code paths.
        """
        page_num = response.meta['page']
        request_time = time.time() - response.meta.get('request_start', time.time())

        playwright_page = response.meta.get('playwright_page')

        self.logger.info(
            f"[RESPONSE] Page {page_num} | "
            f"Status: {response.status} | "
            f"Size: {len(response.body)/1024:.1f}KB | "
            f"Time: {request_time:.2f}s"
        )

        # === Issue #6: Check if we should stop pagination ===
        if self.stop_pagination:
            self.logger.info(f"[PAGINATION-STOP] Skipping page {page_num} - pagination stopped due to {self.max_consecutive_empty} consecutive empty pages")
            if playwright_page:
                await playwright_page.close()
            return

        if not playwright_page:
            # RESILIENCE: Retry up to 3 times if Playwright page is missing
            retry_count = response.meta.get('retry_count', 0)
            if retry_count < 3:
                backoff_delay = (retry_count + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                self.logger.warning(f"[RETRY] No Playwright page for page {page_num} ({response.url}), retrying ({retry_count + 1}/3) after {backoff_delay}s")
                await asyncio.sleep(backoff_delay)  # Use asyncio.sleep in async context
                yield response.request.replace(
                    meta={**response.meta, 'retry_count': retry_count + 1},
                    dont_filter=True
                )
                return
            self.logger.error(f"[ERROR] No Playwright page for page {page_num} ({response.url}) after 3 retries")
            return

        # === ROBUSTNESS FIX: Wait for page to load BEFORE checking if empty ===
        # Knight Frank uses React - content loads dynamically after initial HTML shell
        # We must wait for networkidle or a reasonable timeout before checking for empty
        try:
            # Wait for network to settle (dynamic content to load)
            await playwright_page.wait_for_load_state('networkidle', timeout=15000)
            await playwright_page.wait_for_timeout(2000)  # Extra buffer for React rendering
        except Exception as e:
            self.logger.debug(f"[LOAD-WAIT] Network idle wait: {e}")

        # Now check for "no results" indicators AFTER content has loaded
        try:
            is_empty = await asyncio.wait_for(
                playwright_page.evaluate('''() => {
                    // Check for common "no results" indicators
                    const noResultsIndicators = [
                        'no properties match',
                        'no results found',
                        '0 results',
                        'no properties found',
                        'sorry, we couldn',
                    ];
                    const bodyText = document.body.innerText.toLowerCase();
                    for (const indicator of noResultsIndicators) {
                        if (bodyText.includes(indicator)) {
                            return true;
                        }
                    }
                    return false;
                }'''),
                timeout=5.0
            )

            if is_empty:
                self.consecutive_empty_pages += 1
                self.logger.warning(f"[EMPTY-PAGE] Page {page_num} detected as empty (no results text). Consecutive empty: {self.consecutive_empty_pages}/{self.max_consecutive_empty}")

                if self.consecutive_empty_pages >= self.max_consecutive_empty:
                    self.stop_pagination = True
                    self.logger.info(f"[PAGINATION-STOP] Stopping pagination after {self.consecutive_empty_pages} consecutive empty pages")

                await playwright_page.close()
                return

        except asyncio.TimeoutError:
            # Quick check timed out, continue with normal flow
            pass
        except Exception as e:
            self.logger.debug(f"[EMPTY-CHECK] Empty check failed: {e}")

        # Wait for page to fully load with RETRY logic
        selector_loaded = False
        for attempt in range(3):
            try:
                await playwright_page.wait_for_selector('.property-features', timeout=30000)
                await playwright_page.wait_for_timeout(2000)
                selector_loaded = True
                break
            except Exception as e:
                if attempt < 2:
                    backoff_delay = (attempt + 1) * 2000  # Exponential backoff: 2s, 4s
                    self.logger.warning(f"[RETRY] Selector timeout on page {page_num} ({response.url}), attempt {attempt + 1}/3, reloading after {backoff_delay}ms...")
                    try:
                        await playwright_page.wait_for_timeout(backoff_delay)
                        await playwright_page.reload(timeout=30000)
                        await playwright_page.wait_for_timeout(3000)
                    except Exception as reload_err:
                        self.logger.warning(f"[RETRY] Reload failed on page {page_num}: {reload_err}")
                else:
                    self.logger.warning(f"[WARNING] Timeout waiting for cards on page {page_num} ({response.url}) after 3 attempts: {e}")
                    # === Issue #6: Treat selector timeout as empty page ===
                    self.consecutive_empty_pages += 1
                    self.logger.warning(f"[EMPTY-PAGE] Page {page_num} selector timeout. Consecutive empty: {self.consecutive_empty_pages}/{self.max_consecutive_empty}")
                    if self.consecutive_empty_pages >= self.max_consecutive_empty:
                        self.stop_pagination = True
                        self.logger.info(f"[PAGINATION-STOP] Stopping pagination after {self.consecutive_empty_pages} consecutive empty pages")

        if not selector_loaded:
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

        # === Issue #6: Track empty pages for pagination stopping ===
        if not cards_data:
            self.consecutive_empty_pages += 1
            self.logger.warning(f"[EMPTY-PAGE] Page {page_num} has no property cards. Consecutive empty: {self.consecutive_empty_pages}/{self.max_consecutive_empty}")

            if self.consecutive_empty_pages >= self.max_consecutive_empty:
                self.stop_pagination = True
                self.logger.info(f"[PAGINATION-STOP] Stopping pagination after {self.consecutive_empty_pages} consecutive empty pages")
            return
        else:
            # Reset counter when we find a page with results
            if self.consecutive_empty_pages > 0:
                self.logger.info(f"[PAGINATION] Resetting empty page counter (was {self.consecutive_empty_pages})")
            self.consecutive_empty_pages = 0

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
                                # Add timeout for page navigation to prevent indefinite hangs
                                'playwright_page_goto_kwargs': {
                                    'timeout': 30000,  # 30 seconds max for page load
                                    'wait_until': 'domcontentloaded',  # Don't wait for all resources
                                },
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
        # Issue #28 FIX: Use deterministic hash instead of Python's randomized hash()
        prop_id = id_match.group(1) if id_match else f"kf_{hashlib.sha256(href.encode()).hexdigest()[:16]}"

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

        # Bedrooms, bathrooms - single digits in sequence AFTER the address line
        # Must skip address to avoid picking up street numbers like "3 Riverlight Quay"
        item['bedrooms'] = None
        item['bathrooms'] = None

        # Find the address line index
        address_line_idx = 0
        for i, line in enumerate(lines):
            if re.search(r'[A-Z]{1,2}\d{1,2}', line):  # Line with postcode is address
                address_line_idx = i
                break

        # Look for single digits in lines AFTER address
        post_address_text = '\n'.join(lines[address_line_idx + 1:])
        nums = re.findall(r'\b(\d)\b', post_address_text)

        if len(nums) >= 3:
            # Format after address: Type, sqft, beds, baths, recs, price
            # Beds/baths/recs are usually the last 3 single digits before price
            item['bedrooms'] = int(nums[-3]) if len(nums) >= 3 else None
            item['bathrooms'] = int(nums[-2]) if len(nums) >= 2 else None
        elif len(nums) >= 2:
            item['bedrooms'] = int(nums[0])
            item['bathrooms'] = int(nums[1])

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
        """Convert postcode to area name.

        Issue #34 FIX: Uses centralized mapping from cli.registry.
        """
        from cli.registry import postcode_to_area as _postcode_to_area
        result = _postcode_to_area(postcode)
        return result if result else postcode

    async def parse_detail(self, response):
        """Parse property detail page to extract description, features, and floorplans.

        Uses asyncio.wait_for() to prevent hanging on slow/stuck pages.
        """
        item = response.meta.get('item', {})
        playwright_page = response.meta.get('playwright_page')

        if not playwright_page:
            self.logger.warning(f"[DETAIL] No Playwright page for {item.get('property_id')}")
            yield PropertyItem(**item)
            return

        try:
            # Wrap entire detail page processing with a timeout
            result = await asyncio.wait_for(
                self._process_detail_page(playwright_page, item),
                timeout=DETAIL_PAGE_TIMEOUT
            )
            if result:
                item.update(result)

        except asyncio.TimeoutError:
            self.logger.warning(f"[DETAIL-TIMEOUT] {item.get('property_id')}: Detail page processing timed out after {DETAIL_PAGE_TIMEOUT}s")
        except Exception as e:
            self.logger.error(f"[DETAIL-ERROR] {item.get('property_id')}: {e}")
        finally:
            # Always close the page to prevent resource leaks
            if playwright_page:
                try:
                    await playwright_page.close()
                except Exception:
                    pass

        yield PropertyItem(**item)

    async def _process_detail_page(self, playwright_page, item):
        """Process detail page with proper timeouts. Called by parse_detail()."""
        result = {}

        # Wait for description content to load
        await playwright_page.wait_for_timeout(2000)

        # Extract description from multiple possible selectors using safe_evaluate
        description = await safe_evaluate(playwright_page, '''() => {
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
        }''', timeout=15.0)

        # Extract key features/highlights using safe_evaluate
        features = await safe_evaluate(playwright_page, '''() => {
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
        }''', timeout=15.0, default=[])

        # Extract floorplan if enabled
        if self.fetch_floorplans:
            floorplan_data = await self._extract_floorplan_data(playwright_page, item)
            if floorplan_data:
                result.update(floorplan_data)

        # Update result with extracted data
        if description and len(description) > 50:
            # Clean description
            description = re.sub(r'\s+', ' ', description).strip()
            if len(description) > 5000:
                description = description[:5000]
            result['summary'] = description
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
            result['features'] = json.dumps({k: v for k, v in amenities.items() if v})

        return result

    async def _extract_floorplan_data(self, page, item):
        """Extract floorplan URL and run OCR if sqft missing.

        Uses safe_evaluate() with timeouts to prevent hanging.
        """
        data = {}

        try:
            # Find floorplan links in the page using safe_evaluate
            # Knight Frank uses: content.knightfrank.com/property/{id}/floorplans/en/{id}-en-floorplan-{guid}.jpg
            floorplan_url = await safe_evaluate(page, '''() => {
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
            }''', timeout=15.0)

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
        """Download floorplan image and extract sqft using OCR.

        Issue #26 FIX: Tracks OCR attempts, successes, and failures for metrics.
        """
        # Issue #26 FIX: Track OCR attempt
        self.stats['ocr_attempts'] += 1

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
                # Issue #26 FIX: Track OCR failure (no data extracted)
                self.stats['ocr_failures'] += 1
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

            if result:
                # Issue #26 FIX: Track OCR success
                self.stats['ocr_successes'] += 1
                return result
            else:
                # Issue #26 FIX: Track OCR failure (empty result)
                self.stats['ocr_failures'] += 1
                return None

        except asyncio.TimeoutError:
            # Issue #26 FIX: Track OCR timeout separately
            self.stats['ocr_timeouts'] += 1
            self.logger.debug(f"[OCR] Timeout processing floorplan")
            return None
        except Exception as e:
            # Issue #26 FIX: Track OCR failure
            self.stats['ocr_failures'] += 1
            self.logger.debug(f"[OCR] Error: {e}")
            return None

    def closed(self, reason):
        """Log summary when spider closes."""
        # Issue #21 FIX: Wait for threads to complete to prevent orphan threads
        if self.executor:
            self.executor.shutdown(wait=True)

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
            # Issue #26 FIX: Report OCR metrics
            if self.stats['ocr_attempts'] > 0:
                ocr_success_rate = (self.stats['ocr_successes'] / self.stats['ocr_attempts'] * 100)
                self.logger.info(f"[OCR-METRICS] Attempts: {self.stats['ocr_attempts']}")
                self.logger.info(f"[OCR-METRICS] Successes: {self.stats['ocr_successes']} ({ocr_success_rate:.0f}%)")
                self.logger.info(f"[OCR-METRICS] Failures: {self.stats['ocr_failures']}")
                self.logger.info(f"[OCR-METRICS] Timeouts: {self.stats['ocr_timeouts']}")
        if self.fetch_details:
            self.logger.info(f"[SUMMARY] With descriptions: {self.stats['descriptions_found']} ({desc_pct:.0f}%)")

        if self.stats['prices']:
            prices = sorted(self.stats['prices'])
            self.logger.info(f"[PRICES] Average: £{sum(prices)//len(prices):,}/pcm")
            self.logger.info(f"[PRICES] Median: £{prices[len(prices)//2]:,}/pcm")

        self.logger.info(f"[CLOSE] Reason: {reason}")
        self.logger.info("=" * 70)
