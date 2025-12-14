"""
Savills Spider - Scrapes rental listings from search.savills.com

Uses Playwright to render the React-based search interface.
Savills is a premium London estate agent with excellent sqft data.

Key Discovery: Uses traditional pagination (84 pages), NOT infinite scroll.
IMPORTANT: Must CLICK through pages - URL parameters don't work with React state.

Usage:
    scrapy crawl savills -a max_properties=500
    scrapy crawl savills -a max_pages=20

Requires Playwright settings enabled in settings.py
"""

import scrapy
import json
import re
import time
import asyncio
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from property_scraper.items import PropertyItem
from scrapy_playwright.page import PageMethod

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
        await asyncio.wait_for(
            page.click(selector, timeout=timeout * 1000),
            timeout=timeout + 5  # Extra buffer for the operation
        )
        return True
    except Exception:
        return False

# OCR support for floorplan extraction
try:
    from property_scraper.utils.floorplan_extractor import FloorplanExtractor
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class SavillsSpider(scrapy.Spider):
    """Spider for Savills rental listings using Playwright with click-based pagination."""

    name = 'savills'
    allowed_domains = ['savills.com', 'search.savills.com']

    # Prime London areas we want
    TARGET_POSTCODES = [
        'SW1', 'SW3', 'SW5', 'SW7', 'SW10',  # Chelsea, South Ken, Knightsbridge
        'W8', 'W11', 'W2', 'W1',  # Kensington, Notting Hill, Mayfair
        'NW1', 'NW3', 'NW8',  # St John's Wood, Hampstead
    ]

    def __init__(self, max_properties=None, max_pages=None, fetch_floorplans=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parse max_properties (None = unlimited)
        if max_properties is None or str(max_properties).lower() in ('none', '0', ''):
            self.max_properties = None
        else:
            try:
                self.max_properties = int(max_properties)
            except (ValueError, TypeError):
                self.max_properties = None

        # Parse max_pages (None = unlimited)
        if max_pages is None or str(max_pages).lower() in ('none', '0', ''):
            self.max_pages = None
        else:
            try:
                self.max_pages = int(max_pages)
            except (ValueError, TypeError):
                self.max_pages = None

        # Enable floorplan extraction via detail pages
        self.fetch_floorplans = str(fetch_floorplans).lower() in ('true', '1', 'yes')

        self.stats = {
            'total': 0,
            'prices': [],
            'sqft_found': 0,
            'sqft_from_ocr': 0,
            'floorplans_found': 0,
            'start_time': time.time(),
            'pages_scraped': 0,
        }

        # Thread pool for OCR (reused across requests)
        if self.fetch_floorplans and OCR_AVAILABLE:
            self.executor = ThreadPoolExecutor(max_workers=4)
        else:
            self.executor = None

        self.logger.info("=" * 70)
        self.logger.info("SAVILLS SPIDER INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"[CONFIG] Max properties: {self.max_properties or 'unlimited'}")
        self.logger.info(f"[CONFIG] Max pages: {self.max_pages or 'unlimited'}")
        self.logger.info(f"[CONFIG] Target postcodes: {', '.join(self.TARGET_POSTCODES)}")
        self.logger.info(f"[CONFIG] Fetch floorplans: {self.fetch_floorplans}")
        self.logger.info(f"[CONFIG] OCR available: {OCR_AVAILABLE}")
        self.logger.info("[CONFIG] Using Playwright with CLICK-based pagination")
        self.logger.info("=" * 70)

    def start_requests(self):
        """Generate initial request for page 1."""
        url = 'https://search.savills.com/list/property-to-rent/uk'
        self.logger.info(f"[REQUEST] Starting: {url}")

        yield scrapy.Request(
            url,
            callback=self.parse_all_pages,
            meta={
                'playwright': True,
                'playwright_include_page': True,
                'playwright_page_methods': [
                    PageMethod('wait_for_timeout', 8000),
                ],
                'request_start': time.time(),
            },
            dont_filter=True,
            errback=self.handle_error
        )

    def handle_error(self, failure):
        """Handle request failures."""
        self.logger.error(f"[ERROR] Request failed: {failure.value}")

    async def parse_all_pages(self, response):
        """Parse all pages by clicking through pagination within a single session."""
        request_time = time.time() - response.meta.get('request_start', time.time())
        playwright_page = response.meta.get('playwright_page')

        self.logger.info(
            f"[RESPONSE] Initial page | "
            f"Status: {response.status} | "
            f"Size: {len(response.body)/1024:.1f}KB | "
            f"Time: {request_time:.2f}s"
        )

        if not playwright_page:
            # RESILIENCE: Retry up to 3 times if Playwright page is missing
            retry_count = response.meta.get('retry_count', 0)
            if retry_count < 3:
                backoff_delay = (retry_count + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                self.logger.warning(f"[RETRY] No Playwright page for {response.url}, retrying ({retry_count + 1}/3) after {backoff_delay}s")
                await asyncio.sleep(backoff_delay)  # Use asyncio.sleep in async context
                yield response.request.replace(
                    meta={**response.meta, 'retry_count': retry_count + 1},
                    dont_filter=True
                )
                return
            self.logger.error(f"[ERROR] No Playwright page for {response.url} after 3 retries")
            return

        current_page = 1
        seen_ids = set()  # Track property IDs to avoid duplicates

        # Continue while within limits (None = unlimited)
        while (self.max_pages is None or current_page <= self.max_pages) and \
              (self.max_properties is None or self.stats['total'] < self.max_properties):
            # Extract property data from current page
            cards_data = await playwright_page.evaluate('''() => {
                const results = [];

                // Find all listing items
                const listings = document.querySelectorAll('li.sv-results-listing__item');

                for (const listing of listings) {
                    const text = listing.innerText || '';

                    // MUST have sqft - required for model training
                    const sqftMatch = text.match(/(\\d+(?:,\\d+)?)\\s*sq\\s*ft/i);
                    if (!sqftMatch) continue;

                    // MUST have price (Monthly or Weekly)
                    const monthlyMatch = text.match(/£([\\d,]+)\\s*Monthly/);
                    const weeklyMatch = text.match(/£([\\d,]+)\\s*Weekly/);
                    if (!monthlyMatch && !weeklyMatch) continue;

                    // Convert to monthly
                    let priceValue;
                    let priceType;
                    if (monthlyMatch) {
                        priceValue = parseInt(monthlyMatch[1].replace(/,/g, ''));
                        priceType = 'monthly';
                    } else {
                        priceValue = parseInt(weeklyMatch[1].replace(/,/g, '')) * 52 / 12;
                        priceType = 'weekly';
                    }

                    // Extract other fields
                    const pcMatch = text.match(/([A-Z]{1,2}\\d{1,2}[A-Z]?\\s*\\d[A-Z]{2})/);
                    const bedsMatch = text.match(/(\\d+)\\s*Bedrooms?/);
                    const bathsMatch = text.match(/(\\d+)\\s*Bathrooms?/);

                    // Address extraction - look for street patterns
                    const addressMatch = text.match(/([^\\n]*(?:Street|Road|Avenue|Lane|Court|Gardens|Square|Place|Terrace|House|Walk|Close|Way|Drive|Mews|Row)[^\\n]*)/i);

                    const postcode = pcMatch ? pcMatch[1].toUpperCase() : null;
                    const price = Math.round(priceValue);  // Already calculated above
                    const sqft = sqftMatch[1].replace(/,/g, '');
                    const beds = bedsMatch ? bedsMatch[1] : null;
                    const baths = bathsMatch ? bathsMatch[1] : null;

                    // Determine furnished status
                    const hasFurnished = /\\bFURNISHED\\b/.test(text);
                    const hasUnfurnished = /\\bUNFURNISHED\\b/.test(text);
                    const isBoth = /FURNISHED \\/ UNFURNISHED|FURNISHED\\/UNFURNISHED/.test(text);

                    let furnished = null;
                    if (isBoth) furnished = 'part_furnished';
                    else if (hasFurnished && !hasUnfurnished) furnished = 'furnished';
                    else if (hasUnfurnished && !hasFurnished) furnished = 'unfurnished';

                    // Get property link - look for property-detail URLs
                    const linkEl = listing.querySelector('a[href*="/property-detail/"]') ||
                                   listing.querySelector('a[href*="/property/"]') ||
                                   listing.querySelector('a');
                    const href = linkEl ? linkEl.href : '';

                    // Build address
                    let address = addressMatch ? addressMatch[1].trim() : '';
                    if (postcode && !address.includes(postcode)) {
                        address += (address ? ', ' : '') + postcode;
                    }

                    results.push({
                        href: href,
                        text: text.substring(0, 2000),
                        address: address,
                        sqft: sqft,
                        price: price,
                        beds: beds,
                        baths: baths,
                        postcode: postcode,
                        furnished: furnished
                    });
                }

                return results;
            }''')

            # Get pagination info
            pagination_info = await playwright_page.evaluate('''() => {
                const pagination = document.querySelector('.sv-pagination');
                if (!pagination) return { totalPages: 1, currentPage: 1, hasNext: false };

                const lastPageEl = pagination.querySelector('.sv-pagination__last a');
                const totalPages = lastPageEl ? parseInt(lastPageEl.textContent) : 1;

                const selectedEl = pagination.querySelector('.sv--selected');
                const currentPage = selectedEl ? parseInt(selectedEl.textContent) : 1;

                // Check for Next button (case-insensitive)
                const links = pagination.querySelectorAll('a');
                let hasNext = false;
                for (const link of links) {
                    if (link.textContent && link.textContent.toLowerCase().includes('next')) {
                        hasNext = true;
                        break;
                    }
                }

                return { totalPages, currentPage, hasNext };
            }''')

            self.stats['pages_scraped'] += 1
            total_pages = pagination_info.get('totalPages', 1)
            has_next = pagination_info.get('hasNext', False)
            detected_page = pagination_info.get('currentPage', current_page)

            new_items = 0
            for card_data in cards_data:
                item = self.parse_card_data(card_data)
                if item:
                    prop_id = item.get('property_id', '')
                    if prop_id in seen_ids:
                        continue  # Skip duplicates within session
                    seen_ids.add(prop_id)

                    if self.is_target_area(item):
                        self.stats['total'] += 1
                        new_items += 1

                        if item.get('price_pcm'):
                            self.stats['prices'].append(item['price_pcm'])
                        if item.get('size_sqft'):
                            self.stats['sqft_found'] += 1

                        # Queue detail page for floorplan extraction, or yield immediately
                        if self.fetch_floorplans and item.get('url'):
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

                        if self.max_properties and self.stats['total'] >= self.max_properties:
                            self.logger.info(
                                f"[COMPLETE] Reached max properties ({self.max_properties})"
                            )
                            await playwright_page.close()
                            return

            self.logger.info(
                f"[DISCOVERY] Page {detected_page}: {len(cards_data)} listings, "
                f"{new_items} new in target areas | "
                f"Total pages: {total_pages} | Has next: {has_next}"
            )

            # Navigate to next page by clicking NEXT button
            if has_next and (self.max_pages is None or current_page < self.max_pages):
                try:
                    # Find and click the Next button (case-insensitive)
                    clicked = await playwright_page.evaluate('''() => {
                        const pagination = document.querySelector('.sv-pagination');
                        if (!pagination) return false;

                        const links = pagination.querySelectorAll('a');
                        for (const link of links) {
                            if (link.textContent && link.textContent.toLowerCase().includes('next')) {
                                link.click();
                                return true;
                            }
                        }
                        return false;
                    }''')

                    if clicked:
                        self.logger.info(f"[PAGINATION] Clicked NEXT, waiting for page {current_page + 1}...")
                        # Wait for page number to change in pagination
                        try:
                            await playwright_page.wait_for_function(
                                f'''() => {{
                                    const selected = document.querySelector('.sv-pagination .sv--selected');
                                    return selected && parseInt(selected.textContent) === {current_page + 1};
                                }}''',
                                timeout=10000
                            )
                            self.logger.info(f"[PAGINATION] Page indicator updated to {current_page + 1}")

                            # CRITICAL: Wait for network to be idle (all fetch requests complete)
                            # This ensures React has finished fetching and rendering new data
                            await playwright_page.wait_for_load_state('networkidle', timeout=15000)
                            self.logger.info(f"[PAGINATION] Network idle for page {current_page + 1}")

                            # Now wait for listings to actually load with fresh data
                            await playwright_page.wait_for_selector(
                                'li.sv-results-listing__item',
                                timeout=10000
                            )
                            # Additional wait for React to fully hydrate content
                            await playwright_page.wait_for_timeout(3000)
                            self.logger.info(f"[PAGINATION] Listings loaded for page {current_page + 1}")
                        except Exception as e:
                            self.logger.warning(f"[PAGINATION] Timeout waiting for page: {e}")
                            # Fallback to fixed wait
                            await playwright_page.wait_for_timeout(5000)
                        current_page += 1
                    else:
                        self.logger.warning("[PAGINATION] Could not click NEXT button")
                        break
                except Exception as e:
                    self.logger.error(f"[PAGINATION] Error clicking next: {e}")
                    break
            else:
                self.logger.info(f"[PAGINATION] No more pages (has_next={has_next}, current={current_page})")
                break

        await playwright_page.close()
        self.logger.info(f"[COMPLETE] Scraped {self.stats['total']} properties from {current_page} pages")

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
        text = card_data.get('text', '')
        address = card_data.get('address', '')

        # Skip if no useful data
        if not href and not text:
            return None

        # Extract property ID from URL
        # Format: /property-detail/gbkerekel220052l (full URL from Savills search)
        id_match = re.search(r'/property-detail/([^/?#]+)', href)
        if id_match:
            prop_id = id_match.group(1)
        else:
            # Issue #28 FIX: Use deterministic hash instead of Python's randomized hash()
            # Fallback - use URL path or deterministic hash for consistency across runs
            prop_id = href.split('/')[-1] if href else f"savills_{hashlib.sha256(text.encode()).hexdigest()[:16]}"

        item['source'] = 'savills'
        item['property_id'] = prop_id

        # URL
        item['url'] = href if href else ''

        # Address - try to extract from text if not in card_data
        if not address:
            lines = text.split('\n')
            for line in lines:
                if re.search(r'[A-Z]{1,2}\d', line):
                    address = line.strip()
                    break
        item['address'] = address

        # Postcode
        postcode = card_data.get('postcode', '')
        if not postcode and address:
            postcode_match = re.search(r'([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d?[A-Z]{0,2}', address.upper())
            postcode = postcode_match.group(1) if postcode_match else ''
        item['postcode'] = postcode

        # Determine area from postcode
        if postcode:
            item['area'] = self.postcode_to_area(postcode)
        else:
            item['area'] = ''

        # Price
        price_str = card_data.get('price', '')
        if price_str:
            try:
                price = int(price_str)
                item['price_pcm'] = price
                item['price_pw'] = int(price * 12 / 52)
                item['price'] = price
                item['price_period'] = 'pcm'
            except ValueError:
                item['price'] = 0
                item['price_pcm'] = 0
                item['price_pw'] = 0
                item['price_period'] = 'pcm'
        else:
            item['price'] = 0
            item['price_pcm'] = 0
            item['price_pw'] = 0
            item['price_period'] = 'pcm'

        # Square footage
        sqft_str = card_data.get('sqft', '')
        if sqft_str:
            try:
                item['size_sqft'] = int(sqft_str)
            except ValueError:
                item['size_sqft'] = None
        else:
            item['size_sqft'] = None

        # Bedrooms
        beds_str = card_data.get('beds', '')
        if beds_str:
            try:
                item['bedrooms'] = int(beds_str)
            except ValueError:
                item['bedrooms'] = None
        else:
            bed_match = re.search(r'(\d+)\s*bed', text, re.I)
            item['bedrooms'] = int(bed_match.group(1)) if bed_match else None

        # Bathrooms
        baths_str = card_data.get('baths', '')
        if baths_str:
            try:
                item['bathrooms'] = int(baths_str)
            except ValueError:
                item['bathrooms'] = None
        else:
            bath_match = re.search(r'(\d+)\s*bath', text, re.I)
            item['bathrooms'] = int(bath_match.group(1)) if bath_match else None

        # Property type
        text_lower = text.lower()
        if 'apartment' in text_lower or 'flat' in text_lower:
            item['property_type'] = 'flat'
        elif 'house' in text_lower:
            item['property_type'] = 'house'
        elif 'studio' in text_lower:
            item['property_type'] = 'studio'
        elif 'penthouse' in text_lower:
            item['property_type'] = 'penthouse'
        else:
            item['property_type'] = 'flat'

        # No coordinates in card view
        item['latitude'] = None
        item['longitude'] = None

        # Agent info
        item['agent_name'] = 'Savills'
        item['agent_phone'] = ''

        # Status
        item['let_agreed'] = 'let agreed' in text_lower or 'under offer' in text_lower

        # Dates
        item['added_date'] = ''
        item['scraped_at'] = datetime.utcnow().isoformat()

        # Additional
        item['summary'] = text[:500] if text else ''

        # Features - extract furnished status and amenities
        features = {}
        if card_data.get('furnished'):
            features['has_furnished'] = card_data['furnished'] == 'furnished'

        # Common amenities
        features['has_gym'] = 'gym' in text_lower
        features['has_porter'] = 'porter' in text_lower or 'concierge' in text_lower
        features['has_parking'] = 'parking' in text_lower
        features['has_balcony'] = 'balcony' in text_lower
        features['has_terrace'] = 'terrace' in text_lower
        features['has_garden'] = 'garden' in text_lower
        features['has_lift'] = 'lift' in text_lower

        item['features'] = json.dumps({k: v for k, v in features.items() if v})

        return item

    def postcode_to_area(self, postcode: str) -> str:
        """Convert postcode to area name.

        Issue #34 FIX: Uses centralized mapping from cli.registry.
        """
        from cli.registry import postcode_to_area as _postcode_to_area
        result = _postcode_to_area(postcode)
        return result if result else postcode

    async def parse_detail(self, response):
        """Parse property detail page to extract floorplan.

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
            floorplan_data = await asyncio.wait_for(
                self._process_detail_page(playwright_page, item),
                timeout=DETAIL_PAGE_TIMEOUT
            )
            if floorplan_data:
                item.update(floorplan_data)

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
        # Wait for page to load
        await playwright_page.wait_for_timeout(2000)

        # Extract floorplan data
        return await self._extract_floorplan_data(playwright_page, item)

    async def _extract_floorplan_data(self, page, item):
        """Click Plans tab and extract floorplan URL and description.

        All page.evaluate() calls use safe_evaluate() with timeouts to prevent hanging.
        """
        data = {}

        try:
            # First, extract description before clicking away from main content
            description = await safe_evaluate(page, '''() => {
                // Savills description is typically in the property description section
                const descSelectors = [
                    '.property-description',
                    '.sv-description',
                    '[class*="description"]',
                    '.full-description',
                    '.property-details__description',
                    'section[class*="description"] p',
                    '.property-summary'
                ];

                for (const selector of descSelectors) {
                    const el = document.querySelector(selector);
                    if (el) {
                        const text = el.innerText.trim();
                        if (text && text.length > 100) {
                            return text;
                        }
                    }
                }

                // Fallback: Look for large text blocks that might be descriptions
                const allDivs = document.querySelectorAll('div, section, p');
                for (const el of allDivs) {
                    const text = el.innerText.trim();
                    if (text.length > 200 && text.length < 3000 &&
                        (text.match(/^(A |An |This |The |Stunning |Beautiful |Spacious |Luxur)/i) ||
                         text.match(/apartment|flat|property|bedroom|floor|interior/i))) {
                        // Make sure it's not a container with lots of nested elements
                        const childCount = el.querySelectorAll('button, a, nav, h1, h2').length;
                        if (childCount < 5) {
                            return text;
                        }
                    }
                }
                return null;
            }''', timeout=15.0)

            if description:
                # Clean up the description
                description = re.sub(r'\s+', ' ', description).strip()
                if len(description) > 5000:
                    description = description[:5000]
                data['description'] = description
                self.logger.debug(f"[DESCRIPTION] Found: {len(description)} chars")

            # Savills has a "Plans" tab - click it using safe_evaluate with timeout
            clicked = await safe_evaluate(page, '''() => {
                const tabs = document.querySelectorAll('[role="tab"], button, a');
                for (const tab of tabs) {
                    const text = tab.innerText || tab.textContent || '';
                    if (text.toLowerCase().includes('plan')) {
                        tab.click();
                        return true;
                    }
                }
                return false;
            }''', timeout=10.0, default=False)

            if clicked:
                await page.wait_for_timeout(2000)

            # Find floorplan URL (assets.savills.com/properties/)
            floorplan_url = await safe_evaluate(page, '''() => {
                // Look for images with savills assets URL
                const imgs = document.querySelectorAll('img');
                for (const img of imgs) {
                    const src = img.src || img.getAttribute('data-src') || '';
                    if (src.includes('assets.savills.com/properties/') &&
                        (src.toLowerCase().includes('floorplan') ||
                         src.toLowerCase().includes('_fp') ||
                         src.toLowerCase().includes('plan'))) {
                        return src;
                    }
                }

                // Check background images
                const html = document.documentElement.innerHTML;
                const match = html.match(/https:\\/\\/assets\\.savills\\.com\\/properties\\/[^"'\\s]+(?:floorplan|_fp|plan)[^"'\\s]*/i);
                return match ? match[0] : null;
            }''', timeout=15.0)

            if floorplan_url:
                floorplan_url = floorplan_url.rstrip('\\').strip()
                data['floorplan_url'] = floorplan_url
                self.stats['floorplans_found'] += 1
                self.logger.debug(f"[FLOORPLAN] Found: {floorplan_url}")

                # Run OCR if sqft is missing (rare for Savills, but possible)
                if not item.get('size_sqft') and OCR_AVAILABLE and self.executor:
                    ocr_data = await self._extract_sqft_via_ocr(floorplan_url)
                    if ocr_data:
                        data.update(ocr_data)
                        if ocr_data.get('size_sqft'):
                            self.stats['sqft_from_ocr'] += 1

        except Exception as e:
            self.logger.debug(f"[FLOORPLAN] Error: {e}")

        return data

    async def _extract_sqft_via_ocr(self, floorplan_url):
        """Download floorplan and extract sqft using OCR."""
        try:
            import requests

            def download_and_extract(url):
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

            # Floor data for ML
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

            return result if result else None

        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.debug(f"[OCR] Error: {e}")
            return None

    def closed(self, reason):
        """Log summary when spider closes."""
        # Issue #21 FIX: Wait for threads to complete to prevent orphan threads
        if self.executor:
            # CRITICAL FIX: Use cancel_futures to prevent indefinite hang
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                self.executor.shutdown(wait=False)

        elapsed = time.time() - self.stats['start_time']
        sqft_pct = (self.stats['sqft_found'] / self.stats['total'] * 100) if self.stats['total'] else 0
        floorplan_pct = (self.stats['floorplans_found'] / self.stats['total'] * 100) if self.stats['total'] else 0

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("SAVILLS SCRAPING COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"[SUMMARY] Duration: {elapsed:.1f}s")
        self.logger.info(f"[SUMMARY] Pages scraped: {self.stats['pages_scraped']}")
        self.logger.info(f"[SUMMARY] Total listings: {self.stats['total']}")
        self.logger.info(f"[SUMMARY] With sqft: {self.stats['sqft_found']} ({sqft_pct:.0f}%)")
        if self.fetch_floorplans:
            self.logger.info(f"[SUMMARY] Floorplans found: {self.stats['floorplans_found']} ({floorplan_pct:.0f}%)")
            self.logger.info(f"[SUMMARY] Sqft from OCR: {self.stats['sqft_from_ocr']}")

        if self.stats['prices']:
            prices = sorted(self.stats['prices'])
            self.logger.info(f"[PRICES] Average: £{sum(prices)//len(prices):,}/pcm")
            self.logger.info(f"[PRICES] Median: £{prices[len(prices)//2]:,}/pcm")

        self.logger.info(f"[CLOSE] Reason: {reason}")
        self.logger.info("=" * 70)
