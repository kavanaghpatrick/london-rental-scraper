"""
John D Wood Spider - Scrapes rental listings from johndwood.co.uk

Uses Playwright to render pages and bypass Cloudflare protection.
John D Wood is a premium London estate agent with excellent floorplan data.

Key Features:
- Extracts sqft AND sqm from floorplan tab
- Captures floorplan image URLs for OCR enrichment
- Extracts individual room dimensions
- High sqft coverage expected (similar to Savills)

Usage:
    scrapy crawl johndwood -a max_properties=100
    scrapy crawl johndwood -a areas=Chelsea,Belgravia

Requires Playwright settings enabled in settings.py
"""

import scrapy
import json
import re
import time
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from property_scraper.items import PropertyItem
from scrapy_playwright.page import PageMethod

# OCR support for floorplan extraction
try:
    import pytesseract
    from PIL import Image
    from property_scraper.utils.floorplan_extractor import FloorplanExtractor, extract_from_floorplan
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class JohnDWoodSpider(scrapy.Spider):
    """Spider for John D Wood rental listings using Playwright."""

    name = 'johndwood'
    allowed_domains = ['johndwood.co.uk']

    # Prime London areas - John D Wood specialty areas
    DEFAULT_AREAS = [
        'chelsea', 'belgravia', 'kensington', 'knightsbridge',
        'south-kensington', 'notting-hill', 'holland-park',
        'mayfair', 'marylebone', 'st-johns-wood', 'hampstead',
        'fulham', 'earls-court', 'pimlico', 'westminster',
    ]

    def __init__(self, max_properties=200, max_pages=20, areas=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_properties = int(max_properties) if max_properties else 200
        self.max_pages = int(max_pages) if max_pages else 20

        # Parse areas from comma-separated string
        if areas:
            self.areas = [a.strip().lower().replace(' ', '-') for a in areas.split(',')]
        else:
            self.areas = self.DEFAULT_AREAS

        self.stats = {
            'total': 0,
            'with_sqft': 0,
            'with_floorplan': 0,
            'pages_scraped': 0,
            'errors': 0,
            'start_time': time.time(),
        }

        self.seen_ids = set()

        # Single thread pool executor for OCR (avoids per-request creation overhead)
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.logger.info("=" * 70)
        self.logger.info("JOHN D WOOD SPIDER INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"[CONFIG] Max properties: {self.max_properties}")
        self.logger.info(f"[CONFIG] Max pages per area: {self.max_pages}")
        self.logger.info(f"[CONFIG] Areas: {', '.join(self.areas)}")
        self.logger.info("=" * 70)

    def start_requests(self):
        """Generate initial request for all lettings."""
        # John D Wood uses a single listings page - no area-based URLs
        url = 'https://www.johndwood.co.uk/properties/lettings'
        self.logger.info(f"[REQUEST] Starting: {url}")

        yield scrapy.Request(
            url,
            callback=self.parse_search,
            meta={
                'playwright': True,
                'playwright_include_page': True,
                'playwright_page_methods': [
                    PageMethod('wait_for_timeout', 8000),
                ],
                'page': 1,
            },
            dont_filter=True,
            errback=self.handle_error,
        )

    def handle_error(self, failure):
        """Handle request failures."""
        self.stats['errors'] += 1
        self.logger.error(f"[ERROR] Request failed: {failure.value}")

    async def parse_search(self, response):
        """Parse search results page and follow property links."""
        page = response.meta.get('page', 1)
        playwright_page = response.meta.get('playwright_page')
        requests_to_yield = []  # Collect requests to yield after page closes

        self.logger.info(f"[SEARCH] Parsing page {page}")

        if not playwright_page:
            self.logger.error("[ERROR] No Playwright page available")
            return

        try:
            # Dismiss cookie banner
            try:
                await playwright_page.click('text=ACCEPT ALL', timeout=3000)
                await playwright_page.wait_for_timeout(1000)
            except:
                pass

            # Wait for listings to load - use .card selector
            try:
                await playwright_page.wait_for_selector('.card', timeout=15000)
            except Exception as e:
                self.logger.warning(f"[SEARCH] No listings found on page {page}: {e}")
                return  # finally block will close page

            # Extract property links from search results
            property_links = await playwright_page.evaluate('''() => {
                const links = [];
                const cards = document.querySelectorAll('a[href*="/properties/"]');

                for (const card of cards) {
                    const href = card.href;
                    if (href && href.includes('/properties/') && href.includes('/lettings/')) {
                        // Extract property ID from URL: /properties/20639639/lettings/SHL250035
                        const match = href.match(/properties\\/(\\d+)\\/lettings\\/([^#\\/]+)/);
                        if (match) {
                            // Also extract area from card text
                            const text = card.innerText || '';
                            const areaMatch = text.match(/,\\s*([A-Za-z\\s]+),\\s*([A-Z]{1,2}\\d)/);

                            links.push({
                                url: href.split('#')[0] + '#/',
                                property_id: match[1],
                                ref: match[2],
                                area: areaMatch ? areaMatch[1].trim() : 'London'
                            });
                        }
                    }
                }

                // Dedupe
                const seen = new Set();
                return links.filter(l => {
                    if (seen.has(l.property_id)) return false;
                    seen.add(l.property_id);
                    return true;
                });
            }''')

            self.logger.info(f"[SEARCH] Found {len(property_links)} properties on page {page}")

            # Collect property requests (don't yield yet - need to close page first)
            properties_queued = 0
            for prop in property_links:
                if self.stats['total'] + properties_queued >= self.max_properties:
                    break

                if prop['property_id'] in self.seen_ids:
                    continue

                self.seen_ids.add(prop['property_id'])
                properties_queued += 1

                requests_to_yield.append(scrapy.Request(
                    prop['url'],
                    callback=self.parse_property,
                    meta={
                        'playwright': True,
                        'playwright_include_page': True,
                        'playwright_page_methods': [
                            PageMethod('wait_for_timeout', 5000),
                        ],
                        'area': prop.get('area', 'London'),
                        'property_id': prop['property_id'],
                        'ref': prop['ref'],
                    },
                    priority=1,  # HIGH priority - process properties before more search pages
                    dont_filter=True,
                    errback=self.handle_error,
                ))

            self.stats['pages_scraped'] += 1

            # Check for "Load More" or pagination
            # Use seen_ids count (queued) not stats['total'] (completed) to avoid over-queuing
            if page < self.max_pages and len(self.seen_ids) < self.max_properties:
                has_more = await playwright_page.evaluate('''() => {
                    const loadMore = document.querySelector('button[class*="load"], .load-more, [data-action="load-more"]');
                    return !!loadMore;
                }''')

                if has_more:
                    try:
                        await playwright_page.click('button[class*="load"], .load-more, [data-action="load-more"]')
                        await playwright_page.wait_for_timeout(3000)

                        # Queue pagination request
                        requests_to_yield.append(scrapy.Request(
                            response.url,
                            callback=self.parse_search,
                            meta={
                                'playwright': True,
                                'playwright_include_page': True,
                                'playwright_page_methods': [
                                    PageMethod('wait_for_timeout', 5000),
                                ],
                                'page': page + 1,
                            },
                            priority=-1,  # LOW priority - process properties first
                            dont_filter=True,
                            errback=self.handle_error,
                        ))
                    except:
                        pass

        finally:
            # ALWAYS close page, even on exceptions
            await playwright_page.close()

        # Yield all requests after page is closed
        for req in requests_to_yield:
            yield req

    async def parse_property(self, response):
        """Parse individual property page and extract all data."""
        area = response.meta.get('area', 'unknown')
        property_id = response.meta.get('property_id')
        ref = response.meta.get('ref')
        playwright_page = response.meta.get('playwright_page')

        self.logger.info(f"[PROPERTY] Parsing {property_id} ({ref})")

        if not playwright_page:
            self.logger.error(f"[ERROR] No Playwright page for {property_id}")
            return

        try:
            # Dismiss cookie banner if present
            try:
                await playwright_page.click('text=ACCEPT ALL', timeout=3000)
                await playwright_page.wait_for_timeout(1000)
            except:
                pass

            # Extract basic property data from page
            property_data = await playwright_page.evaluate('''() => {
                const data = {};
                const text = document.body.innerText || '';

                // Address - from title or header
                const titleEl = document.querySelector('h1, .property-title, [class*="address"]');
                data.address = titleEl ? titleEl.innerText.trim() : null;

                // Price - look for £X,XXX pw or pcm patterns
                // John D Wood typically shows weekly prices (pw)
                const priceMatch = text.match(/£([\\d,]+)\\s*(pw|pcm|p\\.?w\\.?|p\\.?c\\.?m\\.?|per\\s*week|per\\s*month|weekly|monthly)?/i);
                if (priceMatch) {
                    data.price = parseInt(priceMatch[1].replace(/,/g, ''));
                    const period = (priceMatch[2] || '').toLowerCase();
                    // Default to pw (weekly) for John D Wood if no period specified
                    // Only treat as monthly if explicitly says pcm/month
                    if (period.includes('month') || period.includes('pcm')) {
                        data.price_period = 'pcm';
                    } else {
                        // Default to weekly - John D Wood standard
                        data.price_period = 'pw';
                    }
                }

                // Bedrooms, bathrooms, receptions - from icons/stats
                const bedsMatch = text.match(/(\\d+)\\s*(?:bed|bedroom)/i);
                const bathsMatch = text.match(/(\\d+)\\s*(?:bath|bathroom)/i);
                const recMatch = text.match(/(\\d+)\\s*(?:reception|living)/i);

                data.bedrooms = bedsMatch ? parseInt(bedsMatch[1]) : null;
                data.bathrooms = bathsMatch ? parseInt(bathsMatch[1]) : null;
                data.reception_rooms = recMatch ? parseInt(recMatch[1]) : null;

                // Property type
                const typeMatch = text.match(/(flat|apartment|house|studio|maisonette|duplex|penthouse)/i);
                data.property_type = typeMatch ? typeMatch[1].toLowerCase() : null;

                // Furnished status
                if (text.toLowerCase().includes('unfurnished')) {
                    data.furnished = 'unfurnished';
                } else if (text.toLowerCase().includes('part furnished') || text.toLowerCase().includes('part-furnished')) {
                    data.furnished = 'part-furnished';
                } else if (text.toLowerCase().includes('furnished')) {
                    data.furnished = 'furnished';
                }

                // Postcode - SW1W 8AA pattern
                const pcMatch = text.match(/([A-Z]{1,2}\\d{1,2}[A-Z]?)\\s*(\\d[A-Z]{2})/);
                data.postcode = pcMatch ? pcMatch[1] + ' ' + pcMatch[2] : null;

                // Description
                const descEl = document.querySelector('.property-description, [class*="description"], .about-property');
                data.description = descEl ? descEl.innerText.trim().slice(0, 2000) : null;

                // Agent info
                const agentEl = document.querySelector('.agent-name, [class*="office"], [class*="branch"]');
                data.agent_name = agentEl ? 'John D Wood - ' + agentEl.innerText.trim() : 'John D Wood';

                const phoneEl = document.querySelector('a[href^="tel:"]');
                data.agent_phone = phoneEl ? phoneEl.innerText.trim() : null;

                // EPC rating
                const epcMatch = text.match(/EPC[:\\s]*([A-G])/i);
                data.epc_rating = epcMatch ? epcMatch[1].toUpperCase() : null;

                // Features list
                const features = [];
                const featureEls = document.querySelectorAll('.feature, .amenity, li[class*="feature"]');
                featureEls.forEach(el => {
                    const feat = el.innerText.trim();
                    if (feat && feat.length < 100) features.push(feat);
                });
                data.features = features.slice(0, 20);

                return data;
            }''')

            # Click on FLOORPLANS tab to get sqft/sqm data
            floorplan_data = await self._extract_floorplan_data(playwright_page)

            # Merge data
            property_data.update(floorplan_data)

            # Build PropertyItem
            item = PropertyItem()
            item['source'] = 'johndwood'
            item['property_id'] = property_id
            item['url'] = response.url
            item['area'] = self._normalize_area(area)

            # Pricing
            if property_data.get('price'):
                if property_data.get('price_period') == 'pw':
                    item['price_pw'] = property_data['price']
                    item['price_pcm'] = int(property_data['price'] * 52 / 12)
                else:
                    item['price_pcm'] = property_data['price']
                    item['price_pw'] = int(property_data['price'] * 12 / 52)
                item['price'] = item.get('price_pcm')

            # Location
            item['address'] = property_data.get('address')
            item['postcode'] = property_data.get('postcode')

            # Property details
            item['bedrooms'] = property_data.get('bedrooms')
            item['bathrooms'] = property_data.get('bathrooms')
            item['reception_rooms'] = property_data.get('reception_rooms')
            item['property_type'] = property_data.get('property_type')
            item['furnished'] = property_data.get('furnished')
            item['epc_rating'] = property_data.get('epc_rating')

            # Size from floorplan
            item['size_sqft'] = property_data.get('size_sqft')
            item['size_sqm'] = property_data.get('size_sqm')
            item['floorplan_url'] = property_data.get('floorplan_url')
            item['room_details'] = property_data.get('room_details')

            # Binary floor data for ML training
            item['has_basement'] = property_data.get('has_basement')
            item['has_lower_ground'] = property_data.get('has_lower_ground')
            item['has_ground'] = property_data.get('has_ground')
            item['has_mezzanine'] = property_data.get('has_mezzanine')
            item['has_first_floor'] = property_data.get('has_first_floor')
            item['has_second_floor'] = property_data.get('has_second_floor')
            item['has_third_floor'] = property_data.get('has_third_floor')
            item['has_fourth_plus'] = property_data.get('has_fourth_plus')
            item['has_roof_terrace'] = property_data.get('has_roof_terrace')
            item['floor_count'] = property_data.get('floor_count')
            item['property_levels'] = property_data.get('property_levels')

            # Content
            item['description'] = property_data.get('description')
            item['features'] = property_data.get('features', [])
            item['agent_name'] = property_data.get('agent_name', 'John D Wood')
            item['agent_phone'] = property_data.get('agent_phone')

            # Timestamps
            item['scraped_at'] = datetime.utcnow().isoformat()

            # Update stats
            self.stats['total'] += 1
            if item.get('size_sqft'):
                self.stats['with_sqft'] += 1
            if item.get('floorplan_url'):
                self.stats['with_floorplan'] += 1

            self.logger.info(
                f"[YIELD] {property_id} | "
                f"{item.get('bedrooms', '?')} bed | "
                f"£{item.get('price_pcm', 0):,} pcm | "
                f"{item.get('size_sqft', 'no')} sqft | "
                f"{item.get('postcode', 'no PC')}"
            )

            yield item

        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"[ERROR] Failed to parse {property_id}: {e}")

        finally:
            await playwright_page.close()

    async def _extract_floorplan_data(self, page):
        """Click FLOORPLANS tab and extract size data with OCR fallback."""
        data = {}

        try:
            # Click FLOORPLANS tab
            await page.click('text=FLOORPLANS', timeout=5000)
            await page.wait_for_timeout(2000)

            # Extract sqft/sqm from page text first (fast)
            text = await page.evaluate('() => document.body.innerText')

            # Pattern: "1540 Sq Ft - 143.07 Sq M" or similar
            sqft_match = re.search(r'(\d[\d,]*)\s*(?:Sq\.?\s*Ft|sq\.?\s*ft|sqft)', text)
            sqm_match = re.search(r'(\d[\d,.]*)\s*(?:Sq\.?\s*M|sq\.?\s*m|sqm)', text)

            if sqft_match:
                data['size_sqft'] = int(sqft_match.group(1).replace(',', ''))
            if sqm_match:
                sqm_str = sqm_match.group(1).replace(',', '')
                data['size_sqm'] = int(float(sqm_str))

            # Get floorplan image URL - search all elements for homeflow floorplan pattern
            floorplan_url = await page.evaluate('''() => {
                // Search all elements for floorplan URLs
                const allElements = document.querySelectorAll('*');
                for (const el of allElements) {
                    // Check img src
                    if (el.tagName === 'IMG') {
                        const src = el.src || el.getAttribute('data-src') || '';
                        if (src.includes('floorplan')) {
                            return src.startsWith('//') ? 'https:' + src : src;
                        }
                    }

                    // Check background-image style
                    const style = window.getComputedStyle(el);
                    const bgImage = style.backgroundImage;
                    if (bgImage && bgImage.includes('floorplan')) {
                        const match = bgImage.match(/url\\(['"]?([^'"\\)]+)['"]?\\)/);
                        if (match) {
                            const url = match[1];
                            return url.startsWith('//') ? 'https:' + url : url;
                        }
                    }
                }
                return null;
            }''')

            if floorplan_url:
                data['floorplan_url'] = floorplan_url

                # If no sqft found in text, try OCR on the floorplan image
                if not data.get('size_sqft') and OCR_AVAILABLE:
                    self.logger.debug(f"[FLOORPLAN] No sqft in text, trying OCR on {floorplan_url}")
                    ocr_data = await self._extract_sqft_via_ocr(page, floorplan_url)
                    if ocr_data:
                        data.update(ocr_data)

            # Extract room dimensions from floorplan area text
            room_details = self._extract_room_details(text)
            if room_details:
                data['room_details'] = room_details

        except Exception as e:
            self.logger.debug(f"[FLOORPLAN] Could not extract: {e}")

        return data

    async def _extract_sqft_via_ocr(self, page, floorplan_url):
        """Download floorplan image and extract sqft using OCR (non-blocking)."""
        try:
            import requests
            import asyncio

            def download_and_extract(url):
                """Combined download + OCR in thread pool to avoid blocking event loop."""
                # Download image
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    return None

                # Run OCR extraction (this is CPU-bound, must be in thread pool)
                extractor = FloorplanExtractor()
                return extractor.extract_from_bytes(resp.content)

            loop = asyncio.get_event_loop()
            # Use class-level executor and wrap entire operation in timeout
            floorplan_data = await asyncio.wait_for(
                loop.run_in_executor(self.executor, download_and_extract, floorplan_url),
                timeout=20.0  # 20 second timeout for download + OCR combined
            )

            if not floorplan_data:
                self.logger.debug(f"[OCR] Failed to download/extract floorplan")
                return None

            result = {}

            # Extract total area
            if floorplan_data.total_sqft:
                result['size_sqft'] = floorplan_data.total_sqft
                self.logger.info(f"[OCR] Extracted {floorplan_data.total_sqft} sqft via OCR")

            if floorplan_data.total_sqm:
                result['size_sqm'] = floorplan_data.total_sqm

            # Extract comprehensive room data
            if floorplan_data.rooms:
                room_details = []
                for room in floorplan_data.rooms:
                    room_info = {
                        'type': room.type,
                        'name': room.name,
                    }
                    if room.dimensions_imperial:
                        room_info['dimensions'] = room.dimensions_imperial
                    if room.sqft:
                        room_info['sqft'] = room.sqft
                    room_details.append(room_info)

                if room_details:
                    result['room_details'] = room_details

            # Add room counts
            if floorplan_data.room_counts:
                result['room_counts'] = floorplan_data.room_counts

            # Add special features
            if floorplan_data.special_features:
                result['special_features'] = floorplan_data.special_features

            # Add binary floor data for ML training
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

                if fd.floor_count > 0:
                    self.logger.info(f"[OCR] Detected {fd.floor_count} floors: {fd.property_levels}")

            return result if result else None

        except ImportError as e:
            self.logger.warning(f"[OCR] Missing dependency for image download: {e}")
            return None
        except Exception as e:
            if "TimeoutError" in type(e).__name__:
                self.logger.debug(f"[OCR] Timeout downloading floorplan image")
            else:
                self.logger.debug(f"[OCR] Failed to extract via OCR: {e}")
            return None

    def _extract_room_details(self, text):
        """Extract individual room dimensions from floorplan text."""
        rooms = []

        # Pattern: "Reception Room 25' x 14'6" (7.62m x 4.42m)"
        # or "Bedroom 11'9" x 10'3" 3.58m x 3.12m"
        room_pattern = r'(Reception|Bedroom|Kitchen|Bathroom|Master|Living|Dining|Study|Utility|Entrance|Hall)[\s\w]*?(\d+[\'\"]?\s*[x×]\s*\d+[\'\"]?)'

        matches = re.findall(room_pattern, text, re.IGNORECASE)

        for room_type, dimensions in matches[:10]:  # Limit to 10 rooms
            rooms.append({
                'type': room_type.strip().lower(),
                'dimensions': dimensions.strip()
            })

        return rooms if rooms else None

    def _normalize_area(self, area):
        """Convert URL-style area to display name."""
        return area.replace('-', ' ').title()

    def closed(self, reason):
        """Log final statistics when spider closes."""
        # Clean up the thread pool executor
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)

        elapsed = time.time() - self.stats['start_time']

        self.logger.info("=" * 70)
        self.logger.info("JOHN D WOOD SPIDER COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Total properties: {self.stats['total']}")
        self.logger.info(f"With sqft: {self.stats['with_sqft']} ({self.stats['with_sqft']*100/max(1,self.stats['total']):.1f}%)")
        self.logger.info(f"With floorplan URL: {self.stats['with_floorplan']}")
        self.logger.info(f"Pages scraped: {self.stats['pages_scraped']}")
        self.logger.info(f"Errors: {self.stats['errors']}")
        self.logger.info(f"Duration: {elapsed:.1f}s")
        self.logger.info(f"Close reason: {reason}")
        self.logger.info("=" * 70)
