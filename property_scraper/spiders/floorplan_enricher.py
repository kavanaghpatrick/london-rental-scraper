"""
Floorplan Enricher Spider - Adds floorplan URLs to existing listings

This spider reads existing listings from the database and fetches
detail pages to extract floorplan URLs. Supports multiple sources:
- knightfrank: Uses Playwright, extracts from JS-rendered content
- chestertons: Uses Playwright, extracts from property cards
- savills: Uses Playwright, clicks "Plans" tab
- foxtons: Uses standard HTTP, extracts from JSON (no Playwright needed)

Usage:
    scrapy crawl floorplan_enricher -a source=knightfrank
    scrapy crawl floorplan_enricher -a source=savills -a limit=100
    scrapy crawl floorplan_enricher -a source=foxtons
"""

import scrapy
import sqlite3
import os
import json
import re
import time
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# OCR support for floorplan extraction
try:
    from property_scraper.utils.floorplan_extractor import FloorplanExtractor
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def _load_sqft_sanity_gate():
    """Import sqft_passes_sanity_gate from scripts/ocr_enrich.py (single source of
    truth for the [150,10000]+ppsf/spb window). scripts/ is not a package, so load it
    by path — same pattern as tests/test_ocr_enrich_guards.py. Importing it (not
    redefining the constants) keeps every writer on ONE gate."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "ocr_enrich_gate", str(root / "scripts" / "ocr_enrich.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sqft_passes_sanity_gate


sqft_passes_sanity_gate = _load_sqft_sanity_gate()


class FloorplanEnricherSpider(scrapy.Spider):
    """Spider to add floorplan URLs to existing listings."""

    name = 'floorplan_enricher'

    # Source-specific THROTTLING settings to avoid rate limiting
    # Chestertons is particularly aggressive with 429 responses
    SOURCE_THROTTLE = {
        'chestertons': {
            'CONCURRENT_REQUESTS': 1,  # One request at a time
            'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
            'DOWNLOAD_DELAY': 10,  # 10 seconds between requests
            'AUTOTHROTTLE_MAX_DELAY': 300,  # Up to 5 minutes if needed
            'AUTOTHROTTLE_TARGET_CONCURRENCY': 0.5,  # Very conservative
        },
        # Other sources use default settings (more lenient)
    }

    # Source-specific settings
    SOURCE_CONFIG = {
        'knightfrank': {
            'domain': 'knightfrank.co.uk',
            'needs_playwright': True,
            'url_template': None,  # Use URL from DB
        },
        'chestertons': {
            # Issue #30 FIX: Correct domain (was chestertons.com, URLs are chestertons.co.uk)
            'domain': 'chestertons.co.uk',
            'needs_playwright': True,
            'url_template': None,
        },
        'savills': {
            'domain': 'savills.com',
            'needs_playwright': True,
            'url_template': None,
        },
        'foxtons': {
            'domain': 'foxtons.co.uk',
            'needs_playwright': False,
            'url_template': None,  # Use database URLs (format: /properties-to-rent/{area}/{property_id})
        },
        'rightmove': {
            'domain': 'rightmove.co.uk',
            'needs_playwright': False,
            'url_template': 'https://www.rightmove.co.uk/properties/{property_id}',
        },
    }

    def __init__(self, source=None, limit=None, use_ocr=False, db_path=None,
                 postgres=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not source or source not in self.SOURCE_CONFIG:
            raise ValueError(f"Must specify source: {list(self.SOURCE_CONFIG.keys())}")

        self.source = source
        self.source_config = self.SOURCE_CONFIG[source]
        self.allowed_domains = [self.source_config['domain']]
        self.limit = int(limit) if limit else None
        # db_path defaults to the canonical DB; override via -a db_path=... for testing
        self.db_path = db_path or 'output/rentals.db'
        self.use_ocr = str(use_ocr).lower() in ('true', '1', 'yes') and OCR_AVAILABLE

        # Postgres mode: scheduled (daily-scrape) runs write PROD Postgres, the same
        # DB that scripts/ocr_enrich.py reads. Without this the captured floorplan_url
        # would land in local SQLite and the OCR step would still starve (#WS3).
        # Gated on BOTH -a postgres=true AND a POSTGRES_URL in the env, so a stray
        # env var on a local run never accidentally writes prod.
        flag_pg = str(postgres).lower() in ('true', '1', 'yes')
        self.postgres_url = os.environ.get('POSTGRES_URL')
        self.use_postgres = flag_pg and bool(self.postgres_url)
        if flag_pg and not self.postgres_url:
            self.logger.error(
                "[CONFIG] postgres=true but POSTGRES_URL is not set — "
                "falling back to SQLite (%s)", self.db_path
            )

        # ThreadPoolExecutor for OCR
        self.executor = ThreadPoolExecutor(max_workers=2) if self.use_ocr else None

        # Stats
        self.stats = {
            'total_to_enrich': 0,
            'enriched': 0,
            'floorplans_found': 0,
            'descriptions_found': 0,
            'sqft_from_ocr': 0,
            'failed': 0,
            'start_time': time.time(),
        }

        self.logger.info("=" * 70)
        self.logger.info(f"FLOORPLAN ENRICHER - {source.upper()}")
        self.logger.info("=" * 70)
        self.logger.info(f"[CONFIG] Source: {source}")
        self.logger.info(f"[CONFIG] Needs Playwright: {self.source_config['needs_playwright']}")
        if self.limit:
            self.logger.info(f"[CONFIG] Limit: {self.limit}")
        self.logger.info(f"[CONFIG] OCR available: {OCR_AVAILABLE}")
        self.logger.info(f"[CONFIG] Use OCR: {self.use_ocr}")

        # Apply source-specific throttling if defined
        if source in self.SOURCE_THROTTLE:
            throttle = self.SOURCE_THROTTLE[source]
            self.logger.info(f"[THROTTLE] Applying aggressive throttling for {source}:")
            self.logger.info(f"[THROTTLE]   CONCURRENT_REQUESTS: {throttle.get('CONCURRENT_REQUESTS', 'default')}")
            self.logger.info(f"[THROTTLE]   DOWNLOAD_DELAY: {throttle.get('DOWNLOAD_DELAY', 'default')}s")

    @classmethod
    def update_settings(cls, settings):
        """Apply source-specific throttle settings from spider args."""
        # This is called before __init__, so we need to check args differently
        # The source will be set via -a source=X, which we can access via
        # the spider's attributes after construction
        pass

    def _get_throttle_settings(self):
        """Get custom settings for this source."""
        if self.source in self.SOURCE_THROTTLE:
            return self.SOURCE_THROTTLE[self.source]
        return {}

    def _get_connection(self):
        """Open a DB connection: Postgres (prod) when use_postgres, else SQLite.

        Returns (conn, placeholder) where placeholder is the param token ('%s'
        for Postgres, '?' for SQLite). Caller closes the connection.
        """
        if self.use_postgres:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(self.postgres_url)
            return conn, '%s', psycopg2.extras
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn, '?', None

    def start_requests(self):
        """Read properties from database that need enrichment (floorplan or description).

        CRITICAL FIX: Use try/finally to ensure database connection is always closed,
        even if query execution fails.
        """
        conn = None
        rows = []

        db_label = 'Postgres' if self.use_postgres else f'SQLite ({self.db_path})'
        self.logger.info(f"[START] Reading listings from {db_label}")

        try:
            conn, ph, pg_extras = self._get_connection()
            if self.use_postgres:
                cursor = conn.cursor(cursor_factory=pg_extras.RealDictCursor)
            else:
                cursor = conn.cursor()

            # Find listings missing floorplan URL OR description.
            # LENGTH() is portable across SQLite and Postgres.
            query = f'''
                SELECT property_id, url, address, price_pcm, bedrooms, size_sqft,
                       floorplan_url, description
                FROM listings
                WHERE source = {ph}
                AND (
                    (floorplan_url IS NULL OR floorplan_url = '')
                    OR (description IS NULL OR description = '')
                )
                AND url IS NOT NULL AND LENGTH(url) > 10
            '''
            params = [self.source]

            if self.limit:
                query += f' LIMIT {int(self.limit)}'

            cursor.execute(query, params)
            fetched = cursor.fetchall()
            # Normalise to plain dicts so downstream row['col'] works for both
            # sqlite3.Row and psycopg2 RealDictRow.
            rows = [dict(r) for r in fetched]
        except Exception as e:
            self.logger.error(f"[START] Database error: {e}")
            rows = []
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

        self.stats['total_to_enrich'] = len(rows)
        self.logger.info(f"[START] Found {len(rows)} {self.source} listings needing enrichment")

        for i, row in enumerate(rows):
            if i % 50 == 0:
                self.logger.info(f"[QUEUE] Queuing {i+1}/{len(rows)}...")

            # Build URL
            url = row['url']
            if self.source_config['url_template']:
                url = self.source_config['url_template'].format(property_id=row['property_id'])

            meta = {
                'property_id': row['property_id'],
                'address': row['address'],
                'price_pcm': row['price_pcm'],
                'bedrooms': row['bedrooms'],
                'existing_sqft': row['size_sqft'],
                'request_start': time.time()
            }

            # Add Playwright meta if needed
            if self.source_config['needs_playwright']:
                meta['playwright'] = True
                meta['playwright_include_page'] = True

            yield scrapy.Request(
                url,
                callback=self.parse_detail,
                meta=meta,
                dont_filter=True,
                errback=self.handle_error
            )

    def handle_error(self, failure):
        """Handle request failures."""
        prop_id = failure.request.meta.get('property_id', 'unknown')
        self.stats['failed'] += 1
        self.logger.error(f"[ERROR] Failed to fetch {prop_id}: {failure.value}")

    async def parse_detail(self, response):
        """Parse property detail page for floorplan URL."""
        prop_id = response.meta['property_id']
        playwright_page = response.meta.get('playwright_page')

        if response.status != 200:
            self.stats['failed'] += 1
            self.logger.warning(f"[HTTP-ERROR] {prop_id}: status {response.status}")
            if playwright_page:
                await playwright_page.close()
            return

        floorplan_url = None
        description = None

        try:
            # Source-specific extraction - now extracts both floorplan and description
            if self.source == 'knightfrank':
                floorplan_url, description = await self._extract_knightfrank(playwright_page, response)
            elif self.source == 'chestertons':
                floorplan_url, description = await self._extract_chestertons(playwright_page, response)
            elif self.source == 'savills':
                floorplan_url, description = await self._extract_savills(playwright_page, response)
            elif self.source == 'foxtons':
                floorplan_url, description = self._extract_foxtons(response)
            elif self.source == 'rightmove':
                floorplan_url, description = self._extract_rightmove(response)

        except Exception as e:
            self.logger.debug(f"[PARSE] {prop_id}: extraction error - {e}")

        finally:
            if playwright_page:
                await playwright_page.close()

        self.stats['enriched'] += 1

        # Update database with whatever we found
        if floorplan_url or description:
            if floorplan_url:
                self.stats['floorplans_found'] += 1
            if description:
                self.stats['descriptions_found'] += 1
            self.update_database(prop_id, floorplan_url, description)
            self.logger.debug(f"[FOUND] {prop_id}: floorplan={bool(floorplan_url)}, desc={len(description) if description else 0} chars")

            # OCR if no existing sqft
            # Issue #31 FIX: Run OCR in executor to avoid blocking the async event loop
            if self.use_ocr and floorplan_url and not response.meta.get('existing_sqft'):
                loop = asyncio.get_event_loop()
                ocr_sqft = await loop.run_in_executor(
                    self.executor,
                    self._extract_sqft_via_ocr,
                    floorplan_url
                )
                # Gate the OCR value through the single sanity gate BEFORE writing:
                # an out-of-[150,10000] (or economically impossible) reading is a
                # sqm-as-sqft / scan artifact and must be NULLED-not-written. beds +
                # price are already in meta; pass None where absent (gate skips that check).
                gated = sqft_passes_sanity_gate(
                    ocr_sqft,
                    response.meta.get('bedrooms'),
                    response.meta.get('price_pcm'),
                )
                if gated is not None:
                    self.stats['sqft_from_ocr'] += 1
                    self.update_database_sqft(prop_id, gated)
                    self.logger.debug(f"[OCR] {prop_id}: extracted {gated} sqft")
                elif ocr_sqft:
                    self.logger.debug(
                        f"[OCR-REJECT] {prop_id}: {ocr_sqft} sqft failed sanity gate"
                    )
        else:
            self.logger.debug(f"[MISS] {prop_id}: no floorplan or description found")

        # Progress logging
        if self.stats['enriched'] % 25 == 0:
            pct = (self.stats['floorplans_found'] / self.stats['enriched'] * 100) if self.stats['enriched'] else 0
            self.logger.info(
                f"[PROGRESS] {self.stats['enriched']}/{self.stats['total_to_enrich']} | "
                f"Floorplans: {self.stats['floorplans_found']} ({pct:.0f}%)"
            )

    async def _extract_knightfrank(self, page, response):
        """Extract floorplan URL and description from Knight Frank detail page."""
        if not page:
            return None, None

        await page.wait_for_timeout(2000)

        # Extract description first (before navigating away from main content)
        description = await page.evaluate('''() => {
            // Look for property description section
            const descEl = document.querySelector('.property-description, .kf-description, [class*="description"]');
            if (descEl) return descEl.innerText.trim();

            // Try data-testid or common patterns
            const sections = document.querySelectorAll('[class*="PropertyDetails"], [class*="property-details"]');
            for (const sec of sections) {
                const text = sec.innerText.trim();
                if (text.length > 100) return text;
            }
            return null;
        }''')

        # Click floorplan tab
        clicked = await page.evaluate('''() => {
            const tabs = document.querySelectorAll('button, [role="tab"], a');
            for (const tab of tabs) {
                const text = (tab.innerText || tab.textContent || '').toLowerCase();
                if (text.includes('floor') && text.includes('plan')) {
                    tab.click();
                    return true;
                }
            }
            return false;
        }''')

        if clicked:
            await page.wait_for_timeout(2000)

        # Find floorplan URL
        floorplan_url = await page.evaluate('''() => {
            const imgs = document.querySelectorAll('img');
            for (const img of imgs) {
                const src = img.src || img.getAttribute('data-src') || '';
                if (src.includes('content.knightfrank.com') &&
                    (src.includes('floorplan') || src.includes('_FLP_'))) {
                    return src;
                }
            }
            // Check HTML for pattern
            const html = document.documentElement.innerHTML;
            const match = html.match(/https:\/\/content\.knightfrank\.com\/[^"'\s]+floorplan[^"'\s]*/i);
            return match ? match[0] : null;
        }''')

        return floorplan_url, description

    async def _extract_chestertons(self, page, response):
        """Extract floorplan URL and description from Chestertons detail page."""
        if not page:
            return None, None

        # Check if page is still valid
        try:
            if page.is_closed():
                self.logger.debug("[CHESTERTONS] Page already closed")
                return None, None
        except Exception:
            return None, None

        # Skip Cloudflare check - pages load fine with headed browser
        # Just wait a moment for page to fully render
        await page.wait_for_timeout(3000)

        await page.wait_for_timeout(2000)

        # Extract description first - WITH TIMEOUT
        # Test script found descriptions in <div class="mb-24"> after tab headers
        description = None
        try:
            description = await asyncio.wait_for(
                page.evaluate('''() => {
                    // Chestertons uses mb-24 divs - find ones with property description text
                    const mb24Divs = document.querySelectorAll('.mb-24, [class*="mb-24"]');
                    for (const div of mb24Divs) {
                        const text = div.innerText || '';
                        // Skip nav/header elements - look for actual property description
                        if (text.length > 200 &&
                            !text.startsWith('Directions') &&
                            !text.startsWith('Share Property')) {
                            // Extract just the description part (after tab headers)
                            const lines = text.split('\\n').filter(l => l.trim());
                            // Skip tab headers like "Property Details", "Location & Nearby", etc.
                            const descStart = lines.findIndex(l =>
                                l.length > 50 &&
                                !['Property Details', 'Location & Nearby', 'EPC', 'Brochure',
                                  'Directions', 'Share Property', 'Back to results'].includes(l.trim())
                            );
                            if (descStart >= 0) {
                                return lines.slice(descStart).join(' ').trim();
                            }
                        }
                    }

                    // Fallback: Look for any large text block that looks like a description
                    const allDivs = document.querySelectorAll('div, section, article');
                    for (const el of allDivs) {
                        const text = (el.innerText || '').trim();
                        // Property descriptions typically start with certain patterns
                        if (text.length > 150 && text.length < 3000 &&
                            (text.match(/^(A |An |This |The |Stunning |Beautiful |Spacious |Luxur)/i) ||
                             text.match(/apartment|flat|property|bedroom|floor|interior/i))) {
                            // Make sure it's not containing navigation elements
                            const childCount = el.querySelectorAll('button, a, nav').length;
                            if (childCount < 3) {
                                return text;
                            }
                        }
                    }

                    return null;
                }'''),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            self.logger.debug("[CHESTERTONS] Description extraction timed out")
        except Exception as e:
            self.logger.debug(f"[CHESTERTONS] Description error: {e}")

        # Click floor plans tab - WITH TIMEOUT
        clicked = False
        try:
            clicked = await asyncio.wait_for(
                page.evaluate('''() => {
                    const tabs = document.querySelectorAll('button, [role="tab"], a, .tab');
                    for (const tab of tabs) {
                        const text = (tab.innerText || tab.textContent || '').toLowerCase();
                        if (text.includes('floor') && text.includes('plan')) {
                            tab.click();
                            return true;
                        }
                    }
                    return false;
                }'''),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            self.logger.debug("[CHESTERTONS] Tab click timed out")
        except Exception as e:
            self.logger.debug(f"[CHESTERTONS] Tab click error: {e}")

        if clicked:
            await page.wait_for_timeout(2000)

        # Find floorplan URL - WITH TIMEOUT
        floorplan_url = None
        try:
            floorplan_url = await asyncio.wait_for(
                page.evaluate('''() => {
                    const imgs = document.querySelectorAll('img');
                    for (const img of imgs) {
                        const src = img.src || img.getAttribute('data-src') || '';
                        if (src.includes('homeflow-assets.co.uk') &&
                            src.includes('floorplan')) {
                            return src;
                        }
                    }
                    return null;
                }'''),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            self.logger.debug("[CHESTERTONS] Floorplan extraction timed out")
        except Exception as e:
            self.logger.debug(f"[CHESTERTONS] Floorplan error: {e}")

        return floorplan_url, description

    async def _extract_savills(self, page, response):
        """Extract floorplan URL and description from Savills detail page.

        Savills floorplans are GIF files that appear after clicking the PLANS button.
        The floorplan can be identified by:
        1. Alt text contains "Floorplan"
        2. File extension is .GIF (photos are .jpg)
        3. Image is from assets.savills.com/properties/
        """
        if not page:
            return None, None

        await page.wait_for_timeout(2000)

        # Extract description first (before clicking away)
        description = await page.evaluate('''() => {
            // Savills uses sv- prefix for classes
            const descEl = document.querySelector('.sv-property-description, .sv-description, [class*="description"]');
            if (descEl) return descEl.innerText.trim();

            // Try to find main property content
            const sections = document.querySelectorAll('.sv-property-details, .property-details, .sv-content');
            for (const sec of sections) {
                const text = sec.innerText.trim();
                if (text.length > 100) return text;
            }
            return null;
        }''')

        # Click the PLANS button (not a tab - it's a button with specific class)
        clicked = await page.evaluate('''() => {
            // Look for PLANS button specifically
            const buttons = document.querySelectorAll('button');
            for (const btn of buttons) {
                const text = (btn.innerText || btn.textContent || '').trim().toLowerCase();
                // Exact match for "plans" to avoid "planning"
                if (text === 'plans') {
                    btn.click();
                    return true;
                }
            }
            return false;
        }''')

        if clicked:
            await page.wait_for_timeout(3000)  # Wait for floorplan to load

        # Find floorplan URL - look for GIF files or images with "Floorplan" in alt
        floorplan_url = await page.evaluate('''() => {
            const imgs = document.querySelectorAll('img');
            for (const img of imgs) {
                const src = img.src || '';
                const alt = img.alt || '';

                // Must be from savills properties assets
                if (!src.includes('assets.savills.com/properties/')) continue;

                // Strategy 1: Alt text contains "Floorplan"
                if (alt.toLowerCase().includes('floorplan')) {
                    return src;
                }

                // Strategy 2: GIF file (floorplans are GIFs, photos are JPGs)
                if (src.toLowerCase().endsWith('.gif')) {
                    return src;
                }
            }

            // Strategy 3: Search HTML for GIF URLs from properties
            const html = document.documentElement.innerHTML;
            const match = html.match(/https:\/\/assets\.savills\.com\/properties\/[A-Z0-9]+\/[^"'\s]+\.gif/i);
            return match ? match[0] : null;
        }''')

        return floorplan_url, description

    def _extract_foxtons(self, response):
        """Extract floorplan URL and description from Foxtons JSON (no Playwright needed).

        Foxtons detail pages (2026) expose fields directly on
        props.pageProps.propertyDetail (floorplan.src is a full URL, features/
        bulletPoints are lists, shortDescription holds the prose). Older structures
        (propertyBlob, pageData.data.propertyBlob) are kept as fallbacks.
        """
        floorplan_url = None
        description = None

        # Extract __NEXT_DATA__ JSON
        script = response.css('script#__NEXT_DATA__::text').get()
        if not script:
            return None, None

        try:
            data = json.loads(script)
            props = data.get('props', {}).get('pageProps', {})
            property_detail = props.get('propertyDetail', {}) or {}

            # --- NEW PATH (2026): fields directly on propertyDetail ---
            # Floorplan is a dict {"id", "src", "updatedAt"} where src is a full URL
            floorplan = property_detail.get('floorplan', {}) or {}
            if isinstance(floorplan, dict) and floorplan.get('src'):
                floorplan_url = floorplan['src']

            # Description: shortDescription prose + bulletPoints / features list
            description = property_detail.get('shortDescription', '') or property_detail.get('description', '')
            features = (property_detail.get('bulletPoints', [])
                        or property_detail.get('features', [])
                        or property_detail.get('keyFeatures', []) or [])

            # --- FALLBACK: legacy propertyBlob structures ---
            if not floorplan_url and not description and not features:
                prop_blob = property_detail.get('propertyBlob', {}) or {}
                if not prop_blob:
                    page_data = props.get('pageData', {}).get('data', {})
                    prop_blob = page_data.get('propertyBlob', {}) or {}

                description = prop_blob.get('description', '') or prop_blob.get('descriptionShort', '')
                if not description:
                    property_info = prop_blob.get('propertyInfo', {}) or {}
                    description = property_info.get('description', '')
                features = prop_blob.get('keyFeatures', []) or prop_blob.get('features', []) or []

                asset_info = prop_blob.get('assetInfo', {}) or {}
                assets = asset_info.get('assets', {}) or {}
                floorplan_data = assets.get('floorplan', {}) or {}
                if floorplan_data.get('large') and floorplan_data['large'].get('filename'):
                    floorplan_url = f"https://assets.foxtons.co.uk/{floorplan_data['large']['filename']}"
                elif floorplan_data.get('small') and floorplan_data['small'].get('filename'):
                    floorplan_url = f"https://assets.foxtons.co.uk/{floorplan_data['small']['filename']}"

            # Append features to description for richer text
            if features and description:
                description = description + '\n\nFeatures:\n' + '\n'.join(f'- {f}' for f in features)
            elif features and not description:
                description = 'Features:\n' + '\n'.join(f'- {f}' for f in features)

        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return floorplan_url, description

    @staticmethod
    def _clean_floorplan_url(url):
        """Strip JSON-escape artefacts off a floorplan URL captured from raw HTML.

        Live Rightmove serves the page model as escaped JSON inside a <script>, so
        a regex over response.text can pick up trailing backslashes (\\" / \\/) and
        escaped slashes mid-URL. Normalise them so the stored URL is fetchable.
        """
        if not url:
            return url
        # Unescape JSON slashes (\/ -> /, / -> /) then drop any trailing
        # backslash / quote left over from the escape boundary.
        url = url.replace('\\u002F', '/').replace('\\u002f', '/').replace('\\/', '/')
        url = url.rstrip('\\').rstrip('"').rstrip('\\')
        return url

    def _extract_rightmove(self, response):
        """Extract floorplan URL AND description from Rightmove page.

        Rightmove floorplan URLs (2026) live under the /property-floorplan/ path:
            https://media.rightmove.co.uk/property-floorplan/<dir>/<id>/<hash>.jpeg
        Older listings used a '_FLP_' filename token:
            https://media.rightmove.co.uk/XXXk/XXXXXX/ID/XXXXX_FLP_00_0000.jpeg
        Both are matched; '_max_' thumbnails are skipped in favour of full size.

        The page no longer ships <script id="__NEXT_DATA__"> — inline data moved to
        window.__PAGE_MODEL — so the floorplan must be recovered from the raw HTML
        media URLs. The __NEXT_DATA__ branch is kept for back-compat with any page
        that still carries it.

        Returns: (floorplan_url, description)
        """
        floorplan_url = None
        description = None

        # Strategy 1: Search raw HTML for floorplan media URLs directly (most
        # reliable on the current page model, which has no __NEXT_DATA__).
        # Match BOTH the new /property-floorplan/ path and the legacy _FLP_ token.
        media_pattern = (
            r'https://media\.rightmove\.co\.uk/'
            r'(?:[^"\'<>\s\\]*property-floorplan[^"\'<>\s\\]*'
            r'|[^"\'<>\s\\]*_FLP_[^"\'<>\s\\]*)'
        )
        matches = [self._clean_floorplan_url(m) for m in re.findall(media_pattern, response.text, re.I)]
        if matches:
            # Prefer the full-size image (drop '_max_' resized thumbnails) and
            # avoid the resized '/dir/' CDN variant when a clean URL exists.
            def _rank(u):
                return ('_max_' in u, '/dir/' in u)
            best = sorted(matches, key=_rank)[0]
            floorplan_url = best

        # Strategy 2: Try __NEXT_DATA__ JSON (for floorplan and description)
        script = response.css('script#__NEXT_DATA__::text').get()
        if script:
            try:
                data = json.loads(script)
                props = data.get('props', {})
                page_props = props.get('pageProps', {})
                property_data = page_props.get('propertyData', {})

                # Extract description from text object
                text_data = property_data.get('text', {})
                description = text_data.get('description', '')

                # Also get key features and append
                key_features = property_data.get('keyFeatures', [])
                if key_features:
                    description = (description or '') + '\n\nKey Features:\n' + '\n'.join(f'- {f}' for f in key_features)

                # Check images array for floorplan type (if not found above)
                if not floorplan_url:
                    images = property_data.get('images', [])
                    for img in images:
                        url = img.get('url', '') or img.get('srcUrl', '')
                        if '_FLP_' in url or '_flp_' in url.lower():
                            floorplan_url = url
                            break

                # Check floorplans array directly
                if not floorplan_url:
                    floorplans = property_data.get('floorplans', [])
                    if floorplans:
                        for fp in floorplans:
                            url = fp.get('url', '') or fp.get('srcUrl', '')
                            if url:
                                floorplan_url = url
                                break

            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Fallback: Extract description via regex if __NEXT_DATA__ failed
        if not description:
            # Look for description in raw JSON embedded in HTML
            desc_match = re.search(r'"description"\s*:\s*"([^"]{100,})"', response.text)
            if desc_match:
                description = desc_match.group(1)
                # Unescape JSON string
                description = description.replace('\\n', '\n').replace('\\r', '').replace('\\"', '"')

        return floorplan_url, description

    def _extract_sqft_via_ocr(self, floorplan_url):
        """Download floorplan and extract sqft using OCR."""
        if not self.use_ocr or not self.executor:
            return None

        try:
            import requests

            resp = requests.get(floorplan_url, timeout=15)
            if resp.status_code != 200:
                return None

            extractor = FloorplanExtractor()
            result = extractor.extract_from_bytes(resp.content)
            return result.total_sqft if result else None

        except Exception as e:
            self.logger.debug(f"[OCR] Error: {e}")
            return None

    def update_database(self, property_id: str, floorplan_url: str, description: str = None):
        """Update the database with floorplan URL and/or description.

        Issue #33 FIX: Added retry logic with exponential backoff for database locks.
        """
        # Build dynamic update based on what we have
        updates = []
        params = []

        if floorplan_url:
            updates.append('floorplan_url = ?')
            params.append(floorplan_url)

        if description:
            updates.append('description = ?')
            params.append(description)

        if not updates:
            return

        # Rebuild the SET clause with the correct placeholder for the backend.
        ph = '%s' if self.use_postgres else '?'
        set_clause = ', '.join(u.replace('= ?', f'= {ph}') for u in updates)
        params.extend([self.source, property_id])

        # Issue #33 FIX: Retry with exponential backoff (SQLite lock contention).
        max_retries = 5
        for attempt in range(max_retries):
            conn = None
            try:
                conn, _ph, _ = self._get_connection()
                if not self.use_postgres:
                    conn.execute("PRAGMA busy_timeout = 60000")  # 60s timeout
                cursor = conn.cursor()
                cursor.execute(f'''
                    UPDATE listings
                    SET {set_clause}
                    WHERE source = {ph} AND property_id = {ph}
                ''', params)
                conn.commit()
                conn.close()
                return  # Success
            except sqlite3.OperationalError as e:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s, 8s
                    self.logger.warning(f"[DB-RETRY] {property_id}: database locked, retrying in {wait_time}s ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"[DB-ERROR] Failed to update {property_id} after {max_retries} attempts: {e}")
            except Exception as e:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                self.logger.error(f"[DB-ERROR] Failed to update {property_id}: {e}")
                break

    def update_database_sqft(self, property_id: str, sqft: int):
        """Update the database with sqft from OCR.

        Issue #33 FIX: Added retry logic with exponential backoff for database locks.
        """
        ph = '%s' if self.use_postgres else '?'
        max_retries = 5
        for attempt in range(max_retries):
            conn = None
            try:
                conn, _ph, _ = self._get_connection()
                if not self.use_postgres:
                    conn.execute("PRAGMA busy_timeout = 60000")  # 60s timeout
                cursor = conn.cursor()
                cursor.execute(f'''
                    UPDATE listings
                    SET size_sqft = {ph}
                    WHERE source = {ph} AND property_id = {ph}
                    AND (size_sqft IS NULL OR size_sqft = 0)
                ''', (sqft, self.source, property_id))
                conn.commit()
                conn.close()
                return  # Success
            except sqlite3.OperationalError as e:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.warning(f"[DB-RETRY] {property_id} sqft: database locked, retrying in {wait_time}s ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"[DB-ERROR] Failed to update sqft for {property_id} after {max_retries} attempts: {e}")
            except Exception as e:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                self.logger.error(f"[DB-ERROR] Failed to update sqft for {property_id}: {e}")
                break

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
        pct = (self.stats['floorplans_found'] / self.stats['enriched'] * 100) if self.stats['enriched'] else 0

        desc_pct = (self.stats['descriptions_found'] / self.stats['enriched'] * 100) if self.stats['enriched'] else 0

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"ENRICHMENT COMPLETE - {self.source.upper()}")
        self.logger.info("=" * 70)
        self.logger.info(f"[SUMMARY] Duration: {elapsed:.1f}s")
        self.logger.info(f"[SUMMARY] Properties processed: {self.stats['enriched']}")
        self.logger.info(f"[SUMMARY] Floorplans found: {self.stats['floorplans_found']} ({pct:.0f}%)")
        self.logger.info(f"[SUMMARY] Descriptions found: {self.stats['descriptions_found']} ({desc_pct:.0f}%)")
        self.logger.info(f"[SUMMARY] Sqft from OCR: {self.stats['sqft_from_ocr']}")
        self.logger.info(f"[SUMMARY] Failed requests: {self.stats['failed']}")
        self.logger.info("=" * 70)
