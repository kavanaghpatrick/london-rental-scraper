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


class FloorplanEnricherSpider(scrapy.Spider):
    """Spider to add floorplan URLs to existing listings."""

    name = 'floorplan_enricher'

    # Source-specific settings
    SOURCE_CONFIG = {
        'knightfrank': {
            'domain': 'knightfrank.co.uk',
            'needs_playwright': True,
            'url_template': None,  # Use URL from DB
        },
        'chestertons': {
            'domain': 'chestertons.com',
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
            'url_template': 'https://www.foxtons.co.uk/properties/{property_id}',
        },
    }

    def __init__(self, source=None, limit=None, use_ocr=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not source or source not in self.SOURCE_CONFIG:
            raise ValueError(f"Must specify source: {list(self.SOURCE_CONFIG.keys())}")

        self.source = source
        self.source_config = self.SOURCE_CONFIG[source]
        self.allowed_domains = [self.source_config['domain']]
        self.limit = int(limit) if limit else None
        self.db_path = 'output/rentals.db'
        self.use_ocr = str(use_ocr).lower() in ('true', '1', 'yes') and OCR_AVAILABLE

        # ThreadPoolExecutor for OCR
        self.executor = ThreadPoolExecutor(max_workers=2) if self.use_ocr else None

        # Stats
        self.stats = {
            'total_to_enrich': 0,
            'enriched': 0,
            'floorplans_found': 0,
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

    def start_requests(self):
        """Read properties from database that need floorplan URLs."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Find listings without floorplan URL that have a valid detail URL
        query = '''
            SELECT property_id, url, address, price_pcm, bedrooms, size_sqft
            FROM listings
            WHERE source = ?
            AND (floorplan_url IS NULL OR floorplan_url = '')
            AND url IS NOT NULL AND LENGTH(url) > 10
        '''
        params = [self.source]

        if self.limit:
            query += f' LIMIT {self.limit}'

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        self.stats['total_to_enrich'] = len(rows)
        self.logger.info(f"[START] Found {len(rows)} {self.source} listings needing floorplan URLs")

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

        try:
            # Source-specific extraction
            if self.source == 'knightfrank':
                floorplan_url = await self._extract_knightfrank(playwright_page, response)
            elif self.source == 'chestertons':
                floorplan_url = await self._extract_chestertons(playwright_page, response)
            elif self.source == 'savills':
                floorplan_url = await self._extract_savills(playwright_page, response)
            elif self.source == 'foxtons':
                floorplan_url = self._extract_foxtons(response)

        except Exception as e:
            self.logger.debug(f"[PARSE] {prop_id}: extraction error - {e}")

        finally:
            if playwright_page:
                await playwright_page.close()

        self.stats['enriched'] += 1

        if floorplan_url:
            self.stats['floorplans_found'] += 1
            self.update_database(prop_id, floorplan_url)
            self.logger.debug(f"[FOUND] {prop_id}: {floorplan_url[:60]}...")

            # OCR if no existing sqft
            if self.use_ocr and not response.meta.get('existing_sqft'):
                ocr_sqft = self._extract_sqft_via_ocr(floorplan_url)
                if ocr_sqft:
                    self.stats['sqft_from_ocr'] += 1
                    self.update_database_sqft(prop_id, ocr_sqft)
                    self.logger.debug(f"[OCR] {prop_id}: extracted {ocr_sqft} sqft")
        else:
            self.logger.debug(f"[MISS] {prop_id}: no floorplan found")

        # Progress logging
        if self.stats['enriched'] % 25 == 0:
            pct = (self.stats['floorplans_found'] / self.stats['enriched'] * 100) if self.stats['enriched'] else 0
            self.logger.info(
                f"[PROGRESS] {self.stats['enriched']}/{self.stats['total_to_enrich']} | "
                f"Floorplans: {self.stats['floorplans_found']} ({pct:.0f}%)"
            )

    async def _extract_knightfrank(self, page, response):
        """Extract floorplan URL from Knight Frank detail page."""
        if not page:
            return None

        await page.wait_for_timeout(2000)

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

        return floorplan_url

    async def _extract_chestertons(self, page, response):
        """Extract floorplan URL from Chestertons detail page."""
        if not page:
            return None

        await page.wait_for_timeout(2000)

        # Click floor plans tab
        clicked = await page.evaluate('''() => {
            const tabs = document.querySelectorAll('button, [role="tab"], a, .tab');
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
                if (src.includes('homeflow-assets.co.uk') &&
                    src.includes('floorplan')) {
                    return src;
                }
            }
            return null;
        }''')

        return floorplan_url

    async def _extract_savills(self, page, response):
        """Extract floorplan URL from Savills detail page.

        Savills floorplans are GIF files that appear after clicking the PLANS button.
        The floorplan can be identified by:
        1. Alt text contains "Floorplan"
        2. File extension is .GIF (photos are .jpg)
        3. Image is from assets.savills.com/properties/
        """
        if not page:
            return None

        await page.wait_for_timeout(2000)

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

        return floorplan_url

    def _extract_foxtons(self, response):
        """Extract floorplan URL from Foxtons JSON (no Playwright needed)."""
        # Extract __NEXT_DATA__ JSON
        script = response.css('script#__NEXT_DATA__::text').get()
        if not script:
            return None

        try:
            data = json.loads(script)
            props = data.get('props', {}).get('pageProps', {})
            page_data = props.get('pageData', {}).get('data', {})

            # Get property blob
            prop_blob = page_data.get('propertyBlob', {}) or {}
            asset_info = prop_blob.get('assetInfo', {}) or {}
            assets = asset_info.get('assets', {}) or {}
            floorplan_data = assets.get('floorplan', {}) or {}

            # Get large floorplan URL
            if floorplan_data.get('large') and floorplan_data['large'].get('filename'):
                filename = floorplan_data['large']['filename']
                return f"https://assets.foxtons.co.uk/{filename}"

            if floorplan_data.get('small') and floorplan_data['small'].get('filename'):
                filename = floorplan_data['small']['filename']
                return f"https://assets.foxtons.co.uk/{filename}"

        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return None

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

    def update_database(self, property_id: str, floorplan_url: str):
        """Update the database with floorplan URL."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE listings
                SET floorplan_url = ?
                WHERE source = ? AND property_id = ?
            ''', (floorplan_url, self.source, property_id))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"[DB-ERROR] Failed to update {property_id}: {e}")

    def update_database_sqft(self, property_id: str, sqft: int):
        """Update the database with sqft from OCR."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE listings
                SET size_sqft = ?
                WHERE source = ? AND property_id = ?
                AND (size_sqft IS NULL OR size_sqft = 0)
            ''', (sqft, self.source, property_id))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"[DB-ERROR] Failed to update sqft for {property_id}: {e}")

    def closed(self, reason):
        """Log summary when spider closes."""
        if self.executor:
            self.executor.shutdown(wait=False)

        elapsed = time.time() - self.stats['start_time']
        pct = (self.stats['floorplans_found'] / self.stats['enriched'] * 100) if self.stats['enriched'] else 0

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"FLOORPLAN ENRICHMENT COMPLETE - {self.source.upper()}")
        self.logger.info("=" * 70)
        self.logger.info(f"[SUMMARY] Duration: {elapsed:.1f}s")
        self.logger.info(f"[SUMMARY] Properties processed: {self.stats['enriched']}")
        self.logger.info(f"[SUMMARY] Floorplans found: {self.stats['floorplans_found']} ({pct:.0f}%)")
        self.logger.info(f"[SUMMARY] Sqft from OCR: {self.stats['sqft_from_ocr']}")
        self.logger.info(f"[SUMMARY] Failed requests: {self.stats['failed']}")
        self.logger.info("=" * 70)
