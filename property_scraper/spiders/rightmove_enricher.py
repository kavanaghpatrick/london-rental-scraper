"""
Rightmove Enricher Spider - Fetches property detail pages to get square footage

This spider reads existing Rightmove listings from the database and fetches
their detail pages to extract square footage data that's not available in
search results.

Usage:
    scrapy crawl rightmove_enricher
    scrapy crawl rightmove_enricher -a limit=100  # Only enrich first 100
"""

import scrapy
import sqlite3
import json
import re
import time
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Try to import FloorplanExtractor for OCR
try:
    from property_scraper.utils.floorplan_extractor import FloorplanExtractor
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class RightmoveEnricherSpider(scrapy.Spider):
    """Spider to enrich Rightmove listings with square footage data."""

    name = 'rightmove_enricher'
    allowed_domains = ['rightmove.co.uk']

    def __init__(self, limit=None, use_ocr=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.limit = int(limit) if limit else None
        self.db_path = 'output/rentals.db'
        self.use_ocr = str(use_ocr).lower() in ('true', '1', 'yes') and OCR_AVAILABLE

        # ThreadPoolExecutor for OCR (non-blocking)
        self.executor = ThreadPoolExecutor(max_workers=2) if self.use_ocr else None

        # Stats
        self.stats = {
            'total_to_enrich': 0,
            'enriched': 0,
            'sqft_found': 0,
            'sqft_from_ocr': 0,
            'floorplans_found': 0,
            'failed': 0,
            'start_time': time.time(),
        }

        self.logger.info("=" * 70)
        self.logger.info("RIGHTMOVE ENRICHER INITIALIZED")
        self.logger.info("=" * 70)
        if self.limit:
            self.logger.info(f"[CONFIG] Limit: {self.limit} properties")
        self.logger.info(f"[CONFIG] OCR available: {OCR_AVAILABLE}")
        self.logger.info(f"[CONFIG] Use OCR: {self.use_ocr}")

    def start_requests(self):
        """Read properties from database that need enrichment."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Find Rightmove listings without square footage
        query = '''
            SELECT property_id, url, address, price_pcm, bedrooms
            FROM listings
            WHERE source = 'rightmove'
            AND (size_sqft IS NULL OR size_sqft = 0)
        '''
        if self.limit:
            query += f' LIMIT {self.limit}'

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        self.stats['total_to_enrich'] = len(rows)
        self.logger.info(f"[START] Found {len(rows)} Rightmove listings needing square footage")

        for i, row in enumerate(rows):
            if i % 50 == 0:
                self.logger.info(f"[QUEUE] Queuing {i+1}/{len(rows)}...")

            yield scrapy.Request(
                row['url'],
                callback=self.parse_detail,
                meta={
                    'property_id': row['property_id'],
                    'address': row['address'],
                    'price_pcm': row['price_pcm'],
                    'bedrooms': row['bedrooms'],
                    'request_start': time.time()
                },
                dont_filter=True,
                errback=self.handle_error
            )

    def handle_error(self, failure):
        """Handle request failures."""
        prop_id = failure.request.meta.get('property_id', 'unknown')
        self.stats['failed'] += 1
        self.logger.error(f"[ERROR] Failed to fetch {prop_id}: {failure.value}")

    def parse_detail(self, response):
        """Parse property detail page for square footage and floorplan."""
        prop_id = response.meta['property_id']
        request_time = time.time() - response.meta.get('request_start', time.time())

        if response.status != 200:
            self.stats['failed'] += 1
            self.logger.warning(f"[HTTP-ERROR] {prop_id}: status {response.status}")
            return

        sqft = None
        floorplan_url = None
        property_data = {}

        # Extract window.PAGE_MODEL JSON (Rightmove's current format)
        page_model_match = re.search(r'window\.PAGE_MODEL\s*=\s*', response.text)
        if page_model_match:
            try:
                start = page_model_match.end()
                # Find matching closing brace
                brace_count = 0
                i = start
                text = response.text
                while i < len(text):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            break
                    i += 1

                json_str = text[start:i+1]
                data = json.loads(json_str)
                property_data = data.get('propertyData', {})

                # Extract sqft from sizings
                sizings = property_data.get('sizings', [])
                for sizing in sizings:
                    if sizing.get('unit') == 'sqft':
                        sqft = sizing.get('minimumSize') or sizing.get('maximumSize')
                        if sqft:
                            sqft = int(sqft)
                        break

                # Extract floorplan URL
                floorplans = property_data.get('floorplans', [])
                if floorplans:
                    floorplan_url = floorplans[0].get('url')
                    if floorplan_url:
                        self.stats['floorplans_found'] += 1

                # Try text content for sqft
                if not sqft:
                    text_data = property_data.get('text', {})
                    description = text_data.get('description', '') + ' ' + text_data.get('propertyPhrase', '')
                    sqft_match = re.search(r'(\d{3,5})\s*(?:sq\.?\s*ft|square\s*feet)', description, re.I)
                    if sqft_match:
                        sqft = int(sqft_match.group(1))

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.logger.debug(f"[PARSE] {prop_id}: PAGE_MODEL parse issue - {e}")

        # Try fallback: search in page text
        if not sqft:
            page_text = response.text
            sqft_patterns = [
                r'(\d{3,5})\s*sq\.?\s*ft',
                r'(\d{3,5})\s*square\s*feet',
                r'(\d{3,5})\s*sqft',
            ]
            for pattern in sqft_patterns:
                match = re.search(pattern, page_text, re.I)
                if match:
                    sqft = int(match.group(1))
                    break

        # If no sqft but have floorplan and OCR is enabled, try OCR
        if not sqft and floorplan_url and self.use_ocr:
            ocr_sqft = self._extract_sqft_via_ocr(floorplan_url)
            if ocr_sqft:
                sqft = ocr_sqft
                self.stats['sqft_from_ocr'] += 1
                self.logger.debug(f"[OCR] {prop_id}: extracted {sqft} sqft from floorplan")

        self.stats['enriched'] += 1

        # Update database if we found anything
        if (sqft and 100 < sqft < 50000) or floorplan_url:
            if sqft and 100 < sqft < 50000:
                self.stats['sqft_found'] += 1
            self.update_database(prop_id, sqft if (sqft and 100 < sqft < 50000) else None, floorplan_url)
            self.logger.debug(
                f"[FOUND] {prop_id}: sqft={sqft}, floorplan={bool(floorplan_url)} | "
                f"{response.meta['bedrooms']}bed @ £{response.meta['price_pcm']:,}"
            )
        else:
            self.logger.debug(f"[MISS] {prop_id}: no sqft or floorplan found")

        # Progress logging
        if self.stats['enriched'] % 25 == 0:
            pct = (self.stats['sqft_found'] / self.stats['enriched'] * 100) if self.stats['enriched'] else 0
            self.logger.info(
                f"[PROGRESS] {self.stats['enriched']}/{self.stats['total_to_enrich']} | "
                f"Found sqft: {self.stats['sqft_found']} ({pct:.0f}%) | "
                f"Floorplans: {self.stats['floorplans_found']}"
            )

    def _extract_sqft_via_ocr(self, floorplan_url):
        """Download floorplan and extract sqft using OCR.

        Uses ThreadPoolExecutor to avoid blocking the Scrapy reactor.
        """
        if not self.use_ocr or not self.executor:
            return None

        try:
            import requests

            def download_and_extract(url):
                """Blocking function to run in executor thread."""
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    return None
                extractor = FloorplanExtractor()
                result = extractor.extract_from_bytes(resp.content)
                return result.total_sqft if result else None

            # Submit to executor and wait for result (non-blocking to reactor)
            future = self.executor.submit(download_and_extract, floorplan_url)
            return future.result(timeout=25)

        except Exception as e:
            self.logger.debug(f"[OCR] Error extracting sqft: {e}")
            return None

    def update_database(self, property_id: str, sqft: int = None, floorplan_url: str = None):
        """Update the database with the enriched data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build dynamic update query
            updates = []
            values = []

            if sqft is not None:
                updates.append('size_sqft = ?')
                values.append(sqft)

            if floorplan_url is not None:
                updates.append('floorplan_url = ?')
                values.append(floorplan_url)

            if updates:
                values.append(property_id)
                query = f'''
                    UPDATE listings
                    SET {', '.join(updates)}
                    WHERE source = 'rightmove' AND property_id = ?
                '''
                cursor.execute(query, values)
                conn.commit()

            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"[DB-ERROR] Failed to update {property_id}: {e}")

    def closed(self, reason):
        """Log summary when spider closes."""
        # Cleanup executor
        if self.executor:
            self.executor.shutdown(wait=False)

        elapsed = time.time() - self.stats['start_time']
        sqft_pct = (self.stats['sqft_found'] / self.stats['enriched'] * 100) if self.stats['enriched'] else 0
        floorplan_pct = (self.stats['floorplans_found'] / self.stats['enriched'] * 100) if self.stats['enriched'] else 0

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("RIGHTMOVE ENRICHMENT COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"[SUMMARY] Duration: {elapsed:.1f}s")
        self.logger.info(f"[SUMMARY] Properties processed: {self.stats['enriched']}")
        self.logger.info(f"[SUMMARY] Square footage found: {self.stats['sqft_found']} ({sqft_pct:.0f}%)")
        self.logger.info(f"[SUMMARY] Sqft from OCR: {self.stats['sqft_from_ocr']}")
        self.logger.info(f"[SUMMARY] Floorplans found: {self.stats['floorplans_found']} ({floorplan_pct:.0f}%)")
        self.logger.info(f"[SUMMARY] Failed requests: {self.stats['failed']}")
        self.logger.info("=" * 70)
