"""
Features Enricher Spider - Extracts amenities from property detail pages

Extracts:
- Balcony/terrace
- Air conditioning
- High ceilings / floor-to-ceiling windows
- Floor level
- Porter/concierge
- Parking
- Garden
- Gym/pool access
- Furnished status

Usage:
    scrapy crawl features_enricher
    scrapy crawl features_enricher -a source=knightfrank -a limit=50
"""

import scrapy
import sqlite3
import json
import re
import time
from datetime import datetime


class FeaturesEnricherSpider(scrapy.Spider):
    """Spider to enrich listings with amenity features from detail pages."""

    name = 'features_enricher'

    # Custom settings for this spider
    custom_settings = {
        'CONCURRENT_REQUESTS': 4,
        'DOWNLOAD_DELAY': 2,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'RETRY_TIMES': 3,
        'COOKIES_ENABLED': False,
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-GB,en;q=0.9',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        },
    }

    def __init__(self, source=None, limit=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.source_filter = source  # Optional: 'rightmove', 'foxtons', 'knightfrank', 'chestertons'
        self.limit = int(limit) if limit else None
        self.db_path = 'output/rentals.db'

        self.stats = {
            'total': 0,
            'processed': 0,
            'features_found': 0,
            'failed': 0,
            'start_time': time.time(),
        }

        self.logger.info("=" * 70)
        self.logger.info("FEATURES ENRICHER INITIALIZED")
        self.logger.info("=" * 70)
        if self.source_filter:
            self.logger.info(f"[CONFIG] Source filter: {self.source_filter}")
        if self.limit:
            self.logger.info(f"[CONFIG] Limit: {self.limit}")

    def start_requests(self):
        """Read properties from database that need feature enrichment."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Find listings without features extracted (features is null or empty '[]')
        query = '''
            SELECT id, source, property_id, url, address, price_pcm, bedrooms
            FROM listings
            WHERE (features IS NULL OR features = '[]' OR features = '')
            AND size_sqft > 0
        '''
        params = []

        if self.source_filter:
            query += " AND source = ?"
            params.append(self.source_filter)

        if self.limit:
            query += f' LIMIT {self.limit}'

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        self.stats['total'] = len(rows)
        self.logger.info(f"[START] Found {len(rows)} listings needing feature enrichment")

        for i, row in enumerate(rows):
            if i % 50 == 0 and i > 0:
                self.logger.info(f"[QUEUE] Queuing {i}/{len(rows)}...")

            source = row['source']

            # Use playwright for JavaScript-heavy sites
            use_playwright = source in ['foxtons', 'chestertons', 'knightfrank']

            meta = {
                'id': row['id'],
                'source': source,
                'property_id': row['property_id'],
                'address': row['address'],
                'price_pcm': row['price_pcm'],
                'bedrooms': row['bedrooms'],
                'request_start': time.time(),
            }

            if use_playwright:
                meta['playwright'] = True
                meta['playwright_include_page'] = False

            yield scrapy.Request(
                row['url'],
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

    def parse_detail(self, response):
        """Parse property detail page for features."""
        meta = response.meta
        source = meta['source']
        prop_id = meta['property_id']

        if response.status != 200:
            self.stats['failed'] += 1
            self.logger.warning(f"[HTTP-ERROR] {prop_id}: status {response.status}")
            return

        # Extract features based on source
        if source == 'rightmove':
            features = self.parse_rightmove(response)
        elif source == 'foxtons':
            features = self.parse_foxtons(response)
        elif source == 'knightfrank':
            features = self.parse_knightfrank(response)
        elif source == 'chestertons':
            features = self.parse_chestertons(response)
        else:
            features = self.parse_generic(response)

        self.stats['processed'] += 1

        if features and any(v for k, v in features.items() if k != 'raw_text'):
            self.stats['features_found'] += 1
            self.update_database(meta['id'], features)
            self.logger.info(
                f"[FOUND] {source}/{prop_id}: {self.summarize_features(features)}"
            )
        else:
            self.logger.debug(f"[MISS] {source}/{prop_id}: no features found")

        # Progress logging
        if self.stats['processed'] % 25 == 0:
            pct = (self.stats['features_found'] / self.stats['processed'] * 100) if self.stats['processed'] else 0
            self.logger.info(
                f"[PROGRESS] {self.stats['processed']}/{self.stats['total']} | "
                f"Features found: {self.stats['features_found']} ({pct:.0f}%)"
            )

    def parse_rightmove(self, response):
        """Parse Rightmove property page."""
        features = {}

        # Extract from __NEXT_DATA__
        script = response.css('script#__NEXT_DATA__::text').get()
        if script:
            try:
                data = json.loads(script)
                props = data.get('props', {}).get('pageProps', {})
                property_data = props.get('propertyData', {})

                # Key features list
                key_features = property_data.get('keyFeatures', [])
                features['key_features'] = key_features

                # Full description
                text_data = property_data.get('text', {})
                description = text_data.get('description', '')
                features['raw_text'] = description

                # Location/address info
                location = property_data.get('location', {})
                # Sometimes has floor level

                # Parse amenities from key features and description
                all_text = ' '.join(key_features) + ' ' + description
                features.update(self.extract_amenities(all_text.lower()))

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.logger.debug(f"[PARSE] Rightmove JSON error: {e}")

        # Fallback to page text
        if not features.get('raw_text'):
            page_text = response.css('div.property-description ::text').getall()
            full_text = ' '.join(page_text)
            features['raw_text'] = full_text
            features.update(self.extract_amenities(full_text.lower()))

        return features

    def parse_foxtons(self, response):
        """Parse Foxtons property page."""
        features = {}

        # Get description
        description = response.css('div.property-description ::text').getall()
        full_text = ' '.join(description)
        features['raw_text'] = full_text

        # Get key features
        key_features = response.css('ul.key-features li::text').getall()
        features['key_features'] = [f.strip() for f in key_features if f.strip()]

        # Get amenities section
        amenities = response.css('div.amenities li::text, div.features li::text').getall()
        features['amenities'] = [a.strip() for a in amenities if a.strip()]

        # Extract structured amenities
        all_text = full_text + ' ' + ' '.join(features.get('key_features', [])) + ' ' + ' '.join(features.get('amenities', []))
        features.update(self.extract_amenities(all_text.lower()))

        return features

    def parse_knightfrank(self, response):
        """Parse Knight Frank property page."""
        features = {}

        # Description
        description = response.css('div.property-description ::text, div.summary ::text').getall()
        full_text = ' '.join(description)
        features['raw_text'] = full_text

        # Key features/highlights
        key_features = response.css('ul.highlights li::text, ul.key-features li::text').getall()
        features['key_features'] = [f.strip() for f in key_features if f.strip()]

        # Look for structured data
        specs = response.css('div.specifications dl dt::text, div.specifications dl dd::text').getall()
        features['specifications'] = specs

        all_text = full_text + ' ' + ' '.join(features.get('key_features', []))
        features.update(self.extract_amenities(all_text.lower()))

        return features

    def parse_chestertons(self, response):
        """Parse Chestertons property page."""
        features = {}

        # Description
        description = response.css('div.property-description ::text, div.description ::text').getall()
        full_text = ' '.join(description)
        features['raw_text'] = full_text

        # Key features
        key_features = response.css('ul.features li::text, div.key-features li::text').getall()
        features['key_features'] = [f.strip() for f in key_features if f.strip()]

        all_text = full_text + ' ' + ' '.join(features.get('key_features', []))
        features.update(self.extract_amenities(all_text.lower()))

        return features

    def parse_generic(self, response):
        """Generic parser for unknown sources."""
        features = {}
        page_text = response.css('body ::text').getall()
        full_text = ' '.join(page_text)
        features['raw_text'] = full_text[:5000]  # Limit
        features.update(self.extract_amenities(full_text.lower()))
        return features

    def extract_amenities(self, text):
        """Extract structured amenities from text."""
        amenities = {}

        # Balcony/outdoor
        amenities['has_balcony'] = bool(re.search(r'\bbalcon[y|ies]\b', text))
        amenities['has_terrace'] = bool(re.search(r'\bterrace\b', text))
        amenities['has_roof_terrace'] = bool(re.search(r'\broof\s*terrace\b', text))
        amenities['has_garden'] = bool(re.search(r'\bgarden\b', text))
        amenities['has_patio'] = bool(re.search(r'\bpatio\b', text))

        # Climate control
        amenities['has_air_conditioning'] = bool(re.search(r'air[\s-]*condition|a/c\b|aircon|\bac\b', text))
        amenities['has_underfloor_heating'] = bool(re.search(r'underfloor\s*heat', text))

        # Ceilings/windows
        amenities['has_high_ceilings'] = bool(re.search(r'high\s*ceiling|tall\s*ceiling|double[\s-]*height|lofty\s*ceiling', text))
        amenities['has_floor_to_ceiling_windows'] = bool(re.search(r'floor[\s-]*to[\s-]*ceiling|full[\s-]*height\s*window', text))

        # Building features
        amenities['has_lift'] = bool(re.search(r'\blift\b|\belevator\b', text))
        amenities['has_porter'] = bool(re.search(r'\bporter|concierge|24[\s-]*hour|24hr\s*security', text))
        amenities['has_gym'] = bool(re.search(r'\bgym\b|fitness\s*(centre|center|room)', text))
        amenities['has_pool'] = bool(re.search(r'\bpool\b|swimming', text))

        # Parking
        amenities['has_parking'] = bool(re.search(r'\bparking\b|\bgarage\b|car\s*space', text))

        # Floor level
        floor = None
        match = re.search(r'(\d+)(?:st|nd|rd|th)\s*floor', text)
        if match:
            floor = int(match.group(1))
        elif 'ground floor' in text:
            floor = 0
        elif re.search(r'basement|lower\s*ground', text):
            floor = -1
        elif re.search(r'penthouse|top\s*floor', text):
            floor = 99
        amenities['floor_level'] = floor

        # Furnished status
        if re.search(r'\bunfurnished\b', text):
            amenities['furnished'] = 'unfurnished'
        elif re.search(r'\bfurnished\b', text):
            amenities['furnished'] = 'furnished'
        elif re.search(r'\bpart[\s-]*furnished\b', text):
            amenities['furnished'] = 'part_furnished'
        else:
            amenities['furnished'] = None

        return amenities

    def summarize_features(self, features):
        """Create a brief summary of found features."""
        found = []
        for key, value in features.items():
            if key.startswith('has_') and value:
                found.append(key.replace('has_', ''))
            elif key == 'floor_level' and value is not None:
                found.append(f"floor_{value}")
            elif key == 'furnished' and value:
                found.append(value)
        return ', '.join(found[:5]) if found else 'none'

    def update_database(self, record_id: int, features: dict):
        """Update the database with extracted features."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Store features as JSON
            features_json = json.dumps({k: v for k, v in features.items() if k != 'raw_text'})

            # Also update individual columns if they exist
            cursor.execute('''
                UPDATE listings
                SET features = ?,
                    description = COALESCE(description, ?)
                WHERE id = ?
            ''', (features_json, features.get('raw_text', '')[:5000], record_id))

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"[DB-ERROR] Failed to update record {record_id}: {e}")

    def closed(self, reason):
        """Log summary when spider closes."""
        elapsed = time.time() - self.stats['start_time']
        pct = (self.stats['features_found'] / self.stats['processed'] * 100) if self.stats['processed'] else 0

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("FEATURES ENRICHMENT COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"[SUMMARY] Duration: {elapsed:.1f}s")
        self.logger.info(f"[SUMMARY] Properties processed: {self.stats['processed']}")
        self.logger.info(f"[SUMMARY] Features found: {self.stats['features_found']} ({pct:.0f}%)")
        self.logger.info(f"[SUMMARY] Failed requests: {self.stats['failed']}")
        self.logger.info("=" * 70)
