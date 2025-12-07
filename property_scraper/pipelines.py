"""
Scrapy Pipelines for property data processing.

Pipelines:
1. CleanDataPipeline - Normalize and validate data
2. DuplicateFilterPipeline - Filter duplicate listings
3. JsonWriterPipeline - Write to JSON files
4. SQLitePipeline - Persist to SQLite database
"""

import os
import json
import sqlite3
import logging
import time
from datetime import datetime
from itemadapter import ItemAdapter

logger = logging.getLogger(__name__)


class CleanDataPipeline:
    """Clean and normalize property data."""

    def __init__(self):
        self.items_processed = 0
        self.fixes_applied = 0

    def open_spider(self, spider):
        logger.info("[PIPELINE:Clean] Initialized - will normalize prices and clean text")

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        fixes = []

        # Normalize price to PCM if needed
        if adapter.get('price_pw') and not adapter.get('price_pcm'):
            adapter['price_pcm'] = int(adapter['price_pw'] * 52 / 12)
            fixes.append('calculated_pcm')

        if adapter.get('price_pcm') and not adapter.get('price_pw'):
            adapter['price_pw'] = int(adapter['price_pcm'] * 12 / 52)
            fixes.append('calculated_pw')

        # Clean address
        if adapter.get('address'):
            original = adapter['address']
            adapter['address'] = ' '.join(adapter['address'].split())
            if original != adapter['address']:
                fixes.append('cleaned_address')

        # Normalize property type
        prop_type = adapter.get('property_type', '')
        if prop_type:
            adapter['property_type'] = prop_type.lower().strip()

        # Ensure bedrooms/bathrooms are integers
        for field in ['bedrooms', 'bathrooms']:
            val = adapter.get(field)
            if val is not None:
                try:
                    adapter[field] = int(val)
                except (ValueError, TypeError):
                    adapter[field] = None
                    fixes.append(f'nulled_{field}')

        # Ensure scraped_at is set
        if not adapter.get('scraped_at'):
            adapter['scraped_at'] = datetime.utcnow().isoformat()
            fixes.append('added_timestamp')

        self.items_processed += 1
        if fixes:
            self.fixes_applied += len(fixes)
            logger.debug(f"[PIPELINE:Clean] {adapter.get('property_id')}: {', '.join(fixes)}")

        return item

    def close_spider(self, spider):
        logger.info(
            f"[PIPELINE:Clean] Complete - {self.items_processed} items, "
            f"{self.fixes_applied} fixes applied"
        )


class DuplicateFilterPipeline:
    """Filter duplicate listings by source + property_id."""

    def __init__(self):
        self.seen = set()
        self.duplicates_filtered = 0

    def open_spider(self, spider):
        logger.info("[PIPELINE:Dedupe] Initialized - tracking source:property_id pairs")

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        source = adapter.get('source', '')
        prop_id = adapter.get('property_id', '')

        key = f"{source}:{prop_id}"

        if key in self.seen:
            self.duplicates_filtered += 1
            logger.debug(f"[PIPELINE:Dedupe] Filtered duplicate: {key}")
            from scrapy.exceptions import DropItem
            raise DropItem(f"Duplicate item: {key}")

        self.seen.add(key)
        return item

    def close_spider(self, spider):
        logger.info(
            f"[PIPELINE:Dedupe] Complete - {len(self.seen)} unique, "
            f"{self.duplicates_filtered} duplicates filtered"
        )


class JsonWriterPipeline:
    """Write items to JSONL files immediately (no memory accumulation)."""

    def __init__(self):
        self.output_dir = None
        self.counts = {}
        self.bytes_written = 0
        self.start_time = None

    def open_spider(self, spider):
        self.output_dir = spider.settings.get('OUTPUT_DIR', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.start_time = time.time()
        logger.info(f"[PIPELINE:JSON] Initialized - writing to {self.output_dir}/")

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        area = adapter.get('area', 'unknown')

        # Write immediately to JSONL (one JSON object per line)
        filepath = os.path.join(self.output_dir, f"{area.lower()}_listings.jsonl")
        try:
            line = json.dumps(dict(adapter)) + '\n'
            with open(filepath, 'a') as f:
                f.write(line)
            self.counts[area] = self.counts.get(area, 0) + 1
            self.bytes_written += len(line)
        except IOError as e:
            logger.error(f"[PIPELINE:JSON] Write failed for {filepath}: {e}")

        return item

    def close_spider(self, spider):
        elapsed = time.time() - self.start_time if self.start_time else 0
        total_items = sum(self.counts.values())

        logger.info("[PIPELINE:JSON] Complete:")
        for area, count in sorted(self.counts.items()):
            logger.info(f"  {area.lower()}_listings.jsonl: {count} items")
        logger.info(
            f"[PIPELINE:JSON] Total: {total_items} items, "
            f"{self.bytes_written/1024:.1f}KB written in {elapsed:.1f}s"
        )


class SQLitePipeline:
    """Persist items to SQLite database with batched commits."""

    BATCH_SIZE = 100  # Commit every N items

    def __init__(self):
        self.conn = None
        self.cursor = None
        self.pending_count = 0
        self.total_inserted = 0
        self.total_errors = 0
        self.batch_count = 0
        self.start_time = None
        self.db_path = None

    def open_spider(self, spider):
        output_dir = spider.settings.get('OUTPUT_DIR', 'output')
        os.makedirs(output_dir, exist_ok=True)

        self.db_path = os.path.join(output_dir, 'rentals.db')
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
        self.start_time = time.time()

        # Log existing count
        self.cursor.execute('SELECT COUNT(*) FROM listings')
        existing = self.cursor.fetchone()[0]
        logger.info(f"[PIPELINE:SQLite] Initialized - {self.db_path}")
        logger.info(f"[PIPELINE:SQLite] Existing records: {existing}")
        logger.info(f"[PIPELINE:SQLite] Batch size: {self.BATCH_SIZE}")

    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                property_id TEXT NOT NULL,
                url TEXT,
                area TEXT,

                price INTEGER,
                price_pw INTEGER,
                price_pcm INTEGER,
                price_period TEXT,

                address TEXT,
                postcode TEXT,
                latitude REAL,
                longitude REAL,

                bedrooms INTEGER,
                bathrooms INTEGER,
                reception_rooms INTEGER,
                property_type TEXT,
                size_sqft INTEGER,
                size_sqm INTEGER,
                furnished TEXT,
                epc_rating TEXT,

                floorplan_url TEXT,
                room_details TEXT,

                let_agreed INTEGER DEFAULT 0,

                agent_name TEXT,
                agent_phone TEXT,

                summary TEXT,
                description TEXT,
                features TEXT,

                added_date TEXT,
                scraped_at TEXT,

                UNIQUE(source, property_id)
            )
        ''')

        # Add new columns to existing table if they don't exist
        new_columns = [
            ('reception_rooms', 'INTEGER'),
            ('size_sqm', 'INTEGER'),
            ('epc_rating', 'TEXT'),
            ('floorplan_url', 'TEXT'),
            ('room_details', 'TEXT'),
            # Binary floor columns for ML model training
            ('has_basement', 'INTEGER'),
            ('has_lower_ground', 'INTEGER'),
            ('has_ground', 'INTEGER'),
            ('has_mezzanine', 'INTEGER'),
            ('has_first_floor', 'INTEGER'),
            ('has_second_floor', 'INTEGER'),
            ('has_third_floor', 'INTEGER'),
            ('has_fourth_plus', 'INTEGER'),
            ('has_roof_terrace', 'INTEGER'),
            ('floor_count', 'INTEGER'),
            ('property_levels', 'TEXT'),  # single_floor, duplex, triplex, multi_floor
        ]
        for col_name, col_type in new_columns:
            try:
                self.cursor.execute(f'ALTER TABLE listings ADD COLUMN {col_name} {col_type}')
                logger.info(f"[PIPELINE:SQLite] Added new column: {col_name}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_area ON listings(area)
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_price ON listings(price_pcm)
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_bedrooms ON listings(bedrooms)
        ''')

        self.conn.commit()
        logger.debug("[PIPELINE:SQLite] Tables and indexes created/verified")

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        # Convert features list to JSON string
        features = adapter.get('features', [])
        if isinstance(features, list):
            features = json.dumps(features)

        # Convert room_details to JSON string if it's a dict/list
        room_details = adapter.get('room_details')
        if isinstance(room_details, (dict, list)):
            room_details = json.dumps(room_details)

        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO listings (
                    source, property_id, url, area,
                    price, price_pw, price_pcm, price_period,
                    address, postcode, latitude, longitude,
                    bedrooms, bathrooms, reception_rooms, property_type,
                    size_sqft, size_sqm, furnished, epc_rating,
                    floorplan_url, room_details,
                    has_basement, has_lower_ground, has_ground, has_mezzanine,
                    has_first_floor, has_second_floor, has_third_floor,
                    has_fourth_plus, has_roof_terrace, floor_count, property_levels,
                    let_agreed,
                    agent_name, agent_phone,
                    summary, description, features,
                    added_date, scraped_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                adapter.get('source'),
                adapter.get('property_id'),
                adapter.get('url'),
                adapter.get('area'),
                adapter.get('price'),
                adapter.get('price_pw'),
                adapter.get('price_pcm'),
                adapter.get('price_period'),
                adapter.get('address'),
                adapter.get('postcode'),
                adapter.get('latitude'),
                adapter.get('longitude'),
                adapter.get('bedrooms'),
                adapter.get('bathrooms'),
                adapter.get('reception_rooms'),
                adapter.get('property_type'),
                adapter.get('size_sqft'),
                adapter.get('size_sqm'),
                adapter.get('furnished'),
                adapter.get('epc_rating'),
                adapter.get('floorplan_url'),
                room_details,
                adapter.get('has_basement'),
                adapter.get('has_lower_ground'),
                adapter.get('has_ground'),
                adapter.get('has_mezzanine'),
                adapter.get('has_first_floor'),
                adapter.get('has_second_floor'),
                adapter.get('has_third_floor'),
                adapter.get('has_fourth_plus'),
                adapter.get('has_roof_terrace'),
                adapter.get('floor_count'),
                adapter.get('property_levels'),
                1 if adapter.get('let_agreed') else 0,
                adapter.get('agent_name'),
                adapter.get('agent_phone'),
                adapter.get('summary'),
                adapter.get('description'),
                features,
                adapter.get('added_date'),
                adapter.get('scraped_at'),
            ))

            self.total_inserted += 1
            self.pending_count += 1

            # Batch commits for performance
            if self.pending_count >= self.BATCH_SIZE:
                self.conn.commit()
                self.batch_count += 1
                logger.info(
                    f"[PIPELINE:SQLite] Batch {self.batch_count} committed "
                    f"({self.total_inserted} total)"
                )
                self.pending_count = 0

        except sqlite3.Error as e:
            self.total_errors += 1
            logger.error(
                f"[PIPELINE:SQLite] Insert failed for {adapter.get('property_id')}: {e}"
            )

        return item

    def close_spider(self, spider):
        if self.conn:
            # Commit any remaining items
            if self.pending_count > 0:
                self.conn.commit()
                logger.info(
                    f"[PIPELINE:SQLite] Final batch committed ({self.pending_count} items)"
                )

            elapsed = time.time() - self.start_time if self.start_time else 0

            # Log final count
            self.cursor.execute('SELECT COUNT(*) FROM listings')
            count = self.cursor.fetchone()[0]

            # Get area breakdown
            self.cursor.execute('''
                SELECT area, COUNT(*) as cnt
                FROM listings
                GROUP BY area
                ORDER BY cnt DESC
            ''')
            area_counts = self.cursor.fetchall()

            logger.info("[PIPELINE:SQLite] Complete:")
            logger.info(f"  Database: {self.db_path}")
            logger.info(f"  Total records: {count}")
            logger.info(f"  Inserted this run: {self.total_inserted}")
            logger.info(f"  Errors: {self.total_errors}")
            logger.info(f"  Batches: {self.batch_count + 1}")
            logger.info(f"  Duration: {elapsed:.1f}s")
            logger.info("[PIPELINE:SQLite] By area:")
            for area, cnt in area_counts:
                logger.info(f"    {area}: {cnt}")

            self.conn.close()
