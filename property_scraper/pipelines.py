"""
Scrapy Pipelines for property data processing.

Pipelines:
1. CleanDataPipeline - Normalize and validate data, generate fingerprints
2. DuplicateFilterPipeline - Filter duplicate listings within session
3. JsonWriterPipeline - Write to JSON files
4. SQLitePipeline - Persist to SQLite with historical tracking

History tracking (PRD-003):
- first_seen: Set once on initial insert
- last_seen: Updated on every scrape
- price_history: Logged on insert AND on price changes
- is_active: Set to 1 on every update (separate deactivation process)
"""

import os
import json
import sqlite3
import logging
import time
from datetime import datetime
from itemadapter import ItemAdapter

logger = logging.getLogger(__name__)


def _import_fingerprint_service():
    """Lazy import of fingerprint service to avoid circular imports."""
    from property_scraper.services.fingerprint import generate_fingerprint
    return generate_fingerprint


class CleanDataPipeline:
    """Clean and normalize property data, generate fingerprints."""

    def __init__(self):
        self.items_processed = 0
        self.fixes_applied = 0
        self.fingerprints_generated = 0
        self._generate_fingerprint = None

    def open_spider(self, spider):
        # Lazy load fingerprint service
        self._generate_fingerprint = _import_fingerprint_service()
        logger.info("[PIPELINE:Clean] Initialized - will normalize prices, clean text, generate fingerprints")

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

        # Generate fingerprint if address or postcode available (PRD-003)
        if not adapter.get('address_fingerprint'):
            address = adapter.get('address', '') or ''
            postcode = adapter.get('postcode', '') or ''
            if address or postcode:
                adapter['address_fingerprint'] = self._generate_fingerprint(address, postcode)
                self.fingerprints_generated += 1
                fixes.append('generated_fingerprint')

        self.items_processed += 1
        if fixes:
            self.fixes_applied += len(fixes)
            logger.debug(f"[PIPELINE:Clean] {adapter.get('property_id')}: {', '.join(fixes)}")

        return item

    def close_spider(self, spider):
        logger.info(
            f"[PIPELINE:Clean] Complete - {self.items_processed} items, "
            f"{self.fixes_applied} fixes applied, {self.fingerprints_generated} fingerprints generated"
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
    """Persist items to SQLite with historical tracking (PRD-003).

    Features:
    - Smart upsert: INSERT new, UPDATE existing (preserves first_seen)
    - Price history: Logs initial price AND all price changes
    - SAVEPOINT-based atomicity: Per-item rollback on errors
    - Compound index: O(1) lookup on (source, property_id)
    """

    BATCH_SIZE = 100  # Commit every N items

    def __init__(self):
        self.conn = None
        self.cursor = None
        self.pending_count = 0
        self.stats = {
            'inserted': 0,
            'updated': 0,
            'price_changes': 0,
            'errors': 0,
        }
        self.batch_count = 0
        self.start_time = None
        self.db_path = None

    def open_spider(self, spider):
        output_dir = spider.settings.get('OUTPUT_DIR', 'output')
        os.makedirs(output_dir, exist_ok=True)

        self.db_path = os.path.join(output_dir, 'rentals.db')
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cursor = self.conn.cursor()
        self._ensure_schema()
        self.start_time = time.time()

        # Log existing count
        self.cursor.execute('SELECT COUNT(*) FROM listings')
        existing = self.cursor.fetchone()[0]
        logger.info(f"[PIPELINE:SQLite] Initialized - {self.db_path}")
        logger.info(f"[PIPELINE:SQLite] Existing records: {existing}")
        logger.info(f"[PIPELINE:SQLite] Batch size: {self.BATCH_SIZE}")

    def _ensure_schema(self):
        """Ensure schema from PRD-001 migration is in place.

        Verifies required columns exist (from migrate_schema_v2.py).
        Creates compound index for O(1) lookups if missing.
        """
        # Check that migration has been run
        self.cursor.execute("PRAGMA table_info(listings)")
        columns = {row[1] for row in self.cursor.fetchall()}

        required = {'address_fingerprint', 'first_seen', 'last_seen', 'is_active', 'price_change_count'}
        missing = required - columns
        if missing:
            raise RuntimeError(
                f"Missing required columns: {missing}. Run migrate_schema_v2.py first."
            )

        # Check price_history table exists
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='price_history'"
        )
        if not self.cursor.fetchone():
            raise RuntimeError("price_history table missing. Run migrate_schema_v2.py first.")

        # Create compound index for O(1) lookup (Gemini recommendation)
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_source_prop ON listings(source, property_id)
        ''')

        # Ensure other useful indexes exist
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_area ON listings(area)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_price ON listings(price_pcm)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_bedrooms ON listings(bedrooms)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_fingerprint ON listings(address_fingerprint)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_active ON listings(is_active)')

        self.conn.commit()
        logger.debug("[PIPELINE:SQLite] Schema verified, indexes ensured")

    def process_item(self, item, spider):
        """Smart upsert with historical tracking (PRD-003).

        - New listings: INSERT with first_seen=last_seen=now, log initial price
        - Existing listings: UPDATE preserving first_seen, update last_seen
        - Price changes: Log to price_history, increment price_change_count
        - Uses SAVEPOINT for per-item atomicity (Gemini recommendation)
        """
        adapter = ItemAdapter(item)
        now = datetime.utcnow().isoformat()

        source = adapter.get('source')
        property_id = adapter.get('property_id')

        # CRITICAL FIX (Gemini review): Separate item processing from batch commit.
        # Previously, if commit() failed after RELEASE SAVEPOINT, the except block
        # would try to ROLLBACK TO SAVEPOINT that no longer exists, causing a crash.
        # Now: Item processing has its own try/except, batch commit has separate handling.

        item_success = False
        try:
            # SAVEPOINT for per-item rollback on errors
            self.cursor.execute("SAVEPOINT item_process")

            # Check if record exists (O(1) with compound index)
            self.cursor.execute(
                '''SELECT id, price_pcm, first_seen, price_change_count
                   FROM listings WHERE source=? AND property_id=?''',
                (source, property_id)
            )
            existing = self.cursor.fetchone()

            if existing:
                # UPDATE existing record
                listing_id, old_price, first_seen, change_count = existing
                new_price = adapter.get('price_pcm')

                # Detect price change (only if both prices are non-null)
                if old_price and new_price and old_price != new_price:
                    self._log_price_change(listing_id, new_price, now)
                    change_count = (change_count or 0) + 1
                    self.stats['price_changes'] += 1

                # Update record, preserving first_seen
                self._update_listing(adapter, listing_id, first_seen, now, change_count or 0)
                self.stats['updated'] += 1
            else:
                # INSERT new record
                listing_id = self._insert_listing(adapter, now)

                # Log initial price to price_history (Gemini recommendation)
                if adapter.get('price_pcm'):
                    self._log_price_change(listing_id, adapter.get('price_pcm'), now)

                self.stats['inserted'] += 1

            # Release savepoint (success)
            self.cursor.execute("RELEASE SAVEPOINT item_process")
            item_success = True

        except Exception as e:
            # Rollback ONLY this item's operations (savepoint still exists here)
            try:
                self.cursor.execute("ROLLBACK TO SAVEPOINT item_process")
            except sqlite3.Error:
                pass  # Savepoint may not exist if error was in SAVEPOINT creation
            self.stats['errors'] += 1
            logger.error(f"[PIPELINE:SQLite] Error for {property_id}: {e}")

        # Batch commit - separate from item try/except to avoid savepoint issues
        if item_success:
            self.pending_count += 1
            if self.pending_count >= self.BATCH_SIZE:
                try:
                    self.conn.commit()
                    self.batch_count += 1
                    total = self.stats['inserted'] + self.stats['updated']
                    logger.info(
                        f"[PIPELINE:SQLite] Batch {self.batch_count} committed "
                        f"({total} total, {self.stats['price_changes']} price changes)"
                    )
                    self.pending_count = 0
                except sqlite3.Error as e:
                    logger.error(f"[PIPELINE:SQLite] Batch commit failed: {e}")

        return item

    def _log_price_change(self, listing_id, price_pcm, recorded_at):
        """Log price to history table."""
        self.cursor.execute('''
            INSERT INTO price_history (listing_id, price_pcm, recorded_at)
            VALUES (?, ?, ?)
        ''', (listing_id, price_pcm, recorded_at))

    def _update_listing(self, adapter, listing_id, first_seen, now, change_count):
        """Update existing listing, preserving first_seen.

        Uses COALESCE for 'sticky' fields (address, sqft, bedrooms, etc.) to prevent
        NULL values from overwriting existing data. This protects against data loss
        when a spider temporarily fails to extract a field that was previously captured.
        (Fix for Codex review finding: unconditional updates could erase enriched data)
        """
        features = adapter.get('features', [])
        if isinstance(features, list):
            features = json.dumps(features)

        room_details = adapter.get('room_details')
        if isinstance(room_details, (dict, list)):
            room_details = json.dumps(room_details)

        # Use COALESCE for sticky fields - preserve existing data if new value is NULL
        # Sticky fields: structural data that doesn't change (sqft, beds, baths, address, etc.)
        # Non-sticky fields: always update (url, price, let_agreed, timestamps, descriptions)
        self.cursor.execute('''
            UPDATE listings SET
                url=?,
                area=COALESCE(?, area),
                price=?, price_pw=?, price_pcm=?, price_period=?,
                address=COALESCE(?, address),
                postcode=COALESCE(?, postcode),
                latitude=COALESCE(?, latitude),
                longitude=COALESCE(?, longitude),
                bedrooms=COALESCE(?, bedrooms),
                bathrooms=COALESCE(?, bathrooms),
                reception_rooms=COALESCE(?, reception_rooms),
                property_type=COALESCE(?, property_type),
                size_sqft=COALESCE(?, size_sqft),
                size_sqm=COALESCE(?, size_sqm),
                furnished=COALESCE(?, furnished),
                epc_rating=COALESCE(?, epc_rating),
                floorplan_url=COALESCE(?, floorplan_url),
                room_details=COALESCE(?, room_details),
                has_basement=COALESCE(?, has_basement),
                has_lower_ground=COALESCE(?, has_lower_ground),
                has_ground=COALESCE(?, has_ground),
                has_mezzanine=COALESCE(?, has_mezzanine),
                has_first_floor=COALESCE(?, has_first_floor),
                has_second_floor=COALESCE(?, has_second_floor),
                has_third_floor=COALESCE(?, has_third_floor),
                has_fourth_plus=COALESCE(?, has_fourth_plus),
                has_roof_terrace=COALESCE(?, has_roof_terrace),
                floor_count=COALESCE(?, floor_count),
                property_levels=COALESCE(?, property_levels),
                let_agreed=?,
                agent_name=COALESCE(?, agent_name),
                agent_phone=COALESCE(?, agent_phone),
                summary=?, description=?, features=?, added_date=?,
                address_fingerprint=COALESCE(?, address_fingerprint),
                last_seen=?, is_active=1, price_change_count=?,
                scraped_at=?
            WHERE id=?
        ''', (
            adapter.get('url'),
            adapter.get('area'),
            adapter.get('price'), adapter.get('price_pw'),
            adapter.get('price_pcm'), adapter.get('price_period'),
            adapter.get('address'), adapter.get('postcode'),
            adapter.get('latitude'), adapter.get('longitude'),
            adapter.get('bedrooms'), adapter.get('bathrooms'),
            adapter.get('reception_rooms'), adapter.get('property_type'),
            adapter.get('size_sqft'), adapter.get('size_sqm'),
            adapter.get('furnished'), adapter.get('epc_rating'),
            adapter.get('floorplan_url'), room_details,
            adapter.get('has_basement'), adapter.get('has_lower_ground'),
            adapter.get('has_ground'), adapter.get('has_mezzanine'),
            adapter.get('has_first_floor'), adapter.get('has_second_floor'),
            adapter.get('has_third_floor'), adapter.get('has_fourth_plus'),
            adapter.get('has_roof_terrace'), adapter.get('floor_count'),
            adapter.get('property_levels'),
            1 if adapter.get('let_agreed') else 0,
            adapter.get('agent_name'), adapter.get('agent_phone'),
            adapter.get('summary'), adapter.get('description'),
            features, adapter.get('added_date'),
            adapter.get('address_fingerprint'),
            now, change_count,
            adapter.get('scraped_at'),
            listing_id
        ))

    def _insert_listing(self, adapter, now):
        """Insert new listing with first_seen=last_seen=now.

        Returns the lastrowid for price history logging.
        """
        features = adapter.get('features', [])
        if isinstance(features, list):
            features = json.dumps(features)

        room_details = adapter.get('room_details')
        if isinstance(room_details, (dict, list)):
            room_details = json.dumps(room_details)

        self.cursor.execute('''
            INSERT INTO listings (
                source, property_id, url, area,
                price, price_pw, price_pcm, price_period,
                address, postcode, latitude, longitude,
                bedrooms, bathrooms, reception_rooms, property_type,
                size_sqft, size_sqm, furnished, epc_rating,
                floorplan_url, room_details,
                has_basement, has_lower_ground, has_ground, has_mezzanine,
                has_first_floor, has_second_floor, has_third_floor,
                has_fourth_plus, has_roof_terrace, floor_count, property_levels,
                let_agreed, agent_name, agent_phone,
                summary, description, features, added_date,
                address_fingerprint,
                first_seen, last_seen, is_active, price_change_count,
                scraped_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, 1, 0, ?
            )
        ''', (
            adapter.get('source'), adapter.get('property_id'),
            adapter.get('url'), adapter.get('area'),
            adapter.get('price'), adapter.get('price_pw'),
            adapter.get('price_pcm'), adapter.get('price_period'),
            adapter.get('address'), adapter.get('postcode'),
            adapter.get('latitude'), adapter.get('longitude'),
            adapter.get('bedrooms'), adapter.get('bathrooms'),
            adapter.get('reception_rooms'), adapter.get('property_type'),
            adapter.get('size_sqft'), adapter.get('size_sqm'),
            adapter.get('furnished'), adapter.get('epc_rating'),
            adapter.get('floorplan_url'), room_details,
            adapter.get('has_basement'), adapter.get('has_lower_ground'),
            adapter.get('has_ground'), adapter.get('has_mezzanine'),
            adapter.get('has_first_floor'), adapter.get('has_second_floor'),
            adapter.get('has_third_floor'), adapter.get('has_fourth_plus'),
            adapter.get('has_roof_terrace'), adapter.get('floor_count'),
            adapter.get('property_levels'),
            1 if adapter.get('let_agreed') else 0,
            adapter.get('agent_name'), adapter.get('agent_phone'),
            adapter.get('summary'), adapter.get('description'),
            features, adapter.get('added_date'),
            adapter.get('address_fingerprint'),
            now, now,  # first_seen = last_seen = now
            adapter.get('scraped_at'),
        ))

        return self.cursor.lastrowid

    def close_spider(self, spider):
        if self.conn:
            # CRITICAL FIX (Grok review): Wrap final commit in try/except
            # Previously, if final commit failed, it could crash and skip logging.
            if self.pending_count > 0:
                try:
                    self.conn.commit()
                    logger.info(
                        f"[PIPELINE:SQLite] Final batch committed ({self.pending_count} items)"
                    )
                except sqlite3.Error as e:
                    logger.error(f"[PIPELINE:SQLite] Final commit failed: {e}")
                    self.stats['errors'] += self.pending_count

            elapsed = time.time() - self.start_time if self.start_time else 0

            # Log final count
            self.cursor.execute('SELECT COUNT(*) FROM listings')
            total_records = self.cursor.fetchone()[0]

            # Count price history entries
            self.cursor.execute('SELECT COUNT(*) FROM price_history')
            price_history_count = self.cursor.fetchone()[0]

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
            logger.info(f"  Total records: {total_records}")
            logger.info(f"  Inserted: {self.stats['inserted']}")
            logger.info(f"  Updated: {self.stats['updated']}")
            logger.info(f"  Price changes logged: {self.stats['price_changes']}")
            logger.info(f"  Price history entries: {price_history_count}")
            logger.info(f"  Errors: {self.stats['errors']}")
            logger.info(f"  Batches: {self.batch_count + 1}")
            logger.info(f"  Duration: {elapsed:.1f}s")
            logger.info("[PIPELINE:SQLite] By area:")
            for area, cnt in area_counts:
                logger.info(f"    {area}: {cnt}")

            self.conn.close()
