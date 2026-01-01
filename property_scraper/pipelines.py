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
import fcntl
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

        # Issue #29 FIX: Always generate fingerprint (PRD-003)
        # generate_fingerprint() handles vague addresses using source+property_id fallback
        if not adapter.get('address_fingerprint'):
            address = adapter.get('address', '') or ''
            postcode = adapter.get('postcode', '') or ''
            source = adapter.get('source', '') or ''
            property_id = adapter.get('property_id', '') or ''
            # Always attempt fingerprint generation - function handles missing data
            adapter['address_fingerprint'] = self._generate_fingerprint(
                address, postcode,
                source=source,
                property_id=str(property_id) if property_id else None
            )
            if adapter['address_fingerprint']:
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
    """
    Filter duplicate listings by:
    1. source + property_id (exact same listing)
    2. fingerprint + price + bedrooms (same property, different listing IDs)

    The second check catches same property listed by multiple agents on aggregators
    like Rightmove, where each agent creates a new listing ID for the same flat.

    Issue #14 FIX: Uses bounded OrderedDict to prevent unbounded memory growth.
    When capacity is reached, oldest entries are evicted (LRU-style).
    """

    # Maximum number of entries to keep in dedup caches
    MAX_CACHE_SIZE = 50000

    def __init__(self):
        # Issue #14 FIX: Use OrderedDict for bounded LRU-style cache
        from collections import OrderedDict
        self._seen_ids = OrderedDict()  # source:property_id -> True
        self._seen_properties = OrderedDict()  # fingerprint:price:beds -> True
        self.id_duplicates = 0
        self.content_duplicates = 0

    def _add_to_cache(self, cache, key):
        """Add key to bounded cache, evicting oldest if at capacity."""
        if key in cache:
            # Move to end (most recently used)
            cache.move_to_end(key)
            return False  # Already existed
        if len(cache) >= self.MAX_CACHE_SIZE:
            # Evict oldest entry
            cache.popitem(last=False)
        cache[key] = True
        return True  # New entry

    def _in_cache(self, cache, key):
        """Check if key exists in cache."""
        return key in cache

    def open_spider(self, spider):
        logger.info("[PIPELINE:Dedupe] Initialized - tracking IDs and content signatures")

    def process_item(self, item, spider):
        from scrapy.exceptions import DropItem
        adapter = ItemAdapter(item)
        source = adapter.get('source', '')
        prop_id = adapter.get('property_id', '')

        # Check 1: Exact ID match (same listing scraped twice)
        id_key = f"{source}:{prop_id}"
        if self._in_cache(self._seen_ids, id_key):
            self.id_duplicates += 1
            raise DropItem(f"Duplicate item: {id_key}")
        self._add_to_cache(self._seen_ids, id_key)

        # Check 2: Content-based match (same property, different listing ID)
        # Only apply to aggregators where multi-agent listings are common
        fingerprint = adapter.get('address_fingerprint', '')
        price = adapter.get('price_pcm', 0)
        beds = adapter.get('bedrooms', 0)

        if fingerprint and price:
            content_key = f"{source}:{fingerprint}:{price}:{beds}"
            if self._in_cache(self._seen_properties, content_key):
                self.content_duplicates += 1
                logger.warning(
                    f"[PIPELINE:Dedupe] Content duplicate filtered: {prop_id} "
                    f"(same as existing {fingerprint[:8]}... @ £{price})"
                )
                raise DropItem(f"Content duplicate: {fingerprint[:8]}:{price}:{beds}")
            self._add_to_cache(self._seen_properties, content_key)

        return item

    def close_spider(self, spider):
        logger.info(
            f"[PIPELINE:Dedupe] Complete - {len(self._seen_ids)} unique IDs, "
            f"{self.id_duplicates} ID dupes, {self.content_duplicates} content dupes filtered"
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
        # Issue #13 FIX: Include source in filename to prevent race conditions
        # when multiple spiders write to same area file concurrently
        source = adapter.get('source', 'unknown')
        filepath = os.path.join(self.output_dir, f"{area.lower()}_{source}_listings.jsonl")
        try:
            line = json.dumps(dict(adapter)) + '\n'
            with open(filepath, 'a') as f:
                # Issue #13 FIX: Use file locking to prevent interleaved writes
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
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

    Price Change Validation:
    - Minimum 1 hour between price changes to prevent same-session false positives
    - Fingerprint fallback requires price within 20% to prevent cross-unit merging
    """

    BATCH_SIZE = 100  # Commit every N items
    MIN_PRICE_CHANGE_HOURS = 1  # Minimum hours between price changes
    FINGERPRINT_PRICE_TOLERANCE = 0.20  # 20% price tolerance for fingerprint matching

    # Price oscillation filter thresholds (Issue: weekly/monthly conversion noise)
    # A change must exceed BOTH thresholds to be recorded as a real price change
    # This filters out noise like £4914 <-> £4940 (0.5%, £26) while preserving
    # real changes like £3000 -> £2850 (5%, £150)
    MIN_PRICE_CHANGE_ABS = 50  # Minimum absolute change in £
    MIN_PRICE_CHANGE_PCT = 0.02  # Minimum percentage change (2%)

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
        self.conn = sqlite3.connect(self.db_path, timeout=60)  # Wait up to 60s for lock
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL for better concurrency
        self.conn.execute("PRAGMA busy_timeout = 60000")  # 60s busy timeout
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
        """Ensure database schema is in place, with auto-migration for new installs.

        Issue #16 FIX: Instead of crashing on missing schema, attempts to create
        tables and columns automatically. Only fails if auto-migration fails.

        Handles:
        1. New database (no tables) - creates full schema
        2. Existing database missing columns - adds columns
        3. Missing price_history table - creates it
        """
        # Check if listings table exists at all
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='listings'"
        )
        if not self.cursor.fetchone():
            # New database - create full schema
            logger.info("[PIPELINE:SQLite] New database detected, creating schema...")
            self._create_full_schema()
            return

        # Check for required columns
        self.cursor.execute("PRAGMA table_info(listings)")
        columns = {row[1] for row in self.cursor.fetchall()}

        required = {'address_fingerprint', 'first_seen', 'last_seen', 'is_active', 'price_change_count'}
        missing = required - columns

        if missing:
            logger.info(f"[PIPELINE:SQLite] Missing columns detected: {missing}. Auto-migrating...")
            try:
                self._add_missing_columns(missing)
            except Exception as e:
                raise RuntimeError(
                    f"Auto-migration failed for columns {missing}: {e}\n"
                    f"Please run 'python migrate_schema_v2.py' manually."
                )

        # Check price_history table exists
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='price_history'"
        )
        if not self.cursor.fetchone():
            logger.info("[PIPELINE:SQLite] Creating price_history table...")
            try:
                self._create_price_history_table()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create price_history table: {e}\n"
                    f"Please run 'python migrate_schema_v2.py' manually."
                )

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

    def _create_full_schema(self):
        """Create full database schema for new installs."""
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
                size_sqm REAL,
                furnished TEXT,
                epc_rating TEXT,
                floorplan_url TEXT,
                room_details TEXT,
                has_basement INTEGER DEFAULT 0,
                has_lower_ground INTEGER DEFAULT 0,
                has_ground INTEGER DEFAULT 0,
                has_mezzanine INTEGER DEFAULT 0,
                has_first_floor INTEGER DEFAULT 0,
                has_second_floor INTEGER DEFAULT 0,
                has_third_floor INTEGER DEFAULT 0,
                has_fourth_plus INTEGER DEFAULT 0,
                has_roof_terrace INTEGER DEFAULT 0,
                floor_count INTEGER,
                property_levels TEXT,
                let_agreed INTEGER DEFAULT 0,
                agent_name TEXT,
                agent_phone TEXT,
                summary TEXT,
                description TEXT,
                features TEXT,
                added_date TEXT,
                address_fingerprint TEXT,
                first_seen TEXT,
                last_seen TEXT,
                is_active INTEGER DEFAULT 1,
                price_change_count INTEGER DEFAULT 0,
                scraped_at TEXT,
                UNIQUE(source, property_id)
            )
        ''')
        self._create_price_history_table()
        self.conn.commit()
        logger.info("[PIPELINE:SQLite] Full schema created successfully")

    def _add_missing_columns(self, missing_columns):
        """Add missing columns to existing listings table."""
        column_defaults = {
            'address_fingerprint': 'TEXT',
            'first_seen': 'TEXT',
            'last_seen': 'TEXT',
            'is_active': 'INTEGER DEFAULT 1',
            'price_change_count': 'INTEGER DEFAULT 0',
        }
        for col in missing_columns:
            col_type = column_defaults.get(col, 'TEXT')
            self.cursor.execute(f'ALTER TABLE listings ADD COLUMN {col} {col_type}')
            logger.info(f"[PIPELINE:SQLite] Added column: {col} ({col_type})")
        self.conn.commit()

    def _create_price_history_table(self):
        """Create price_history table for tracking price changes."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                listing_id INTEGER NOT NULL,
                price_pcm INTEGER,
                recorded_at TEXT,
                FOREIGN KEY (listing_id) REFERENCES listings(id)
            )
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_price_history_listing
            ON price_history(listing_id)
        ''')

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

            # Check if record exists - first by exact (source, property_id) match
            self.cursor.execute(
                '''SELECT id, price_pcm, first_seen, price_change_count, property_id
                   FROM listings WHERE source=? AND property_id=?''',
                (source, property_id)
            )
            existing = self.cursor.fetchone()
            matched_by_fingerprint = False

            # If not found by property_id, try fingerprint match (same property, relisted with new ID)
            # CRITICAL: Also require price within tolerance to prevent cross-unit merging
            # (Different flats at same building have same fingerprint but different prices)
            if not existing:
                fingerprint = adapter.get('address_fingerprint')
                bedrooms = adapter.get('bedrooms')
                new_price = adapter.get('price_pcm')
                if fingerprint and bedrooms and new_price:
                    # Match by source + fingerprint + bedrooms + price within tolerance
                    self.cursor.execute(
                        '''SELECT id, price_pcm, first_seen, price_change_count, property_id
                           FROM listings WHERE source=? AND address_fingerprint=? AND bedrooms=?
                           ORDER BY last_seen DESC LIMIT 1''',
                        (source, fingerprint, bedrooms)
                    )
                    candidate = self.cursor.fetchone()
                    if candidate:
                        old_price = candidate[1]
                        # Only match if price is within tolerance (prevents cross-unit merging)
                        if old_price and new_price:
                            price_diff_pct = abs(new_price - old_price) / old_price
                            if price_diff_pct <= self.FINGERPRINT_PRICE_TOLERANCE:
                                existing = candidate
                                matched_by_fingerprint = True
                                old_pid = candidate[4]
                                logger.info(f"[PIPELINE:SQLite] Fingerprint match: {source} pid {old_pid} -> {property_id} (price diff {price_diff_pct:.1%})")
                            else:
                                logger.debug(f"[PIPELINE:SQLite] Fingerprint match rejected: price diff {price_diff_pct:.1%} > {self.FINGERPRINT_PRICE_TOLERANCE:.0%}")

            # FALLBACK 3: Match by price + size + district + bedrooms (catches address variations)
            # Same property listed with different addresses (e.g., "Young Street" vs "Imperial House")
            if not existing:
                new_price = adapter.get('price_pcm')
                new_size = adapter.get('size_sqft')
                bedrooms = adapter.get('bedrooms')
                postcode = adapter.get('postcode', '')
                # Extract district from postcode (e.g., "SW1W 8AT" -> "SW1W")
                district = postcode.split()[0] if postcode and ' ' in postcode else postcode

                if new_price and new_size and bedrooms and district:
                    # Exact match on price, size, bedrooms, district within same source
                    self.cursor.execute(
                        '''SELECT id, price_pcm, first_seen, price_change_count, property_id, address
                           FROM listings
                           WHERE source=? AND price_pcm=? AND size_sqft=? AND bedrooms=?
                             AND (postcode LIKE ? OR postcode LIKE ?)
                           ORDER BY last_seen DESC LIMIT 1''',
                        (source, new_price, new_size, bedrooms, f"{district} %", f"{district}")
                    )
                    candidate = self.cursor.fetchone()
                    if candidate:
                        existing = candidate[:5]  # Same structure as other queries
                        matched_by_fingerprint = True  # Reuse flag for property_id update
                        old_pid = candidate[4]
                        old_addr = candidate[5][:30] if candidate[5] else 'unknown'
                        new_addr = (adapter.get('address', '')[:30] or 'unknown')
                        logger.info(f"[PIPELINE:SQLite] Content match: {source} '{old_addr}' -> '{new_addr}' (same price/size/beds/district)")

            if existing:
                # UPDATE existing record
                listing_id, old_price, first_seen, change_count, _ = existing
                new_price = adapter.get('price_pcm')

                # Detect price change (only if both prices are non-null)
                # CRITICAL: Multiple safeguards to prevent false positives:
                # 1. Check if change exceeds noise thresholds (filters oscillations)
                # 2. Check if enough time has passed since last record
                if old_price and new_price and old_price != new_price:
                    # First: Check if change is significant (not just noise/oscillation)
                    if not self._is_significant_price_change(old_price, new_price):
                        self.stats['price_changes_noise'] = self.stats.get('price_changes_noise', 0) + 1
                    # Second: Check if enough time has passed since last price record
                    elif not self._can_record_price_change(listing_id, now):
                        self.stats['price_changes_skipped'] = self.stats.get('price_changes_skipped', 0) + 1
                        logger.debug(f"[PIPELINE:SQLite] Price change skipped (too recent): {property_id}")
                    else:
                        # Both checks passed - record the change
                        self._log_price_change(listing_id, new_price, now)
                        change_count = (change_count or 0) + 1
                        self.stats['price_changes'] += 1
                        logger.info(
                            f"[PIPELINE:SQLite] Price change recorded: {property_id} "
                            f"£{old_price} -> £{new_price} ({(new_price-old_price)/old_price:+.1%})"
                        )

                # Update record, preserving first_seen
                # If matched by fingerprint, also update property_id to the new one
                new_property_id = property_id if matched_by_fingerprint else None
                self._update_listing(adapter, listing_id, first_seen, now, change_count or 0, new_property_id)
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
                # Retry commit with exponential backoff for database locks
                for attempt in range(5):
                    try:
                        self.conn.commit()
                        self.batch_count += 1
                        total = self.stats['inserted'] + self.stats['updated']
                        logger.info(
                            f"[PIPELINE:SQLite] Batch {self.batch_count} committed "
                            f"({total} total, {self.stats['price_changes']} price changes)"
                        )
                        self.pending_count = 0
                        break
                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e) and attempt < 4:
                            wait_time = (attempt + 1) * 2  # 2s, 4s, 6s, 8s
                            logger.warning(f"[PIPELINE:SQLite] Database locked, retrying in {wait_time}s ({attempt + 1}/5)")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"[PIPELINE:SQLite] Batch commit failed after 5 attempts: {e}")
                    except sqlite3.Error as e:
                        logger.error(f"[PIPELINE:SQLite] Batch commit failed: {e}")
                        break

        return item

    def _is_significant_price_change(self, old_price, new_price):
        """Check if price change exceeds noise thresholds.

        Filters out oscillation noise from weekly/monthly conversion differences
        (e.g., £4914 <-> £4940 caused by different conversion factors).

        A change must exceed BOTH thresholds to be considered significant:
        - Absolute change >= MIN_PRICE_CHANGE_ABS (£50)
        - Percentage change >= MIN_PRICE_CHANGE_PCT (2%)

        Returns True if the change is significant enough to record.
        """
        if not old_price or not new_price:
            return True  # Can't calculate, allow the change

        abs_change = abs(new_price - old_price)
        pct_change = abs_change / old_price

        is_significant = (abs_change >= self.MIN_PRICE_CHANGE_ABS and
                          pct_change >= self.MIN_PRICE_CHANGE_PCT)

        if not is_significant:
            logger.debug(
                f"[PIPELINE:SQLite] Price change filtered as noise: "
                f"£{old_price} -> £{new_price} (£{abs_change}, {pct_change:.1%})"
            )

        return is_significant

    def _can_record_price_change(self, listing_id, now):
        """Check if enough time has passed since last price record.

        Prevents same-session false positives where different properties
        with same fingerprint get merged and treated as price changes.

        Returns True if at least MIN_PRICE_CHANGE_HOURS have passed since
        the last price_history record for this listing.
        """
        self.cursor.execute('''
            SELECT MAX(recorded_at) FROM price_history WHERE listing_id = ?
        ''', (listing_id,))
        result = self.cursor.fetchone()

        if not result or not result[0]:
            return True  # No previous price record

        last_recorded = result[0]
        try:
            from datetime import datetime
            last_dt = datetime.fromisoformat(last_recorded.replace('Z', '+00:00'))
            now_dt = datetime.fromisoformat(now.replace('Z', '+00:00'))
            hours_diff = (now_dt - last_dt).total_seconds() / 3600
            return hours_diff >= self.MIN_PRICE_CHANGE_HOURS
        except (ValueError, AttributeError):
            return True  # On parse error, allow the change

    def _log_price_change(self, listing_id, price_pcm, recorded_at):
        """Log price to history table."""
        self.cursor.execute('''
            INSERT INTO price_history (listing_id, price_pcm, recorded_at)
            VALUES (?, ?, ?)
        ''', (listing_id, price_pcm, recorded_at))

    def _update_listing(self, adapter, listing_id, first_seen, now, change_count, new_property_id=None):
        """Update existing listing, preserving first_seen.

        Uses COALESCE for 'sticky' fields (address, sqft, bedrooms, etc.) to prevent
        NULL values from overwriting existing data. This protects against data loss
        when a spider temporarily fails to extract a field that was previously captured.
        (Fix for Codex review finding: unconditional updates could erase enriched data)

        If new_property_id is provided, updates property_id (for fingerprint-matched records
        where the source changed the property ID on relist).
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
                property_id=COALESCE(?, property_id),
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
            new_property_id,
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
        """Close database connection with guaranteed cleanup.

        CRITICAL FIX: Wrap all operations in try/finally to ensure connection
        is ALWAYS closed, even if logging/stats queries fail.
        """
        if not self.conn:
            return

        try:
            # Commit any pending items with retry logic
            if self.pending_count > 0:
                for attempt in range(5):
                    try:
                        self.conn.commit()
                        logger.info(
                            f"[PIPELINE:SQLite] Final batch committed ({self.pending_count} items)"
                        )
                        break
                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e) and attempt < 4:
                            wait_time = (attempt + 1) * 2
                            logger.warning(f"[PIPELINE:SQLite] Final commit: database locked, retrying in {wait_time}s ({attempt + 1}/5)")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"[PIPELINE:SQLite] Final commit failed after 5 attempts: {e}")
                            self.stats['errors'] += self.pending_count
                    except sqlite3.Error as e:
                        logger.error(f"[PIPELINE:SQLite] Final commit failed: {e}")
                        self.stats['errors'] += self.pending_count
                        break

            # Gather stats for logging (non-critical, wrapped in try/except)
            elapsed = time.time() - self.start_time if self.start_time else 0
            total_records = 0
            price_history_count = 0
            area_counts = []

            try:
                if self.cursor:
                    self.cursor.execute('SELECT COUNT(*) FROM listings')
                    total_records = self.cursor.fetchone()[0]

                    self.cursor.execute('SELECT COUNT(*) FROM price_history')
                    price_history_count = self.cursor.fetchone()[0]

                    self.cursor.execute('''
                        SELECT area, COUNT(*) as cnt
                        FROM listings
                        GROUP BY area
                        ORDER BY cnt DESC
                    ''')
                    area_counts = self.cursor.fetchall()
            except sqlite3.Error as e:
                logger.warning(f"[PIPELINE:SQLite] Error gathering final stats: {e}")

            # Log summary
            logger.info("[PIPELINE:SQLite] Complete:")
            logger.info(f"  Database: {self.db_path}")
            logger.info(f"  Total records: {total_records}")
            logger.info(f"  Inserted: {self.stats['inserted']}")
            logger.info(f"  Updated: {self.stats['updated']}")
            logger.info(f"  Price changes logged: {self.stats['price_changes']}")
            logger.info(f"  Price changes filtered (noise): {self.stats.get('price_changes_noise', 0)}")
            logger.info(f"  Price changes skipped (too recent): {self.stats.get('price_changes_skipped', 0)}")
            logger.info(f"  Price history entries: {price_history_count}")
            logger.info(f"  Errors: {self.stats['errors']}")
            logger.info(f"  Batches: {self.batch_count + 1}")
            logger.info(f"  Duration: {elapsed:.1f}s")
            if area_counts:
                logger.info("[PIPELINE:SQLite] By area:")
                for area, cnt in area_counts:
                    logger.info(f"    {area}: {cnt}")

        finally:
            # ALWAYS close cursor and connection, even if above code fails
            try:
                if self.cursor:
                    self.cursor.close()
            except Exception as e:
                logger.debug(f"[PIPELINE:SQLite] Error closing cursor: {e}")

            try:
                self.conn.close()
                logger.debug("[PIPELINE:SQLite] Database connection closed")
            except Exception as e:
                logger.error(f"[PIPELINE:SQLite] Error closing connection: {e}")
