"""
PostgreSQL Pipelines for Vercel Postgres.

Replaces SQLite pipeline for cloud deployment.
Uses psycopg2 for direct Postgres connection.
"""

import os
import json
import logging
import time
from datetime import datetime
from urllib.parse import urlparse

import psycopg2
from psycopg2.extras import execute_values
from itemadapter import ItemAdapter

logger = logging.getLogger(__name__)


class PostgresPipeline:
    """Persist items to Vercel Postgres with historical tracking.

    Features:
    - Smart upsert: INSERT new, UPDATE existing (preserves first_seen)
    - Price history: Logs initial price AND all price changes
    - Connection pooling via psycopg2
    """

    BATCH_SIZE = 50  # Smaller batch for Postgres
    MIN_PRICE_CHANGE_HOURS = 1
    FINGERPRINT_PRICE_TOLERANCE = 0.20

    def __init__(self):
        self.conn = None
        self.cursor = None
        self.pending_items = []
        self.stats = {
            'inserted': 0,
            'updated': 0,
            'price_changes': 0,
            'errors': 0,
        }
        self.start_time = None

    def _get_connection_string(self):
        """Get Postgres connection string from environment."""
        # Try various env var names used by Vercel
        for var in ['POSTGRES_URL', 'DATABASE_URL', 'POSTGRES_URL_NON_POOLING']:
            url = os.environ.get(var)
            if url:
                return url
        raise RuntimeError(
            "No Postgres connection string found. "
            "Set POSTGRES_URL environment variable."
        )

    def open_spider(self, spider):
        conn_string = self._get_connection_string()

        # Parse and potentially modify connection string
        parsed = urlparse(conn_string)

        self.conn = psycopg2.connect(conn_string)
        self.conn.autocommit = False
        self.cursor = self.conn.cursor()

        self._ensure_schema()
        self.start_time = time.time()

        # Log existing count
        self.cursor.execute('SELECT COUNT(*) FROM listings')
        existing = self.cursor.fetchone()[0]
        logger.info(f"[PIPELINE:Postgres] Initialized - connected to Vercel Postgres")
        logger.info(f"[PIPELINE:Postgres] Existing records: {existing}")

    def _ensure_schema(self):
        """Ensure database schema exists."""
        # Check if tables exist
        self.cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'listings'
            )
        """)
        if not self.cursor.fetchone()[0]:
            logger.info("[PIPELINE:Postgres] Creating schema...")
            self._create_schema()
        self.conn.commit()

    def _create_schema(self):
        """Create full database schema."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS listings (
                id SERIAL PRIMARY KEY,
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

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id SERIAL PRIMARY KEY,
                listing_id INTEGER NOT NULL REFERENCES listings(id),
                price_pcm INTEGER,
                recorded_at TEXT
            )
        ''')

        # Create indexes
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_prop ON listings(source, property_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_fingerprint ON listings(address_fingerprint)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_active ON listings(is_active)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_history_listing ON price_history(listing_id)')

    def process_item(self, item, spider):
        """Process item - upsert to Postgres."""
        adapter = ItemAdapter(item)
        now = datetime.utcnow().isoformat()

        source = adapter.get('source')
        property_id = adapter.get('property_id')

        try:
            # Check if record exists
            self.cursor.execute(
                '''SELECT id, price_pcm, first_seen, price_change_count, property_id
                   FROM listings WHERE source=%s AND property_id=%s''',
                (source, property_id)
            )
            existing = self.cursor.fetchone()
            matched_by_fingerprint = False

            # Fingerprint fallback
            if not existing:
                fingerprint = adapter.get('address_fingerprint')
                bedrooms = adapter.get('bedrooms')
                new_price = adapter.get('price_pcm')
                if fingerprint and bedrooms and new_price:
                    self.cursor.execute(
                        '''SELECT id, price_pcm, first_seen, price_change_count, property_id
                           FROM listings WHERE source=%s AND address_fingerprint=%s AND bedrooms=%s
                           ORDER BY last_seen DESC LIMIT 1''',
                        (source, fingerprint, bedrooms)
                    )
                    candidate = self.cursor.fetchone()
                    if candidate:
                        old_price = candidate[1]
                        if old_price and new_price:
                            price_diff_pct = abs(new_price - old_price) / old_price
                            if price_diff_pct <= self.FINGERPRINT_PRICE_TOLERANCE:
                                existing = candidate
                                matched_by_fingerprint = True

            if existing:
                listing_id, old_price, first_seen, change_count, _ = existing
                new_price = adapter.get('price_pcm')

                # Price change detection
                if old_price and new_price and old_price != new_price:
                    self._log_price_change(listing_id, new_price, now)
                    change_count = (change_count or 0) + 1
                    self.stats['price_changes'] += 1

                new_property_id = property_id if matched_by_fingerprint else None
                self._update_listing(adapter, listing_id, first_seen, now, change_count or 0, new_property_id)
                self.stats['updated'] += 1
            else:
                listing_id = self._insert_listing(adapter, now)
                if adapter.get('price_pcm'):
                    self._log_price_change(listing_id, adapter.get('price_pcm'), now)
                self.stats['inserted'] += 1

            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            self.stats['errors'] += 1
            logger.error(f"[PIPELINE:Postgres] Error for {property_id}: {e}")

        return item

    def _log_price_change(self, listing_id, price_pcm, recorded_at):
        """Log price to history table."""
        self.cursor.execute('''
            INSERT INTO price_history (listing_id, price_pcm, recorded_at)
            VALUES (%s, %s, %s)
        ''', (listing_id, price_pcm, recorded_at))

    def _update_listing(self, adapter, listing_id, first_seen, now, change_count, new_property_id=None):
        """Update existing listing."""
        features = adapter.get('features', [])
        if isinstance(features, list):
            features = json.dumps(features)

        room_details = adapter.get('room_details')
        if isinstance(room_details, (dict, list)):
            room_details = json.dumps(room_details)

        self.cursor.execute('''
            UPDATE listings SET
                property_id=COALESCE(%s, property_id),
                url=%s,
                area=COALESCE(%s, area),
                price=%s, price_pw=%s, price_pcm=%s, price_period=%s,
                address=COALESCE(%s, address),
                postcode=COALESCE(%s, postcode),
                latitude=COALESCE(%s, latitude),
                longitude=COALESCE(%s, longitude),
                bedrooms=COALESCE(%s, bedrooms),
                bathrooms=COALESCE(%s, bathrooms),
                reception_rooms=COALESCE(%s, reception_rooms),
                property_type=COALESCE(%s, property_type),
                size_sqft=COALESCE(%s, size_sqft),
                size_sqm=COALESCE(%s, size_sqm),
                furnished=COALESCE(%s, furnished),
                epc_rating=COALESCE(%s, epc_rating),
                floorplan_url=COALESCE(%s, floorplan_url),
                room_details=COALESCE(%s, room_details),
                has_basement=COALESCE(%s, has_basement),
                has_lower_ground=COALESCE(%s, has_lower_ground),
                has_ground=COALESCE(%s, has_ground),
                has_mezzanine=COALESCE(%s, has_mezzanine),
                has_first_floor=COALESCE(%s, has_first_floor),
                has_second_floor=COALESCE(%s, has_second_floor),
                has_third_floor=COALESCE(%s, has_third_floor),
                has_fourth_plus=COALESCE(%s, has_fourth_plus),
                has_roof_terrace=COALESCE(%s, has_roof_terrace),
                floor_count=COALESCE(%s, floor_count),
                property_levels=COALESCE(%s, property_levels),
                let_agreed=%s,
                agent_name=COALESCE(%s, agent_name),
                agent_phone=COALESCE(%s, agent_phone),
                summary=%s, description=%s, features=%s, added_date=%s,
                address_fingerprint=COALESCE(%s, address_fingerprint),
                last_seen=%s, is_active=1, price_change_count=%s,
                scraped_at=%s
            WHERE id=%s
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
        """Insert new listing."""
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
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1, 0, %s
            )
            RETURNING id
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
            now, now,
            adapter.get('scraped_at'),
        ))

        return self.cursor.fetchone()[0]

    def close_spider(self, spider):
        """Close database connection."""
        if not self.conn:
            return

        try:
            elapsed = time.time() - self.start_time if self.start_time else 0

            self.cursor.execute('SELECT COUNT(*) FROM listings')
            total_records = self.cursor.fetchone()[0]

            logger.info("[PIPELINE:Postgres] Complete:")
            logger.info(f"  Total records: {total_records}")
            logger.info(f"  Inserted: {self.stats['inserted']}")
            logger.info(f"  Updated: {self.stats['updated']}")
            logger.info(f"  Price changes: {self.stats['price_changes']}")
            logger.info(f"  Errors: {self.stats['errors']}")
            logger.info(f"  Duration: {elapsed:.1f}s")

        finally:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
