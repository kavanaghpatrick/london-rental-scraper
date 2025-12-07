# PRD-003: Pipeline Modifications for Fingerprinting and History

## Overview
Modify the Scrapy pipeline to integrate fingerprinting at scrape time and implement proper historical tracking (first_seen, last_seen, price_history).

## Problem Statement
Current pipeline behavior:
- `INSERT OR REPLACE` overwrites existing records, losing historical data
- No fingerprint generated during scraping
- No price change tracking
- first_seen/last_seen not maintained during updates

## Goals
1. Generate fingerprint in `CleanDataPipeline`
2. Replace `INSERT OR REPLACE` with smart upsert logic
3. Track first_seen (only on first insert)
4. Update last_seen on every update
5. Log price changes to price_history table
6. Maintain backward compatibility

## Non-Goals
- Changing spider code
- Cross-source duplicate detection (handled by fingerprint + separate cleanup)
- Modifying JsonWriterPipeline

## Technical Design

### 1. CleanDataPipeline Modifications

Add fingerprint generation:

```python
from property_scraper.services.fingerprint import generate_fingerprint

class CleanDataPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        # ... existing cleaning logic ...

        # Generate fingerprint if address and postcode available
        address = adapter.get('address', '')
        postcode = adapter.get('postcode', '')
        if address or postcode:
            adapter['address_fingerprint'] = generate_fingerprint(address, postcode)

        return item
```

### 2. SQLitePipeline Modifications

Replace `INSERT OR REPLACE` with smart upsert:

```python
class SQLitePipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        now = datetime.utcnow().isoformat()

        source = adapter.get('source')
        property_id = adapter.get('property_id')

        # Check if record exists
        self.cursor.execute(
            'SELECT id, price_pcm, first_seen FROM listings WHERE source=? AND property_id=?',
            (source, property_id)
        )
        existing = self.cursor.fetchone()

        if existing:
            # UPDATE existing record
            listing_id, old_price, first_seen = existing
            new_price = adapter.get('price_pcm')

            # Log price change if different
            if old_price and new_price and old_price != new_price:
                self._log_price_change(listing_id, new_price, now)
                adapter['price_change_count'] = (adapter.get('price_change_count') or 0) + 1

            # Update record, preserving first_seen
            self._update_listing(adapter, listing_id, first_seen, now)
        else:
            # INSERT new record
            adapter['first_seen'] = now
            adapter['last_seen'] = now
            self._insert_listing(adapter)

        return item

    def _log_price_change(self, listing_id, new_price, recorded_at):
        self.cursor.execute('''
            INSERT INTO price_history (listing_id, price_pcm, recorded_at)
            VALUES (?, ?, ?)
        ''', (listing_id, new_price, recorded_at))

    def _update_listing(self, adapter, listing_id, first_seen, now):
        # Update all fields except first_seen
        self.cursor.execute('''
            UPDATE listings SET
                url=?, area=?, price=?, price_pw=?, price_pcm=?, price_period=?,
                address=?, postcode=?, latitude=?, longitude=?,
                bedrooms=?, bathrooms=?, reception_rooms=?, property_type=?,
                size_sqft=?, size_sqm=?, furnished=?, epc_rating=?,
                floorplan_url=?, room_details=?, features=?,
                let_agreed=?, agent_name=?, agent_phone=?,
                summary=?, description=?, added_date=?,
                address_fingerprint=?, last_seen=?, is_active=1,
                scraped_at=?
            WHERE id=?
        ''', (
            # ... all field values ...
            listing_id
        ))

    def _insert_listing(self, adapter):
        # INSERT new record with first_seen = last_seen = now
        # Similar to current INSERT but with new columns
        pass
```

### 3. Full Implementation

#### Modified `pipelines.py`

```python
class SQLitePipeline:
    """Persist items to SQLite with historical tracking."""

    BATCH_SIZE = 100

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

        self.cursor.execute('SELECT COUNT(*) FROM listings')
        existing = self.cursor.fetchone()[0]
        logger.info(f"[PIPELINE:SQLite] Initialized - {self.db_path}")
        logger.info(f"[PIPELINE:SQLite] Existing records: {existing}")

    def _ensure_schema(self):
        """Ensure all required columns exist (idempotent)."""
        # Schema is managed by migrate_schema_v2.py
        # Just verify key columns exist
        self.cursor.execute("PRAGMA table_info(listings)")
        columns = {row[1] for row in self.cursor.fetchall()}
        required = {'address_fingerprint', 'first_seen', 'last_seen', 'is_active'}
        missing = required - columns
        if missing:
            raise RuntimeError(
                f"Missing required columns: {missing}. Run migrate_schema_v2.py first."
            )

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        now = datetime.utcnow().isoformat()

        source = adapter.get('source')
        property_id = adapter.get('property_id')

        # Ensure fingerprint exists
        if not adapter.get('address_fingerprint'):
            from property_scraper.services.fingerprint import generate_fingerprint
            address = adapter.get('address', '')
            postcode = adapter.get('postcode', '')
            adapter['address_fingerprint'] = generate_fingerprint(address, postcode)

        try:
            # Check if record exists
            self.cursor.execute(
                '''SELECT id, price_pcm, first_seen, price_change_count
                   FROM listings WHERE source=? AND property_id=?''',
                (source, property_id)
            )
            existing = self.cursor.fetchone()

            if existing:
                listing_id, old_price, first_seen, change_count = existing
                new_price = adapter.get('price_pcm')

                # Detect price change
                if old_price and new_price and old_price != new_price:
                    self._log_price_change(listing_id, new_price, now)
                    change_count = (change_count or 0) + 1
                    self.stats['price_changes'] += 1

                # Update
                self._update_listing(adapter, listing_id, first_seen, now, change_count)
                self.stats['updated'] += 1
            else:
                # Insert
                self._insert_listing(adapter, now)
                self.stats['inserted'] += 1

            self.pending_count += 1
            if self.pending_count >= self.BATCH_SIZE:
                self.conn.commit()
                self.batch_count += 1
                self.pending_count = 0

        except sqlite3.Error as e:
            self.stats['errors'] += 1
            logger.error(f"[PIPELINE:SQLite] Error for {property_id}: {e}")

        return item

    def _log_price_change(self, listing_id, new_price, recorded_at):
        """Log price change to history table."""
        self.cursor.execute('''
            INSERT INTO price_history (listing_id, price_pcm, recorded_at)
            VALUES (?, ?, ?)
        ''', (listing_id, new_price, recorded_at))

    def _update_listing(self, adapter, listing_id, first_seen, now, change_count):
        """Update existing listing, preserving first_seen."""
        features = adapter.get('features', [])
        if isinstance(features, list):
            features = json.dumps(features)

        room_details = adapter.get('room_details')
        if isinstance(room_details, (dict, list)):
            room_details = json.dumps(room_details)

        self.cursor.execute('''
            UPDATE listings SET
                url=?, area=?, price=?, price_pw=?, price_pcm=?, price_period=?,
                address=?, postcode=?, latitude=?, longitude=?,
                bedrooms=?, bathrooms=?, reception_rooms=?, property_type=?,
                size_sqft=?, size_sqm=?, furnished=?, epc_rating=?,
                floorplan_url=?, room_details=?,
                let_agreed=?, agent_name=?, agent_phone=?,
                summary=?, description=?, features=?, added_date=?,
                address_fingerprint=?,
                last_seen=?, is_active=1, price_change_count=?,
                scraped_at=?
            WHERE id=?
        ''', (
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
        """Insert new listing with first_seen=last_seen=now."""
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
                let_agreed, agent_name, agent_phone,
                summary, description, features, added_date,
                address_fingerprint,
                first_seen, last_seen, is_active, price_change_count,
                scraped_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?
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
            1 if adapter.get('let_agreed') else 0,
            adapter.get('agent_name'), adapter.get('agent_phone'),
            adapter.get('summary'), adapter.get('description'),
            features, adapter.get('added_date'),
            adapter.get('address_fingerprint'),
            now, now,
            adapter.get('scraped_at'),
        ))

    def close_spider(self, spider):
        if self.conn:
            if self.pending_count > 0:
                self.conn.commit()

            elapsed = time.time() - self.start_time if self.start_time else 0

            self.cursor.execute('SELECT COUNT(*) FROM listings')
            total = self.cursor.fetchone()[0]

            logger.info("[PIPELINE:SQLite] Complete:")
            logger.info(f"  Inserted: {self.stats['inserted']}")
            logger.info(f"  Updated: {self.stats['updated']}")
            logger.info(f"  Price changes logged: {self.stats['price_changes']}")
            logger.info(f"  Errors: {self.stats['errors']}")
            logger.info(f"  Total records: {total}")
            logger.info(f"  Duration: {elapsed:.1f}s")

            self.conn.close()
```

## Testing Plan

1. **Unit test**: Mock database, verify insert vs update logic
2. **Integration test**:
   - Run spider on small set
   - Change price manually in source
   - Re-run spider
   - Verify price_history has new entry
   - Verify first_seen unchanged, last_seen updated

## Success Criteria

- [ ] Fingerprint generated for every item in CleanDataPipeline
- [ ] New listings: first_seen = last_seen = now
- [ ] Updated listings: first_seen preserved, last_seen updated
- [ ] Price changes logged to price_history table
- [ ] price_change_count incremented on price change
- [ ] is_active set to 1 on every update
- [ ] Backward compatible (old spiders work without changes)

## Implementation Steps

1. Add fingerprint generation to CleanDataPipeline
2. Refactor SQLitePipeline with new upsert logic
3. Test with dry run (print SQL instead of execute)
4. Test with single spider
5. Commit and push

## Dependencies
- PRD-001 (Schema migration must be run first)
- PRD-002 (Fingerprint service must exist)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Performance degradation from SELECT before INSERT | Acceptable for batch sizes <1000; index on (source, property_id) |
| Missing price_history baseline | PRD-001 already created baseline snapshot |
| Breaking existing spiders | Fingerprint fallback in SQLitePipeline if not in item |
