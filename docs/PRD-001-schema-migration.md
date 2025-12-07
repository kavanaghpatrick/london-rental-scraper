# PRD-001: Database Schema Migration

## Overview
Extend the existing SQLite schema to support address fingerprinting for cross-source deduplication, historical price tracking, and listing lifecycle management.

## Problem Statement
Current schema limitations:
1. No mechanism for cross-source duplicate detection at insert time
2. `INSERT OR REPLACE` overwrites existing records, losing history
3. No tracking of when listings were first seen, last updated, or became inactive
4. Cannot track price changes over time for market analysis

## Goals
1. Add `address_fingerprint` column for cross-source deduplication
2. Add temporal tracking columns (`first_seen`, `last_seen`, `is_active`)
3. Create separate `price_history` table for change tracking
4. Preserve all existing data during migration
5. Add indexes for efficient lookups

## Non-Goals
- Changing the ORM layer (staying with raw SQLite)
- Migrating to PostgreSQL or other database
- Restructuring to Property/Listing star schema (future PRD)

## Technical Design

### Schema Changes

#### 1. Alter `listings` table
```sql
-- New columns for deduplication and tracking
ALTER TABLE listings ADD COLUMN address_fingerprint TEXT;
ALTER TABLE listings ADD COLUMN first_seen TEXT;
ALTER TABLE listings ADD COLUMN last_seen TEXT;
ALTER TABLE listings ADD COLUMN is_active INTEGER DEFAULT 1;
ALTER TABLE listings ADD COLUMN price_change_count INTEGER DEFAULT 0;

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_fingerprint ON listings(address_fingerprint);
CREATE INDEX IF NOT EXISTS idx_last_seen ON listings(last_seen);
CREATE INDEX IF NOT EXISTS idx_active ON listings(is_active);
CREATE INDEX IF NOT EXISTS idx_source_fingerprint ON listings(source, address_fingerprint);
```

#### 2. Create `price_history` table
```sql
-- Note: source/property_id NOT included - use JOIN to listings for those
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id INTEGER NOT NULL,
    price_pcm INTEGER NOT NULL,
    price_pw INTEGER,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (listing_id) REFERENCES listings(id)
);

CREATE INDEX IF NOT EXISTS idx_ph_listing ON price_history(listing_id);
CREATE INDEX IF NOT EXISTS idx_ph_recorded ON price_history(recorded_at);

-- Baseline: Snapshot current prices as initial history
INSERT INTO price_history (listing_id, price_pcm, price_pw, recorded_at)
SELECT id, price_pcm, price_pw, COALESCE(scraped_at, datetime('now'))
FROM listings
WHERE price_pcm IS NOT NULL;
```

### Migration Script: `migrate_schema_v2.py`

```python
#!/usr/bin/env python3
"""
Schema migration v2: Add fingerprinting and price history support.

Usage:
    python3 migrate_schema_v2.py              # Dry run
    python3 migrate_schema_v2.py --execute    # Apply changes
    python3 migrate_schema_v2.py --rollback   # Undo migration
"""

import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
import argparse

DB_PATH = Path("output/rentals.db")
BACKUP_PATH = DB_PATH.with_suffix(".db.pre_v2_backup")

def check_column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns

def check_table_exists(cursor, table: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    )
    return cursor.fetchone() is not None

def migrate(execute: bool = False):
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return False

    if execute:
        # Create backup before migration
        shutil.copy2(DB_PATH, BACKUP_PATH)
        print(f"Backup created: {BACKUP_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    changes = []

    # Check and add new columns
    new_columns = [
        ("address_fingerprint", "TEXT"),
        ("first_seen", "TEXT"),
        ("last_seen", "TEXT"),
        ("is_active", "INTEGER DEFAULT 1"),
        ("price_change_count", "INTEGER DEFAULT 0"),
    ]

    for col_name, col_type in new_columns:
        if not check_column_exists(cursor, "listings", col_name):
            changes.append(f"ALTER TABLE listings ADD COLUMN {col_name} {col_type}")

    # Indexes
    indexes = [
        ("idx_fingerprint", "listings(address_fingerprint)"),
        ("idx_last_seen", "listings(last_seen)"),
        ("idx_active", "listings(is_active)"),
        ("idx_source_fingerprint", "listings(source, address_fingerprint)"),
    ]

    for idx_name, idx_def in indexes:
        changes.append(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_def}")

    # Price history table
    if not check_table_exists(cursor, "price_history"):
        changes.append("""
CREATE TABLE price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    property_id TEXT NOT NULL,
    price_pcm INTEGER NOT NULL,
    price_pw INTEGER,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (listing_id) REFERENCES listings(id)
)""")
        changes.append("CREATE INDEX idx_ph_listing ON price_history(listing_id)")
        changes.append("CREATE INDEX idx_ph_recorded ON price_history(recorded_at)")

    # Backfill first_seen/last_seen from scraped_at
    changes.append("""
UPDATE listings
SET first_seen = scraped_at,
    last_seen = scraped_at,
    is_active = 1
WHERE first_seen IS NULL AND scraped_at IS NOT NULL
""")

    print(f"\n{'='*60}")
    print(f"Schema Migration v2 - {'DRY RUN' if not execute else 'EXECUTING'}")
    print(f"{'='*60}\n")

    for i, sql in enumerate(changes, 1):
        print(f"{i}. {sql.strip()[:80]}...")
        if execute:
            try:
                cursor.execute(sql)
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    print(f"   (skipped - already exists)")
                else:
                    raise

    if execute:
        conn.commit()
        print(f"\n Migration complete! {len(changes)} changes applied.")
    else:
        print(f"\n Dry run complete. {len(changes)} changes would be applied.")
        print("Run with --execute to apply changes.")

    # Verify
    cursor.execute("SELECT COUNT(*) FROM listings")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM listings WHERE address_fingerprint IS NOT NULL")
    with_fp = cursor.fetchone()[0]
    print(f"\nCurrent state: {total} listings, {with_fp} with fingerprints")

    conn.close()
    return True

def rollback():
    if not BACKUP_PATH.exists():
        print(f"No backup found at {BACKUP_PATH}")
        return False

    shutil.copy2(BACKUP_PATH, DB_PATH)
    print(f"Restored from backup: {BACKUP_PATH}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Apply migration")
    parser.add_argument("--rollback", action="store_true", help="Restore from backup")
    args = parser.parse_args()

    if args.rollback:
        rollback()
    else:
        migrate(execute=args.execute)
```

## Testing Plan

1. **Unit tests** (`tests/test_schema_migration.py`):
   - Test column existence checks
   - Test table existence checks
   - Test idempotent migration (running twice doesn't error)

2. **Integration tests**:
   - Create test database with sample data
   - Run migration
   - Verify all columns exist
   - Verify indexes exist
   - Verify data preserved
   - Verify backfill worked

3. **Rollback test**:
   - Run migration
   - Verify backup created
   - Run rollback
   - Verify original state restored

## Success Criteria

- [ ] All new columns added without data loss
- [ ] `price_history` table created with proper foreign key
- [ ] Indexes created for query performance
- [ ] `first_seen`/`last_seen` backfilled from `scraped_at`
- [ ] Backup created before migration
- [ ] Rollback works correctly
- [ ] Migration is idempotent (can run multiple times safely)

## Implementation Steps

1. Create `docs/` directory if not exists
2. Create `migrate_schema_v2.py` in project root
3. Create `tests/test_schema_migration.py`
4. Run dry-run and verify output
5. Run with `--execute` on copy of database
6. Verify all criteria met
7. Commit migration script

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Data loss during migration | Automatic backup before execution |
| Disk space for backup | ~50MB database, backup manageable |
| Migration fails mid-way | SQLite transactions ensure atomicity |
| Old code breaks with new schema | New columns have defaults, backward compatible |

## Timeline
Single session implementation. No multi-week estimates.

## Dependencies
- None (uses standard library only)

## Future Work
- PRD-002 will implement the fingerprinting service
- PRD-003 will modify pipelines to use new columns
