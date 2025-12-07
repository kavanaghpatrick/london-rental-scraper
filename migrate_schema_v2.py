#!/usr/bin/env python3
"""
Schema migration v2: Add fingerprinting and price history support.

Changes:
- Add address_fingerprint column for cross-source deduplication
- Add first_seen, last_seen, is_active, price_change_count columns
- Create price_history table for tracking price changes
- Backfill temporal columns from scraped_at
- Snapshot current prices as initial history

Usage:
    python3 migrate_schema_v2.py              # Dry run
    python3 migrate_schema_v2.py --execute    # Apply changes
    python3 migrate_schema_v2.py --rollback   # Undo migration

Addresses Gemini review feedback:
- Uses sqlite3.backup() for safe backup (not shutil.copy2)
- Smart is_active backfill based on recency (7 days)
- Baseline price_history snapshot
- No redundant columns in price_history
"""

import sqlite3
import argparse
from datetime import datetime
from pathlib import Path

DB_PATH = Path("output/rentals.db")
BACKUP_PATH = DB_PATH.with_suffix(".db.pre_v2_backup")


def check_column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def check_table_exists(cursor: sqlite3.Cursor, table: str) -> bool:
    """Check if a table exists in the database."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    )
    return cursor.fetchone() is not None


def check_index_exists(cursor: sqlite3.Cursor, index_name: str) -> bool:
    """Check if an index exists in the database."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,)
    )
    return cursor.fetchone() is not None


def create_backup(db_path: Path, backup_path: Path) -> bool:
    """Create atomic backup using SQLite backup API."""
    try:
        source = sqlite3.connect(db_path)
        dest = sqlite3.connect(backup_path)
        source.backup(dest)
        dest.close()
        source.close()
        return True
    except Exception as e:
        print(f"Backup failed: {e}")
        return False


def migrate(execute: bool = False) -> bool:
    """Run the schema migration."""
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return False

    if execute:
        print(f"Creating backup using SQLite backup API...")
        if not create_backup(DB_PATH, BACKUP_PATH):
            print("Aborting migration due to backup failure.")
            return False
        print(f"Backup created: {BACKUP_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")  # Enable FK enforcement
    cursor = conn.cursor()

    changes = []
    skipped = []

    # New columns for listings table
    new_columns = [
        ("address_fingerprint", "TEXT"),
        ("first_seen", "TEXT"),
        ("last_seen", "TEXT"),
        ("is_active", "INTEGER DEFAULT 1"),
        ("price_change_count", "INTEGER DEFAULT 0"),
    ]

    for col_name, col_type in new_columns:
        if check_column_exists(cursor, "listings", col_name):
            skipped.append(f"Column listings.{col_name} already exists")
        else:
            changes.append(("ADD COLUMN", f"ALTER TABLE listings ADD COLUMN {col_name} {col_type}"))

    # Indexes
    indexes = [
        ("idx_fingerprint", "listings(address_fingerprint)"),
        ("idx_last_seen", "listings(last_seen)"),
        ("idx_active", "listings(is_active)"),
        ("idx_source_fingerprint", "listings(source, address_fingerprint)"),
    ]

    for idx_name, idx_def in indexes:
        if check_index_exists(cursor, idx_name):
            skipped.append(f"Index {idx_name} already exists")
        else:
            changes.append(("CREATE INDEX", f"CREATE INDEX {idx_name} ON {idx_def}"))

    # Price history table
    if check_table_exists(cursor, "price_history"):
        skipped.append("Table price_history already exists")
    else:
        changes.append(("CREATE TABLE", """
CREATE TABLE price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id INTEGER NOT NULL,
    price_pcm INTEGER NOT NULL,
    price_pw INTEGER,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (listing_id) REFERENCES listings(id)
)"""))
        changes.append(("CREATE INDEX", "CREATE INDEX idx_ph_listing ON price_history(listing_id)"))
        changes.append(("CREATE INDEX", "CREATE INDEX idx_ph_recorded ON price_history(recorded_at)"))

    # Backfill first_seen/last_seen from scraped_at
    # Smart is_active: only mark active if scraped within last 7 days
    changes.append(("BACKFILL", """
UPDATE listings
SET first_seen = scraped_at,
    last_seen = scraped_at,
    is_active = CASE
        WHEN scraped_at >= date('now', '-7 days') THEN 1
        ELSE 0
    END
WHERE first_seen IS NULL AND scraped_at IS NOT NULL
"""))

    # Baseline price history snapshot (only if table was just created)
    if not check_table_exists(cursor, "price_history"):
        changes.append(("BASELINE", """
INSERT INTO price_history (listing_id, price_pcm, price_pw, recorded_at)
SELECT id, price_pcm, price_pw, COALESCE(scraped_at, datetime('now'))
FROM listings
WHERE price_pcm IS NOT NULL
"""))

    # Print plan
    print(f"\n{'='*60}")
    print(f"Schema Migration v2 - {'DRY RUN' if not execute else 'EXECUTING'}")
    print(f"{'='*60}\n")

    if skipped:
        print("SKIPPED (already exist):")
        for s in skipped:
            print(f"  - {s}")
        print()

    print("CHANGES TO APPLY:")
    for i, (change_type, sql) in enumerate(changes, 1):
        # Truncate long SQL for display
        display_sql = sql.strip().replace('\n', ' ')[:70]
        print(f"  {i}. [{change_type}] {display_sql}...")
    print()

    if execute:
        print("Executing changes...")
        for change_type, sql in changes:
            try:
                cursor.execute(sql)
                print(f"  [{change_type}] OK")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                    print(f"  [{change_type}] SKIPPED (already exists)")
                else:
                    print(f"  [{change_type}] ERROR: {e}")
                    raise

        conn.commit()
        print(f"\nMigration complete! {len(changes)} operations executed.")
    else:
        print(f"Dry run complete. {len(changes)} operations would be applied.")
        print("Run with --execute to apply changes.")

    # Verification (only query existing columns)
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}\n")

    cursor.execute("SELECT COUNT(*) FROM listings")
    total = cursor.fetchone()[0]
    print(f"Total listings: {total}")

    if check_column_exists(cursor, "listings", "first_seen"):
        cursor.execute("SELECT COUNT(*) FROM listings WHERE first_seen IS NOT NULL")
        with_temporal = cursor.fetchone()[0]
        print(f"With temporal data (first_seen): {with_temporal}")
    else:
        print(f"With temporal data (first_seen): N/A (column pending)")

    if check_column_exists(cursor, "listings", "is_active"):
        cursor.execute("SELECT COUNT(*) FROM listings WHERE is_active = 1")
        active = cursor.fetchone()[0]
        print(f"Marked as active: {active}")
    else:
        print(f"Marked as active: N/A (column pending)")

    if check_column_exists(cursor, "listings", "address_fingerprint"):
        cursor.execute("SELECT COUNT(*) FROM listings WHERE address_fingerprint IS NOT NULL")
        with_fp = cursor.fetchone()[0]
        print(f"With fingerprints (expect 0, PRD-002): {with_fp}")
    else:
        print(f"With fingerprints: N/A (column pending)")

    if check_table_exists(cursor, "price_history"):
        cursor.execute("SELECT COUNT(*) FROM price_history")
        history_count = cursor.fetchone()[0]
        print(f"Price history records: {history_count}")
    else:
        print(f"Price history records: N/A (table pending)")

    conn.close()
    return True


def rollback() -> bool:
    """Restore database from backup."""
    if not BACKUP_PATH.exists():
        print(f"No backup found at {BACKUP_PATH}")
        return False

    print(f"Restoring from backup: {BACKUP_PATH}")
    try:
        source = sqlite3.connect(BACKUP_PATH)
        dest = sqlite3.connect(DB_PATH)
        source.backup(dest)
        dest.close()
        source.close()
        print("Restore complete.")
        return True
    except Exception as e:
        print(f"Restore failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Schema migration v2: Add fingerprinting and price history support."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply migration (default is dry run)"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Restore database from pre-migration backup"
    )
    args = parser.parse_args()

    if args.rollback:
        success = rollback()
    else:
        success = migrate(execute=args.execute)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
