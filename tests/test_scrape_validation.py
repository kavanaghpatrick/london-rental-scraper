#!/usr/bin/env python3
"""
Full Scrape Validation Tests

This module provides tests to validate the scraper before and after running.
Run PRE-SCRAPE tests to ensure system is ready, then POST-SCRAPE to validate results.

Usage:
    # Pre-scrape validation (run BEFORE scraping)
    pytest tests/test_scrape_validation.py -v -k "pre_scrape"

    # Capture baseline snapshot
    python3 tests/test_scrape_validation.py --snapshot

    # Post-scrape validation (run AFTER scraping)
    pytest tests/test_scrape_validation.py -v -k "post_scrape"

    # Full validation suite
    pytest tests/test_scrape_validation.py -v
"""

import pytest
import sqlite3
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Database path
DB_PATH = Path(__file__).parent.parent / "output" / "rentals.db"
SNAPSHOT_PATH = Path(__file__).parent / "scrape_snapshot.json"


def get_db_stats() -> Dict[str, Any]:
    """Get current database statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    stats = {}

    # Total counts by source
    cursor.execute("""
        SELECT source, COUNT(*) as total,
               SUM(CASE WHEN size_sqft > 0 THEN 1 ELSE 0 END) as with_sqft,
               COUNT(DISTINCT address_fingerprint) as unique_fingerprints,
               SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active
        FROM listings
        GROUP BY source
    """)
    stats['by_source'] = {row[0]: {
        'total': row[1],
        'with_sqft': row[2],
        'unique_fingerprints': row[3],
        'active': row[4]
    } for row in cursor.fetchall()}

    # Overall totals
    cursor.execute("SELECT COUNT(*) FROM listings")
    stats['total_listings'] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM price_history")
    stats['price_history_count'] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT address_fingerprint) FROM listings WHERE address_fingerprint IS NOT NULL")
    stats['unique_fingerprints'] = cursor.fetchone()[0]

    # Date tracking
    cursor.execute("SELECT MIN(first_seen), MAX(last_seen) FROM listings")
    row = cursor.fetchone()
    stats['earliest_first_seen'] = row[0]
    stats['latest_last_seen'] = row[1]

    # Today's activity
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute(f"SELECT COUNT(*) FROM listings WHERE first_seen LIKE '{today}%'")
    stats['new_today'] = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) FROM listings WHERE last_seen LIKE '{today}%'")
    stats['updated_today'] = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) FROM price_history WHERE recorded_at LIKE '{today}%'")
    stats['price_changes_today'] = cursor.fetchone()[0]

    conn.close()
    return stats


def save_snapshot():
    """Save current database state as snapshot for comparison."""
    stats = get_db_stats()
    stats['snapshot_time'] = datetime.now().isoformat()

    with open(SNAPSHOT_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Snapshot saved to {SNAPSHOT_PATH}")
    print(f"Total listings: {stats['total_listings']}")
    print(f"Price history records: {stats['price_history_count']}")
    return stats


def load_snapshot() -> Dict[str, Any]:
    """Load previous snapshot for comparison."""
    if not SNAPSHOT_PATH.exists():
        pytest.skip("No snapshot found. Run with --snapshot first.")

    with open(SNAPSHOT_PATH) as f:
        return json.load(f)


# =============================================================================
# PRE-SCRAPE TESTS - Run before scraping to validate system readiness
# =============================================================================

class TestPreScrapeValidation:
    """Tests to run BEFORE scraping to ensure system is ready."""

    def test_pre_scrape_database_exists(self):
        """Database file should exist."""
        assert DB_PATH.exists(), f"Database not found at {DB_PATH}"

    def test_pre_scrape_database_readable(self):
        """Database should be readable."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM listings")
        count = cursor.fetchone()[0]
        conn.close()
        assert count > 0, "Database is empty"

    def test_pre_scrape_schema_has_required_columns(self):
        """Schema should have all required columns for historical tracking."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(listings)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        required = {'first_seen', 'last_seen', 'is_active', 'address_fingerprint', 'price_change_count'}
        missing = required - columns
        assert not missing, f"Missing columns: {missing}"

    def test_pre_scrape_price_history_table_exists(self):
        """price_history table should exist."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_history'")
        result = cursor.fetchone()
        conn.close()
        assert result is not None, "price_history table not found"

    def test_pre_scrape_fingerprints_populated(self):
        """Most listings should have address fingerprints."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN address_fingerprint IS NOT NULL THEN 1 ELSE 0 END) as with_fp
            FROM listings
        """)
        total, with_fp = cursor.fetchone()
        conn.close()

        pct = (with_fp / total * 100) if total else 0
        assert pct > 80, f"Only {pct:.1f}% of listings have fingerprints (need >80%)"

    def test_pre_scrape_first_seen_populated(self):
        """Most listings should have first_seen dates."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN first_seen IS NOT NULL THEN 1 ELSE 0 END) as with_date
            FROM listings
        """)
        total, with_date = cursor.fetchone()
        conn.close()

        pct = (with_date / total * 100) if total else 0
        assert pct > 80, f"Only {pct:.1f}% have first_seen (need >80%)"

    def test_pre_scrape_spiders_importable(self):
        """All spiders should be importable without errors."""
        from property_scraper.spiders import (
            savills_spider,
            knightfrank_spider,
            chestertons_spider,
            foxtons_spider,
            rightmove_spider,
        )
        # If we get here, imports succeeded
        assert True

    def test_pre_scrape_pipelines_importable(self):
        """Pipelines should be importable."""
        from property_scraper.pipelines import (
            CleanDataPipeline,
            DuplicateFilterPipeline,
            SQLitePipeline,
        )
        assert True

    def test_pre_scrape_cli_importable(self):
        """CLI should be importable."""
        from cli.main import app
        from cli.registry import SPIDERS
        assert len(SPIDERS) > 0


# =============================================================================
# POST-SCRAPE TESTS - Run after scraping to validate results
# =============================================================================

class TestPostScrapeValidation:
    """Tests to run AFTER scraping to validate results."""

    @pytest.fixture
    def snapshot(self):
        """Load the pre-scrape snapshot."""
        return load_snapshot()

    @pytest.fixture
    def current(self):
        """Get current database state."""
        return get_db_stats()

    def test_post_scrape_no_data_loss(self, snapshot, current):
        """Total listings should not decrease significantly."""
        before = snapshot['total_listings']
        after = current['total_listings']

        # Allow up to 5% decrease (some listings may be removed)
        min_expected = before * 0.95
        assert after >= min_expected, f"Data loss detected: {before} -> {after} ({after - before})"

    def test_post_scrape_last_seen_updated(self, current):
        """Some listings should have today's date in last_seen."""
        assert current['updated_today'] > 0, "No listings updated today - scrape may have failed"

    def test_post_scrape_first_seen_preserved(self, snapshot, current):
        """first_seen dates should not change for existing listings."""
        # If earliest_first_seen changed to today, old data was lost
        today = datetime.now().strftime('%Y-%m-%d')
        assert not current['earliest_first_seen'].startswith(today), \
            "earliest_first_seen is today - historical data may have been lost!"

    def test_post_scrape_fingerprints_generated(self, current):
        """New listings should have fingerprints."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute(f"""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN address_fingerprint IS NOT NULL THEN 1 ELSE 0 END) as with_fp
            FROM listings
            WHERE first_seen LIKE '{today}%'
        """)
        total, with_fp = cursor.fetchone()
        conn.close()

        if total > 0:
            pct = with_fp / total * 100
            assert pct > 90, f"Only {pct:.1f}% of new listings have fingerprints"

    def test_post_scrape_price_history_logged(self, snapshot, current):
        """Price history should have new entries if there were changes."""
        # This is informational - price changes are optional
        before = snapshot['price_history_count']
        after = current['price_history_count']

        if after > before:
            print(f"Price history: {before} -> {after} (+{after - before} changes)")

    def test_post_scrape_sqft_coverage_maintained(self, snapshot, current):
        """Sqft coverage should not decrease significantly."""
        for source in snapshot['by_source']:
            if source not in current['by_source']:
                continue

            before_pct = snapshot['by_source'][source]['with_sqft'] / max(1, snapshot['by_source'][source]['total'])
            after_pct = current['by_source'][source]['with_sqft'] / max(1, current['by_source'][source]['total'])

            # Allow 10% decrease in coverage
            assert after_pct >= before_pct * 0.9, \
                f"{source}: sqft coverage dropped from {before_pct*100:.1f}% to {after_pct*100:.1f}%"


# =============================================================================
# DATA INTEGRITY TESTS - Can run anytime
# =============================================================================

class TestDataIntegrity:
    """Tests for general data integrity."""

    def test_integrity_no_null_sources(self):
        """No listings should have NULL source."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM listings WHERE source IS NULL")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0, f"{count} listings have NULL source"

    def test_integrity_no_null_property_ids(self):
        """No listings should have NULL property_id."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM listings WHERE property_id IS NULL")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0, f"{count} listings have NULL property_id"

    def test_integrity_unique_constraint(self):
        """source + property_id should be unique."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT source, property_id, COUNT(*) as cnt
            FROM listings
            GROUP BY source, property_id
            HAVING cnt > 1
        """)
        duplicates = cursor.fetchall()
        conn.close()
        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate source+property_id combinations"

    def test_integrity_valid_prices(self):
        """Prices should be reasonable (>0, <500000 for ultra-luxury)."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM listings
            WHERE price_pcm IS NOT NULL
            AND (price_pcm <= 0 OR price_pcm > 500000)
        """)
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0, f"{count} listings have invalid prices"

    def test_integrity_valid_sqft(self):
        """Sqft should be reasonable (100-50000)."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM listings
            WHERE size_sqft IS NOT NULL AND size_sqft > 0
            AND (size_sqft < 100 OR size_sqft > 50000)
        """)
        count = cursor.fetchone()[0]
        conn.close()
        # Allow some outliers
        assert count < 50, f"{count} listings have suspicious sqft values"

    def test_integrity_fingerprint_format(self):
        """Fingerprints should be 16 characters."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM listings
            WHERE address_fingerprint IS NOT NULL
            AND LENGTH(address_fingerprint) != 16
        """)
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0, f"{count} listings have invalid fingerprint length"

    def test_integrity_dates_chronological(self):
        """first_seen should be <= last_seen."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM listings
            WHERE first_seen IS NOT NULL AND last_seen IS NOT NULL
            AND first_seen > last_seen
        """)
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0, f"{count} listings have first_seen > last_seen"


# =============================================================================
# COMPARISON REPORT - Shows what changed
# =============================================================================

def generate_comparison_report():
    """Generate a detailed comparison report."""
    if not SNAPSHOT_PATH.exists():
        print("No snapshot found. Run with --snapshot first.")
        return

    snapshot = load_snapshot()
    current = get_db_stats()

    print("\n" + "="*70)
    print("SCRAPE COMPARISON REPORT")
    print("="*70)
    print(f"Snapshot time: {snapshot['snapshot_time']}")
    print(f"Current time:  {datetime.now().isoformat()}")
    print()

    # Overall changes
    print("OVERALL CHANGES:")
    print("-"*40)
    total_diff = current['total_listings'] - snapshot['total_listings']
    print(f"  Total listings: {snapshot['total_listings']} -> {current['total_listings']} ({'+' if total_diff >= 0 else ''}{total_diff})")

    ph_diff = current['price_history_count'] - snapshot['price_history_count']
    print(f"  Price history:  {snapshot['price_history_count']} -> {current['price_history_count']} ({'+' if ph_diff >= 0 else ''}{ph_diff})")

    print(f"  New today:      {current['new_today']}")
    print(f"  Updated today:  {current['updated_today']}")
    print(f"  Price changes:  {current['price_changes_today']}")
    print()

    # Per-source breakdown
    print("BY SOURCE:")
    print("-"*40)
    all_sources = set(snapshot['by_source'].keys()) | set(current['by_source'].keys())

    for source in sorted(all_sources):
        before = snapshot['by_source'].get(source, {'total': 0, 'with_sqft': 0})
        after = current['by_source'].get(source, {'total': 0, 'with_sqft': 0})

        diff = after['total'] - before['total']
        sqft_diff = after['with_sqft'] - before['with_sqft']

        print(f"  {source:15} {before['total']:5} -> {after['total']:5} ({'+' if diff >= 0 else ''}{diff:4})  "
              f"sqft: {before['with_sqft']:4} -> {after['with_sqft']:4} ({'+' if sqft_diff >= 0 else ''}{sqft_diff:3})")

    print()
    print("="*70)


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape validation tools")
    parser.add_argument("--snapshot", action="store_true", help="Save current state as snapshot")
    parser.add_argument("--report", action="store_true", help="Generate comparison report")
    parser.add_argument("--stats", action="store_true", help="Show current database stats")

    args = parser.parse_args()

    if args.snapshot:
        save_snapshot()
    elif args.report:
        generate_comparison_report()
    elif args.stats:
        stats = get_db_stats()
        print(json.dumps(stats, indent=2))
    else:
        # Default: run pytest
        pytest.main([__file__, "-v"])
