#!/usr/bin/env python3
"""Tests for pipeline modifications (PRD-003).

Tests:
1. CleanDataPipeline generates fingerprints
2. SQLitePipeline smart upsert logic
3. Price history logging
4. first_seen/last_seen preservation
"""

import pytest
import sqlite3
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from property_scraper.pipelines import CleanDataPipeline, SQLitePipeline
from property_scraper.items import PropertyItem


def create_test_db(db_path):
    """Create a test database with required schema from PRD-001."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create listings table with all required columns
    cursor.execute('''
        CREATE TABLE listings (
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
            has_basement INTEGER,
            has_lower_ground INTEGER,
            has_ground INTEGER,
            has_mezzanine INTEGER,
            has_first_floor INTEGER,
            has_second_floor INTEGER,
            has_third_floor INTEGER,
            has_fourth_plus INTEGER,
            has_roof_terrace INTEGER,
            floor_count INTEGER,
            property_levels TEXT,
            let_agreed INTEGER DEFAULT 0,
            agent_name TEXT,
            agent_phone TEXT,
            summary TEXT,
            description TEXT,
            features TEXT,
            added_date TEXT,
            scraped_at TEXT,
            address_fingerprint TEXT,
            first_seen TEXT,
            last_seen TEXT,
            is_active INTEGER DEFAULT 1,
            price_change_count INTEGER DEFAULT 0,
            UNIQUE(source, property_id)
        )
    ''')

    # Create price_history table
    cursor.execute('''
        CREATE TABLE price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id INTEGER NOT NULL,
            price_pcm INTEGER,
            recorded_at TEXT NOT NULL,
            FOREIGN KEY (listing_id) REFERENCES listings(id)
        )
    ''')

    conn.commit()
    return conn


class TestCleanDataPipelineFingerprintGeneration:
    """Test fingerprint generation in CleanDataPipeline."""

    def test_generates_fingerprint_from_address_and_postcode(self):
        pipeline = CleanDataPipeline()
        pipeline._generate_fingerprint = lambda a, p: "test_fingerprint_123"

        item = PropertyItem()
        item['property_id'] = 'test123'
        item['source'] = 'test'
        item['address'] = '123 High Street'
        item['postcode'] = 'SW1A 1AA'

        result = pipeline.process_item(item, MagicMock())

        assert result['address_fingerprint'] == 'test_fingerprint_123'
        assert pipeline.fingerprints_generated == 1

    def test_does_not_overwrite_existing_fingerprint(self):
        pipeline = CleanDataPipeline()
        pipeline._generate_fingerprint = lambda a, p: "should_not_use"

        item = PropertyItem()
        item['property_id'] = 'test123'
        item['source'] = 'test'
        item['address'] = '123 High Street'
        item['postcode'] = 'SW1A 1AA'
        item['address_fingerprint'] = 'existing_fingerprint'

        result = pipeline.process_item(item, MagicMock())

        assert result['address_fingerprint'] == 'existing_fingerprint'
        assert pipeline.fingerprints_generated == 0

    def test_handles_missing_address(self):
        pipeline = CleanDataPipeline()
        pipeline._generate_fingerprint = lambda a, p: "fp_from_postcode"

        item = PropertyItem()
        item['property_id'] = 'test123'
        item['source'] = 'test'
        item['postcode'] = 'SW1A 1AA'

        result = pipeline.process_item(item, MagicMock())

        # Should still generate from postcode alone
        assert result['address_fingerprint'] == 'fp_from_postcode'


class TestSQLitePipelineSmartUpsert:
    """Test smart upsert logic in SQLitePipeline."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a temporary test database."""
        db_path = tmp_path / "test_rentals.db"
        conn = create_test_db(str(db_path))
        conn.close()
        return str(db_path)

    @pytest.fixture
    def pipeline(self, test_db, tmp_path):
        """Create a pipeline instance with test database."""
        pipeline = SQLitePipeline()
        pipeline.db_path = test_db
        pipeline.conn = sqlite3.connect(test_db)
        pipeline.conn.execute("PRAGMA foreign_keys = ON")
        pipeline.cursor = pipeline.conn.cursor()
        pipeline.start_time = datetime.now().timestamp()
        return pipeline

    def test_insert_new_listing_sets_first_seen_and_last_seen(self, pipeline):
        """New listings should have first_seen = last_seen = now."""
        item = PropertyItem()
        item['source'] = 'test'
        item['property_id'] = 'new123'
        item['address'] = '123 High Street'
        item['postcode'] = 'SW1A 1AA'
        item['price_pcm'] = 2000
        item['address_fingerprint'] = 'fp123'

        pipeline.process_item(item, MagicMock())
        pipeline.conn.commit()

        pipeline.cursor.execute(
            "SELECT first_seen, last_seen, is_active FROM listings WHERE property_id = 'new123'"
        )
        row = pipeline.cursor.fetchone()

        assert row is not None
        assert row[0] is not None  # first_seen set
        assert row[1] is not None  # last_seen set
        assert row[0] == row[1]    # first_seen == last_seen for new
        assert row[2] == 1         # is_active = 1

    def test_insert_logs_initial_price_to_history(self, pipeline):
        """New listings should log initial price to price_history."""
        item = PropertyItem()
        item['source'] = 'test'
        item['property_id'] = 'new456'
        item['price_pcm'] = 3000
        item['address_fingerprint'] = 'fp456'

        pipeline.process_item(item, MagicMock())
        pipeline.conn.commit()

        # Get listing id
        pipeline.cursor.execute(
            "SELECT id FROM listings WHERE property_id = 'new456'"
        )
        listing_id = pipeline.cursor.fetchone()[0]

        # Check price history
        pipeline.cursor.execute(
            "SELECT price_pcm FROM price_history WHERE listing_id = ?",
            (listing_id,)
        )
        history = pipeline.cursor.fetchall()

        assert len(history) == 1
        assert history[0][0] == 3000

    def test_update_preserves_first_seen(self, pipeline):
        """Updates should preserve original first_seen."""
        # Insert initial listing
        original_time = "2025-01-01T00:00:00"
        pipeline.cursor.execute('''
            INSERT INTO listings (source, property_id, price_pcm, first_seen, last_seen, is_active)
            VALUES ('test', 'existing123', 2000, ?, ?, 1)
        ''', (original_time, original_time))
        pipeline.conn.commit()

        # Update via pipeline
        item = PropertyItem()
        item['source'] = 'test'
        item['property_id'] = 'existing123'
        item['price_pcm'] = 2000  # Same price
        item['address_fingerprint'] = 'fp_existing'

        pipeline.process_item(item, MagicMock())
        pipeline.conn.commit()

        pipeline.cursor.execute(
            "SELECT first_seen FROM listings WHERE property_id = 'existing123'"
        )
        first_seen = pipeline.cursor.fetchone()[0]

        assert first_seen == original_time  # Preserved!

    def test_update_changes_last_seen(self, pipeline):
        """Updates should change last_seen to now."""
        original_time = "2025-01-01T00:00:00"
        pipeline.cursor.execute('''
            INSERT INTO listings (source, property_id, price_pcm, first_seen, last_seen, is_active)
            VALUES ('test', 'existing456', 2000, ?, ?, 1)
        ''', (original_time, original_time))
        pipeline.conn.commit()

        item = PropertyItem()
        item['source'] = 'test'
        item['property_id'] = 'existing456'
        item['price_pcm'] = 2000
        item['address_fingerprint'] = 'fp_existing'

        pipeline.process_item(item, MagicMock())
        pipeline.conn.commit()

        pipeline.cursor.execute(
            "SELECT last_seen FROM listings WHERE property_id = 'existing456'"
        )
        last_seen = pipeline.cursor.fetchone()[0]

        assert last_seen != original_time  # Changed!

    def test_price_change_logs_to_history(self, pipeline):
        """Price changes should be logged to price_history."""
        # Insert initial listing
        pipeline.cursor.execute('''
            INSERT INTO listings (source, property_id, price_pcm, first_seen, last_seen, is_active, price_change_count)
            VALUES ('test', 'price_test', 2000, '2025-01-01', '2025-01-01', 1, 0)
        ''')
        pipeline.conn.commit()

        # Update with different price
        item = PropertyItem()
        item['source'] = 'test'
        item['property_id'] = 'price_test'
        item['price_pcm'] = 2500  # Changed!
        item['address_fingerprint'] = 'fp_price'

        pipeline.process_item(item, MagicMock())
        pipeline.conn.commit()

        # Check price history
        pipeline.cursor.execute(
            "SELECT id FROM listings WHERE property_id = 'price_test'"
        )
        listing_id = pipeline.cursor.fetchone()[0]

        pipeline.cursor.execute(
            "SELECT price_pcm FROM price_history WHERE listing_id = ?",
            (listing_id,)
        )
        history = pipeline.cursor.fetchall()

        assert len(history) == 1
        assert history[0][0] == 2500

        # Check price_change_count
        pipeline.cursor.execute(
            "SELECT price_change_count FROM listings WHERE property_id = 'price_test'"
        )
        count = pipeline.cursor.fetchone()[0]
        assert count == 1

    def test_same_price_no_history_entry(self, pipeline):
        """Same price should not log to history."""
        pipeline.cursor.execute('''
            INSERT INTO listings (source, property_id, price_pcm, first_seen, last_seen, is_active, price_change_count)
            VALUES ('test', 'same_price', 2000, '2025-01-01', '2025-01-01', 1, 0)
        ''')
        pipeline.conn.commit()

        item = PropertyItem()
        item['source'] = 'test'
        item['property_id'] = 'same_price'
        item['price_pcm'] = 2000  # Same price
        item['address_fingerprint'] = 'fp_same'

        pipeline.process_item(item, MagicMock())
        pipeline.conn.commit()

        pipeline.cursor.execute(
            "SELECT id FROM listings WHERE property_id = 'same_price'"
        )
        listing_id = pipeline.cursor.fetchone()[0]

        pipeline.cursor.execute(
            "SELECT COUNT(*) FROM price_history WHERE listing_id = ?",
            (listing_id,)
        )
        count = pipeline.cursor.fetchone()[0]

        assert count == 0  # No history entry for same price

    def test_stats_tracking(self, pipeline):
        """Stats should correctly track inserts, updates, and price changes."""
        # Insert new
        item1 = PropertyItem()
        item1['source'] = 'test'
        item1['property_id'] = 'stats_new'
        item1['price_pcm'] = 1000
        item1['address_fingerprint'] = 'fp1'
        pipeline.process_item(item1, MagicMock())

        assert pipeline.stats['inserted'] == 1
        assert pipeline.stats['updated'] == 0

        pipeline.conn.commit()

        # Update existing (same price)
        item2 = PropertyItem()
        item2['source'] = 'test'
        item2['property_id'] = 'stats_new'
        item2['price_pcm'] = 1000
        item2['address_fingerprint'] = 'fp1'
        pipeline.process_item(item2, MagicMock())

        assert pipeline.stats['inserted'] == 1
        assert pipeline.stats['updated'] == 1
        assert pipeline.stats['price_changes'] == 0

        # Update with price change
        item3 = PropertyItem()
        item3['source'] = 'test'
        item3['property_id'] = 'stats_new'
        item3['price_pcm'] = 1500
        item3['address_fingerprint'] = 'fp1'
        pipeline.process_item(item3, MagicMock())

        assert pipeline.stats['inserted'] == 1
        assert pipeline.stats['updated'] == 2
        assert pipeline.stats['price_changes'] == 1


class TestSQLitePipelineErrorHandling:
    """Test SAVEPOINT-based error handling."""

    @pytest.fixture
    def test_db(self, tmp_path):
        db_path = tmp_path / "test_rentals.db"
        conn = create_test_db(str(db_path))
        conn.close()
        return str(db_path)

    @pytest.fixture
    def pipeline(self, test_db):
        pipeline = SQLitePipeline()
        pipeline.db_path = test_db
        pipeline.conn = sqlite3.connect(test_db)
        pipeline.conn.execute("PRAGMA foreign_keys = ON")
        pipeline.cursor = pipeline.conn.cursor()
        pipeline.start_time = datetime.now().timestamp()
        return pipeline

    def test_error_does_not_affect_subsequent_items(self, pipeline):
        """Errors on one item should not prevent subsequent items."""
        # This item will fail (we'll induce an error by making source NULL which violates NOT NULL)
        item1 = PropertyItem()
        item1['source'] = None  # Will cause error
        item1['property_id'] = 'error_item'
        item1['price_pcm'] = 1000
        item1['address_fingerprint'] = 'fp_error'

        # Process should not raise
        pipeline.process_item(item1, MagicMock())

        # Stats should show error
        assert pipeline.stats['errors'] == 1

        # This item should succeed
        item2 = PropertyItem()
        item2['source'] = 'test'
        item2['property_id'] = 'success_item'
        item2['price_pcm'] = 2000
        item2['address_fingerprint'] = 'fp_success'

        pipeline.process_item(item2, MagicMock())
        pipeline.conn.commit()

        # Verify success item was inserted
        pipeline.cursor.execute(
            "SELECT COUNT(*) FROM listings WHERE property_id = 'success_item'"
        )
        count = pipeline.cursor.fetchone()[0]
        assert count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
