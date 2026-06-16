#!/usr/bin/env python3
"""Pipeline upsert edge cases the scraper layer depends on.

Complements tests/test_pipeline.py (insert/update/price-history) and
tests/test_fingerprint.py (fingerprint *generation*). Here we test the
SQLitePipeline UPSERT *matching* logic:

  - fingerprint-fallback relist: same property reappears with a NEW property_id
    but the same address_fingerprint + bedrooms + a price within tolerance ->
    must UPDATE the existing row (and adopt the new property_id), NOT insert a dup.
  - price-tolerance guard: a fingerprint match with a price OUTSIDE tolerance
    (different unit in same building) must NOT merge -> separate row.

Cross-source sqft merge is covered by tests/test_data_layer_safety.py (data-layer).
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from property_scraper.pipelines import SQLitePipeline
from property_scraper.items import PropertyItem
from tests.test_pipeline import create_test_db


@pytest.fixture
def pipeline(tmp_path):
    db_path = tmp_path / "test_rentals.db"
    conn = create_test_db(str(db_path))
    conn.close()
    p = SQLitePipeline()
    p.db_path = str(db_path)
    p.conn = sqlite3.connect(str(db_path))
    p.conn.execute("PRAGMA foreign_keys = ON")
    p.cursor = p.conn.cursor()
    p.start_time = datetime.now().timestamp()
    return p


def _item(**kw):
    it = PropertyItem()
    it["source"] = kw.get("source", "rightmove")
    it["property_id"] = kw["property_id"]
    it["price_pcm"] = kw["price_pcm"]
    it["bedrooms"] = kw.get("bedrooms", 2)
    it["address_fingerprint"] = kw["fingerprint"]
    it["address"] = kw.get("address", "10 Test Street, SW1 1AA")
    it["postcode"] = kw.get("postcode", "SW1 1AA")
    return it


class TestFingerprintFallbackRelist:
    """A relisted property (new ID, same flat) should match by fingerprint."""

    def test_relist_with_new_id_updates_not_inserts(self, pipeline):
        # Original listing
        pipeline.process_item(
            _item(property_id="OLD123", price_pcm=3000, fingerprint="fp_relist"),
            MagicMock(),
        )
        pipeline.conn.commit()

        # Same flat relisted with a NEW property_id, same fingerprint+beds,
        # price within 20% tolerance -> should UPDATE the existing row.
        pipeline.process_item(
            _item(property_id="NEW999", price_pcm=3100, fingerprint="fp_relist"),
            MagicMock(),
        )
        pipeline.conn.commit()

        pipeline.cursor.execute(
            "SELECT COUNT(*) FROM listings WHERE address_fingerprint='fp_relist'"
        )
        assert pipeline.cursor.fetchone()[0] == 1, "relist created a duplicate row"

        # property_id should have been updated to the new one
        pipeline.cursor.execute(
            "SELECT property_id FROM listings WHERE address_fingerprint='fp_relist'"
        )
        assert pipeline.cursor.fetchone()[0] == "NEW999"

        assert pipeline.stats["inserted"] == 1
        assert pipeline.stats["updated"] == 1

    def test_fingerprint_match_rejected_when_price_outside_tolerance(self, pipeline):
        """Same building/fingerprint but very different price = different unit -> no merge."""
        pipeline.process_item(
            _item(property_id="UNIT_A", price_pcm=2000, fingerprint="fp_building"),
            MagicMock(),
        )
        pipeline.conn.commit()

        # Same fingerprint+beds but price 2x (penthouse vs studio) -> must NOT merge
        pipeline.process_item(
            _item(property_id="UNIT_B", price_pcm=5000, fingerprint="fp_building"),
            MagicMock(),
        )
        pipeline.conn.commit()

        pipeline.cursor.execute(
            "SELECT COUNT(*) FROM listings WHERE address_fingerprint='fp_building'"
        )
        assert pipeline.cursor.fetchone()[0] == 2, (
            "cross-unit merge happened — price-tolerance guard failed"
        )
        assert pipeline.stats["inserted"] == 2

    def test_exact_id_match_takes_precedence_over_fingerprint(self, pipeline):
        """Same property_id always updates in place regardless of fingerprint."""
        pipeline.process_item(
            _item(property_id="SAME", price_pcm=2500, fingerprint="fp_x"),
            MagicMock(),
        )
        pipeline.conn.commit()
        pipeline.process_item(
            _item(property_id="SAME", price_pcm=2500, fingerprint="fp_x"),
            MagicMock(),
        )
        pipeline.conn.commit()

        pipeline.cursor.execute("SELECT COUNT(*) FROM listings WHERE property_id='SAME'")
        assert pipeline.cursor.fetchone()[0] == 1
        assert pipeline.stats["inserted"] == 1
        assert pipeline.stats["updated"] == 1

    def test_different_source_same_fingerprint_does_not_merge(self, pipeline):
        """Fingerprint fallback is scoped to a single source (aggregator vs agent)."""
        pipeline.process_item(
            _item(source="rightmove", property_id="rm1", price_pcm=3000,
                  fingerprint="fp_shared"),
            MagicMock(),
        )
        pipeline.conn.commit()
        pipeline.process_item(
            _item(source="savills", property_id="sv1", price_pcm=3000,
                  fingerprint="fp_shared"),
            MagicMock(),
        )
        pipeline.conn.commit()

        pipeline.cursor.execute(
            "SELECT COUNT(*) FROM listings WHERE address_fingerprint='fp_shared'"
        )
        assert pipeline.cursor.fetchone()[0] == 2, (
            "fingerprint fallback wrongly merged across sources"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
