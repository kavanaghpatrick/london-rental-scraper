"""
T8 (A5/A12) — cross-source dedupe --remove must cascade-delete price_history and
route the delete (incl. a price_history backup) through scripts/_safe_delete.guarded_delete.

Background / the bug this pins
------------------------------
`dedupe_cross_source.remove_duplicates` -> its inner `_do_delete` historically ran
ONLY `DELETE FROM listings WHERE id IN (...)`. The `price_history` rows that point at
the deleted listing (FK `listing_id`) were left behind => ORPHANED history rows that
can never be reached again and that misrepresent the table. The other two destructive
paths already cascade (dedupe_same_source.py:190, daily-scrape.yml:354), so this one
diverged. It also meant the guard's CSV backup captured ONLY the listings rows, so the
lost price_history was unrecoverable.

This test is PURE-UNIT: an in-memory SQLite DB faithful to the real schema (listings +
price_history with the FK), seeded with ONE cross-source duplicate (same structural
address fingerprint + bedrooms + price-within-5%, two distinct sources) and a
price_history row attached to the NON-canonical listing. It asserts:

  (a) ONLY the non-canonical id is deleted (canonical savills row survives);
  (b) NO orphan price_history remains (the cascade ran);
  (c) the guard's backup directory captured the deleted price_history row too
      (recoverability), not just the listings row.

RED (pre-fix): (b) fails — the orphan price_history row survives — and (c) fails — no
price_history backup CSV is written.
"""
import csv
import sqlite3
import sys
from pathlib import Path

import pytest

# Import the CLI module from the repo root (dedupe_cross_source.py lives at the root,
# next to cli/ and property_scraper/, per CLAUDE.md).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import dedupe_cross_source as dcs  # noqa: E402


# The canonical-vs-noncanonical pick: savills (priority 5, has sqft) is KEPT;
# rightmove (priority 1, no sqft) is the duplicate to DELETE.
_ADDRESS = "Flat 2, 100 Kings Road"
_POSTCODE = "SW3 4TX"


def _build_db() -> sqlite3.Connection:
    """In-memory DB with the real listings + price_history schema and ONE cross-source dup.

    Two listings (savills + rightmove) share address/postcode/bedrooms and price within
    5%, so the structural-fingerprint core clusters them. A price_history row is attached
    to the rightmove (non-canonical) listing — exactly the row a correct cascade must
    take with it.
    """
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE listings (
            id INTEGER PRIMARY KEY,
            source TEXT,
            property_id TEXT,
            address_fingerprint TEXT,
            address TEXT,
            postcode TEXT,
            price_pcm INTEGER,
            size_sqft INTEGER,
            bedrooms INTEGER,
            bathrooms INTEGER,
            url TEXT,
            is_active INTEGER DEFAULT 1
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE price_history (
            id INTEGER PRIMARY KEY,
            listing_id INTEGER,
            old_price INTEGER,
            new_price INTEGER,
            recorded_at TEXT,
            FOREIGN KEY (listing_id) REFERENCES listings(id)
        )
        """
    )
    # id=1 savills (canonical: best priority + has sqft) — KEEP
    cur.execute(
        "INSERT INTO listings (id, source, property_id, address, postcode, price_pcm,"
        " size_sqft, bedrooms, bathrooms, url, is_active) VALUES (?,?,?,?,?,?,?,?,?,?,1)",
        (1, "savills", "SV9", _ADDRESS, _POSTCODE, 3000, 850, 2, 2, "http://s/1"),
    )
    # id=2 rightmove (non-canonical: aggregator, no sqft, price within 5%) — DELETE
    cur.execute(
        "INSERT INTO listings (id, source, property_id, address, postcode, price_pcm,"
        " size_sqft, bedrooms, bathrooms, url, is_active) VALUES (?,?,?,?,?,?,?,?,?,?,1)",
        (2, "rightmove", "RM1", _ADDRESS, _POSTCODE, 3050, None, 2, 1, "http://r/2"),
    )
    # A DISTINCT, non-duplicate control listing that must NOT be touched.
    cur.execute(
        "INSERT INTO listings (id, source, property_id, address, postcode, price_pcm,"
        " size_sqft, bedrooms, bathrooms, url, is_active) VALUES (?,?,?,?,?,?,?,?,?,?,1)",
        (3, "foxtons", "FX5", "9 Other Street", "E1 6AA", 1800, 500, 1, 1, "http://f/3"),
    )
    # Distinct filler listings so the table is large enough that deleting the single
    # duplicate (1 row) stays UNDER the guard's 10% delta-abort threshold — otherwise the
    # guard trips on the tiny fixture and we never exercise the cascade/backup path. Each
    # has a unique address/postcode/source so none cluster with the seeded duplicate.
    for i in range(4, 17):
        cur.execute(
            "INSERT INTO listings (id, source, property_id, address, postcode, price_pcm,"
            " size_sqft, bedrooms, bathrooms, url, is_active) VALUES (?,?,?,?,?,?,?,?,?,?,1)",
            (i, "rightmove", f"RMF{i}", f"{i} Filler Avenue", f"N{i} 1AA",
             1500 + i, 400 + i, 1, 1, f"http://f/{i}"),
        )
    # price_history rows. One is attached to the doomed rightmove listing (id=2) — the
    # row the cascade must delete + back up. One is attached to the surviving savills
    # listing (id=1) — must NOT be deleted.
    cur.execute(
        "INSERT INTO price_history (id, listing_id, old_price, new_price, recorded_at)"
        " VALUES (?,?,?,?,?)",
        (10, 2, 3200, 3050, "2026-01-01T00:00:00"),
    )
    cur.execute(
        "INSERT INTO price_history (id, listing_id, old_price, new_price, recorded_at)"
        " VALUES (?,?,?,?,?)",
        (11, 1, 3100, 3000, "2026-01-02T00:00:00"),
    )
    conn.commit()
    return conn


_ALL_IDS = set(range(1, 17))  # 1,2,3 + fillers 4..16


def _which_deleted(conn) -> set:
    rows = conn.execute("SELECT id FROM listings ORDER BY id").fetchall()
    return _ALL_IDS - {r[0] for r in rows}


def test_remove_duplicates_cascades_and_backs_up_price_history(tmp_path):
    conn = _build_db()
    cur = conn.cursor()

    # Sanity: the seed actually contains a cross-source duplicate the core will cluster.
    dups = dcs.find_duplicates(conn)
    assert dups, "seed must contain a cross-source duplicate for this test to be meaningful"

    # Direct the guard's CSV backups to a tmp dir so the test can inspect them.
    deleted = dcs.remove_duplicates(conn, dry_run=False, project_root=tmp_path)

    # (a) ONLY the non-canonical rightmove id (2) is deleted; savills (1) + control (3) stay.
    assert deleted == [2]
    assert _which_deleted(conn) == {2}
    assert cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 15
    assert cur.execute(
        "SELECT id FROM listings WHERE source='savills'"
    ).fetchone()[0] == 1

    # (b) NO orphan price_history remains: the row attached to the deleted listing (id=2)
    # is gone; the row attached to the surviving listing (id=1) stays.
    ph = cur.execute(
        "SELECT id, listing_id FROM price_history ORDER BY id"
    ).fetchall()
    assert ph == [(11, 1)], f"orphaned/over-deleted price_history: {ph}"
    # No price_history row points at a now-missing listing.
    orphans = cur.execute(
        "SELECT ph.id FROM price_history ph "
        "LEFT JOIN listings l ON ph.listing_id = l.id WHERE l.id IS NULL"
    ).fetchall()
    assert orphans == [], f"orphan price_history rows survived: {orphans}"

    # (c) The guard's backup captured the price_history row too (recoverability).
    backup_dir = tmp_path / "output" / "deleted_backups"
    assert backup_dir.exists(), "guarded_delete backup dir not created under project_root"
    ph_backups = list(backup_dir.glob("price_history_deleted_*.csv"))
    assert ph_backups, (
        "no price_history backup written — the deleted history is unrecoverable. "
        f"Files present: {[p.name for p in backup_dir.iterdir()]}"
    )
    # The backup contains the deleted history row (listing_id=2).
    backed = []
    for p in ph_backups:
        with open(p) as f:
            rows = list(csv.DictReader(f))
        backed.extend(rows)
    backed_listing_ids = {int(r["listing_id"]) for r in backed}
    assert 2 in backed_listing_ids, (
        f"deleted price_history (listing_id=2) not in backup; got {backed_listing_ids}"
    )

    # And a listings backup also exists (the original guard behavior is preserved).
    listing_backups = list(backup_dir.glob("listings_deleted_*.csv"))
    assert listing_backups, "listings backup missing — guard regression"

    conn.close()
