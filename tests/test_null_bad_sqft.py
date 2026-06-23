"""T4 — data test for scripts/null_bad_sqft.py.

The guarded one-time cleanup NULLs ECONOMICALLY-BAD size_sqft values (the 222 sub-150
sqm-as-sqft rows + the >10000 garbage rows whose £/sqft is impossibly low) WITHOUT
deleting any row and WITHOUT touching the recovery columns (size_sqm, floorplan_url)
so re-OCR can recover later.

The predicate is ECONOMICS-AWARE (not a flat 10000 cliff):
    size_sqft < 150
    OR size_sqft > 14000
    OR (size_sqft > 10000 AND price_pcm > 0 AND price_pcm/size_sqft < 3.0)
This KEEPS real prime-London mega-mansions (10000-14000 sqft at £>=3/sqft) and NULLs
the OCR garbage that merely landed above 10000.

These tests run against a TEMP COPY of a synthetic DB (never the live rentals.db) and
exercise the script as a module: dry-run (no mutation), --execute (nulls + idempotent),
row-count invariance, recovery-column preservation, and the --max-rows safety cap.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_cleanup():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "null_bad_sqft", str(ROOT / "scripts" / "null_bad_sqft.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cleanup = _load_cleanup()


# Minimal listings schema with the columns the predicate + preservation touch.
_SCHEMA = """
CREATE TABLE listings (
    id INTEGER PRIMARY KEY,
    source TEXT,
    property_id TEXT,
    size_sqft INTEGER,
    size_sqm INTEGER,
    bedrooms INTEGER,
    price_pcm INTEGER,
    floorplan_url TEXT
);
"""

# id, source, sqft, sqm, beds, price, floorplan_url
_ROWS = [
    (1, "rightmove", 84, 8, 1, 950, "http://fp/1.png"),     # bad: sqm-as-sqft  (<150)
    (2, "rightmove", 120, 11, 3, 8000, "http://fp/2.png"),  # bad: sqm-as-sqft  (<150)
    (3, "rightmove", 149, None, 2, 2000, None),             # bad: just below floor
    # id 4: REAL mega-mansion (12415 sqft @ £150,000 pcm = £12.08/sqft >= 3) -> SURVIVES.
    (4, "rightmove", 12415, None, 8, 150000, "http://fp/4.png"),  # GOOD (economics): >10000 but £12/sqft
    # id 5: garbage >10000 (10737 sqft @ £28,000 pcm = £2.61/sqft < 3) -> NULL.
    (5, "rightmove", 10737, None, 7, 28000, None),          # bad: >10000 AND £/sqft<3
    (6, "rightmove", 900, 84, 2, 2700, "http://fp/6.png"),  # GOOD — must survive
    (7, "savills", 100, 9, 1, 1100, None),                  # bad other-source (<150)
    (8, "foxtons", 650, 60, 1, 1800, None),                 # GOOD
    (9, "rightmove", None, None, 1, 1500, None),            # NULL sqft — untouched
    (10, "rightmove", 0, None, 1, 1500, None),              # zero sqft — untouched (>0 guard)
    # id 11: absurd >14000 -> NULL unconditionally (27225 sqft, also £0.18/sqft).
    (11, "rightmove", 27225, None, 5, 4901, None),          # bad: >14000 ceiling
    # id 12: >10000 but price unknown (price NULL) and <=14000 -> economics CANNOT prove
    #        garbage, so it SURVIVES (price_pcm>0 guard on the economic branch).
    (12, "rightmove", 11000, None, 6, None, "http://fp/12.png"),  # GOOD: >10000 but price NULL, <=14000
]


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "rentals_copy.db"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    conn.executemany(
        "INSERT INTO listings (id,source,size_sqft,size_sqm,bedrooms,price_pcm,floorplan_url) "
        "VALUES (?,?,?,?,?,?,?)",
        _ROWS,
    )
    conn.commit()
    conn.close()
    return db


_BAD_PREDICATE = (
    "SELECT COUNT(*) FROM listings WHERE size_sqft IS NOT NULL AND size_sqft > 0 "
    "AND (size_sqft < 150 OR size_sqft > 14000 "
    "OR (size_sqft > 10000 AND price_pcm IS NOT NULL AND price_pcm > 0 "
    "AND CAST(price_pcm AS REAL) / size_sqft < 3.0))"
)


def _bad_count(db: Path) -> int:
    conn = sqlite3.connect(db)
    n = conn.execute(_BAD_PREDICATE).fetchone()[0]
    conn.close()
    return n


def _total(db: Path) -> int:
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    conn.close()
    return n


def test_baseline_has_six_bad_rows(tmp_path):
    db = _make_db(tmp_path)
    # ids 1,2,3 (<150 rightmove) + 5 (>10000 & £2.61/sqft rightmove) + 7 (<150 savills)
    # + 11 (>14000 rightmove) = 6 bad across all sources. id 4 (£12/sqft mansion) and
    # id 12 (>10000 price-NULL, <=14000) are GOOD and excluded.
    assert _bad_count(db) == 6


def test_dry_run_does_not_mutate(tmp_path):
    db = _make_db(tmp_path)
    res = cleanup.run_cleanup(str(db), execute=False)
    assert _bad_count(db) == 6, "dry-run must not change the DB"
    assert res["candidates"] == 6
    assert res["executed"] is False


def test_execute_nulls_all_bad_and_keeps_rows(tmp_path):
    db = _make_db(tmp_path)
    before_total = _total(db)
    res = cleanup.run_cleanup(str(db), execute=True)
    assert res["executed"] is True
    assert res["nulled"] == 6
    # bad-count reaches 0
    assert _bad_count(db) == 0
    # NO deletions
    assert _total(db) == before_total


def test_good_rows_survive(tmp_path):
    db = _make_db(tmp_path)
    cleanup.run_cleanup(str(db), execute=True)
    conn = sqlite3.connect(db)
    # good rows keep their sqft
    assert conn.execute("SELECT size_sqft FROM listings WHERE id=6").fetchone()[0] == 900
    assert conn.execute("SELECT size_sqft FROM listings WHERE id=8").fetchone()[0] == 650
    # ECONOMICS: real mega-mansion (id 4, £12/sqft) and the price-unknown >10000 row
    # (id 12, <=14000) must SURVIVE — they are not provably garbage.
    assert conn.execute("SELECT size_sqft FROM listings WHERE id=4").fetchone()[0] == 12415
    assert conn.execute("SELECT size_sqft FROM listings WHERE id=12").fetchone()[0] == 11000
    # bad rows are now NULL
    for bad_id in (1, 2, 3, 5, 7, 11):
        assert conn.execute("SELECT size_sqft FROM listings WHERE id=?", (bad_id,)).fetchone()[0] is None
    conn.close()


def test_recovery_columns_preserved(tmp_path):
    db = _make_db(tmp_path)
    cleanup.run_cleanup(str(db), execute=True)
    conn = sqlite3.connect(db)
    # size_sqm + floorplan_url on the NULLed ids must be UNTOUCHED so re-OCR can recover.
    row1 = conn.execute("SELECT size_sqm, floorplan_url FROM listings WHERE id=1").fetchone()
    assert row1 == (8, "http://fp/1.png")
    # id 5 was nulled (>10000 & £2.61/sqft); its recovery columns survive.
    row5 = conn.execute("SELECT size_sqm, floorplan_url FROM listings WHERE id=5").fetchone()
    assert row5 == (None, None)  # this synthetic row had no sqm/floorplan to begin with
    conn.close()


def test_idempotent_second_run_finds_zero(tmp_path):
    db = _make_db(tmp_path)
    cleanup.run_cleanup(str(db), execute=True)
    res2 = cleanup.run_cleanup(str(db), execute=True)
    assert res2["candidates"] == 0
    assert res2["nulled"] == 0
    assert _bad_count(db) == 0


def test_source_filter_only_rightmove(tmp_path):
    db = _make_db(tmp_path)
    res = cleanup.run_cleanup(str(db), execute=True, source="rightmove")
    # 5 rightmove bad nulled (ids 1,2,3,5,11); the savills bad (id 7) untouched.
    assert res["nulled"] == 5
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT size_sqft FROM listings WHERE id=7").fetchone()[0] == 100
    # the real mansion (id 4) survives even under the source filter.
    assert conn.execute("SELECT size_sqft FROM listings WHERE id=4").fetchone()[0] == 12415
    conn.close()


def test_max_rows_cap_aborts(tmp_path):
    db = _make_db(tmp_path)
    with pytest.raises(cleanup.CleanupAborted):
        cleanup.run_cleanup(str(db), execute=True, max_rows=2)
    # nothing changed
    assert _bad_count(db) == 6


def test_backup_taken_on_execute(tmp_path):
    db = _make_db(tmp_path)
    res = cleanup.run_cleanup(str(db), execute=True)
    assert res.get("backup_path"), "a backup path must be returned on --execute"
    assert Path(res["backup_path"]).exists()
