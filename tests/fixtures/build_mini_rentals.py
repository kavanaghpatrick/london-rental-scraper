#!/usr/bin/env python3
"""Deterministic builder for the committed mini rentals.db CI fixture.

WHY THIS EXISTS
---------------
tests/test_scrape_validation.py and tests/test_backfill_acceptance.py read a
populated output/rentals.db. That DB is gitignored (it's 20MB+ of live data), so in
PR CI it is ABSENT and tests/conftest.py auto-SKIPS the whole structural suite. A
green CI that skipped 33 data-integrity tests gives zero pre-merge protection.

This module builds a SMALL, deterministic, schema-faithful SQLite DB so those
STRUCTURAL / LOGIC asserts EXECUTE in PR CI against a known fixture. The schema is a
byte-for-byte copy of property_scraper.pipelines.SQLitePipeline._create_full_schema
(+ price_history + scrape_runs), so the fixture exercises the same columns the real
pipeline writes. It is intentionally NOT the live DB: freshness/recency asserts (a
scrape ran TODAY, coverage-vs-snapshot) are gated separately in conftest and still
skip — they only mean something post-scrape.

DETERMINISM
-----------
No randomness, no wall-clock: every value (including first_seen/last_seen dates and
the 16-char address fingerprints, which mirror the sha256[:16] of the real
fingerprint service input key) is fixed. Re-running produces a byte-identical DB
(modulo SQLite page layout), so a committed copy and a freshly-built copy assert the
same facts.

USAGE
-----
  python3 tests/fixtures/build_mini_rentals.py            # writes the committed .db
  from tests.fixtures.build_mini_rentals import build_mini_rentals
  build_mini_rentals(Path("/tmp/x.db"))                   # build anywhere (conftest)

The committed artifact lives next to this file as ``mini_rentals.db``.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "mini_rentals.db"

# Fixed dates — chronological, all in the PAST relative to any realistic run so the
# post-scrape "updated today" freshness asserts (gated separately) never spuriously
# fire on the fixture. first_seen <= last_seen everywhere.
_FIRST_SEEN = "2026-01-05 09:00:00"
_LAST_SEEN = "2026-05-20 09:00:00"
_RUN_STARTED = "2026-05-20 08:00:00"
_RUN_FINISHED = "2026-05-20 08:42:00"


# --- schema (copied from property_scraper/pipelines.py _create_full_schema) ---------
_LISTINGS_DDL = """
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
"""

_PRICE_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id INTEGER NOT NULL,
    price_pcm INTEGER,
    recorded_at TEXT,
    FOREIGN KEY (listing_id) REFERENCES listings(id)
)
"""

# scrape_runs DDL mirrors scripts/_verify_serving_sync.py (the init-db schema) in the
# SQLite flavour the audit logger writes locally.
_SCRAPE_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS scrape_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    spider_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_seconds REAL,
    status TEXT DEFAULT 'running',
    items_scraped INTEGER DEFAULT 0,
    items_new INTEGER DEFAULT 0,
    items_updated INTEGER DEFAULT 0,
    items_dropped INTEGER DEFAULT 0,
    items_errors INTEGER DEFAULT 0,
    request_count INTEGER DEFAULT 0,
    response_count INTEGER DEFAULT 0,
    response_bytes INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    memory_start_mb REAL,
    memory_peak_mb REAL,
    memory_end_mb REAL,
    log_file TEXT,
    exit_reason TEXT,
    error_summary TEXT,
    UNIQUE(run_id, spider_name)
)
"""


def _fp(source: str, address: str, postcode: str) -> str:
    """Deterministic 16-char fingerprint, mirroring the real service's sha256[:16].

    The real generate_fingerprint() keys on a normalized address+postcode; we don't
    need byte-parity with it (the fixture isn't a dedupe oracle), only the SAME
    SHAPE: a stable 16-char lowercase hex string so the
    test_integrity_fingerprint_format (LENGTH == 16) assert runs for real.
    """
    key = f"{address}|{postcode}".lower().strip()
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# Representative tiny dataset: a handful of listings across the real sources, in the
# prime-central postcodes the dashboard cares about (SW3/SW1/SW7/W1). Mix of:
#   * with-sqft and without-sqft (the no-sqft rows are the backfill targets),
#   * with-floorplan_url and without (stage-1 enrich targets),
#   * active and one inactive (is_active=0) to exercise the active filter,
#   * pcm price_period (the only comparable kind),
#   * one row with a price change (-> a price_history row + price_change_count=1).
#
# Tuple order matches _COLS below.
_COLS = [
    "source", "property_id", "url", "area", "price_pcm", "price_period",
    "address", "postcode", "latitude", "longitude", "bedrooms", "bathrooms",
    "property_type", "size_sqft", "floorplan_url", "address_fingerprint",
    "first_seen", "last_seen", "is_active", "price_change_count", "scraped_at",
]

# id is implicit (AUTOINCREMENT, 1-based in insertion order).
_ROWS = [
    # 1 — savills, SW3, with sqft + floorplan, active
    ("savills", "SAV-001", "https://savills.example/1", "Chelsea", 5200, "pcm",
     "10 Cheyne Walk", "SW3 5RA", 51.4839, -0.1700, 2, 2, "flat", 1050,
     "https://cdn.savills.example/fp1.png", None,
     _FIRST_SEEN, _LAST_SEEN, 1, 1, _LAST_SEEN),
    # 2 — knightfrank, SW3, with sqft, active (a comparable peer to #1)
    ("knightfrank", "KF-001", "https://kf.example/2", "Chelsea", 5800, "pcm",
     "22 Flood Street", "SW3 5TE", 51.4861, -0.1672, 2, 2, "flat", 1120,
     "https://cdn.kf.example/fp2.png", None,
     _FIRST_SEEN, _LAST_SEEN, 1, 0, _LAST_SEEN),
    # 3 — foxtons, SW1W, with sqft, active
    ("foxtons", "FOX-001", "https://foxtons.example/3", "Belgravia", 11000, "pcm",
     "5 Eaton Square", "SW1W 9DA", 51.4953, -0.1530, 2, 2, "flat", 1312,
     None, None,
     _FIRST_SEEN, _LAST_SEEN, 1, 0, _LAST_SEEN),
    # 4 — rightmove, SW3, NO sqft but HAS floorplan_url -> a backfill OCR target, active
    ("rightmove", "RM-001", "https://rightmove.example/4", "Chelsea", 4950, "pcm",
     "8 Old Church Street", "SW3 6EA", 51.4847, -0.1719, 2, 1, "flat", None,
     "https://media.rightmove.example/fp4.png", None,
     _FIRST_SEEN, _LAST_SEEN, 1, 0, _LAST_SEEN),
    # 5 — rightmove, SW7, NO sqft and NO floorplan_url -> backfill can't recover, active
    ("rightmove", "RM-002", "https://rightmove.example/5", "South Kensington", 6800,
     "pcm", "14 Queens Gate", "SW7 5JG", 51.4970, -0.1790, 3, 2, "flat", None,
     None, None,
     _FIRST_SEEN, _LAST_SEEN, 1, 0, _LAST_SEEN),
    # 6 — chestertons, W1, with sqft, active
    ("chestertons", "CHE-001", "https://chestertons.example/6", "Marylebone", 7200,
     "pcm", "30 Marylebone High Street", "W1U 4PL", 51.5210, -0.1510, 2, 2, "flat",
     980, "https://cdn.chestertons.example/fp6.png", None,
     _FIRST_SEEN, _LAST_SEEN, 1, 0, _LAST_SEEN),
    # 7 — savills, SW1, with sqft, INACTIVE (is_active=0) -> excluded by active filters
    ("savills", "SAV-002", "https://savills.example/7", "Westminster", 4300, "pcm",
     "3 Smith Square", "SW1P 3HA", 51.4960, -0.1270, 1, 1, "flat", 720,
     "https://cdn.savills.example/fp7.png", None,
     _FIRST_SEEN, _LAST_SEEN, 0, 0, _LAST_SEEN),
    # 8 — foxtons, SW3, studio, with sqft, active (a different-bed peer)
    ("foxtons", "FOX-002", "https://foxtons.example/8", "Chelsea", 2600, "pcm",
     "1 Sloane Avenue", "SW3 3DZ", 51.4920, -0.1660, 0, 1, "studio", 420,
     None, None,
     _FIRST_SEEN, _LAST_SEEN, 1, 0, _LAST_SEEN),
]


# The single row the "backfilled" variant recovers a sqft for: listing #4 (RM-001),
# the active rightmove row that had a floorplan_url but NO sqft. A real OCR backfill
# would read its floorplan and write a sane, trainable sqft (+ a floor flag). This lets
# the backfill RECOVERED-ROW SANITY tests assert real behaviour against the fixtures
# (baseline = pre, backfilled = post), not just the no-op invariants.
_BACKFILLED_RECOVERED = {
    # property_id -> (recovered_size_sqft, has_first_floor)
    "RM-001": (760, 1),
}


def build_mini_rentals(db_path: Path = DEFAULT_DB, variant: str = "baseline") -> Path:
    """Build the deterministic mini fixture DB at ``db_path``. Returns the path.

    variant:
      * "baseline"   — the canonical fixture (also the committed mini_rentals.db).
                       Rightmove RM-001 has a floorplan_url but NO sqft (a backfill
                       TARGET). This is what the scrape-validation structural suite and
                       the backfill BASELINE read.
      * "backfilled" — identical to baseline EXCEPT RM-001 has a recovered, trainable
                       size_sqft (+ floor flag) as a real OCR backfill would produce.
                       Used as the backfill LIVE (post) DB so the recovered-row sanity
                       tests run for real. Row count + identity/price columns are
                       otherwise byte-identical to baseline (non-destructive invariant).

    Idempotent: removes any existing file first so the result is reproducible.
    """
    if variant not in {"baseline", "backfilled"}:
        raise ValueError(f"unknown variant {variant!r}")
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(_LISTINGS_DDL)
        cur.execute(_PRICE_HISTORY_DDL)
        cur.execute(_SCRAPE_RUNS_DDL)

        col_list = ", ".join(_COLS)
        placeholders = ", ".join("?" for _ in _COLS)
        insert_sql = f"INSERT INTO listings ({col_list}) VALUES ({placeholders})"

        for row in _ROWS:
            d = dict(zip(_COLS, row))
            # Fill the fingerprint deterministically if the row left it None.
            if d["address_fingerprint"] is None:
                d["address_fingerprint"] = _fp(d["source"], d["address"], d["postcode"])
            # In the "backfilled" variant, apply the OCR-recovered sqft for the target
            # row (and ONLY size_sqft — identity/price columns stay byte-identical to
            # baseline so the non-destructive invariants hold).
            if variant == "backfilled" and d["property_id"] in _BACKFILLED_RECOVERED:
                d["size_sqft"] = _BACKFILLED_RECOVERED[d["property_id"]][0]
            cur.execute(insert_sql, [d[c] for c in _COLS])

        # The recovered floor flag (a separate column not in _COLS) for the backfilled
        # variant — mirrors OCR also writing has_first_floor when it reads the floorplan.
        if variant == "backfilled":
            for pid, (_sqft, has_first) in _BACKFILLED_RECOVERED.items():
                cur.execute(
                    "UPDATE listings SET has_first_floor=? WHERE property_id=?",
                    (has_first, pid),
                )

        # price_history: an initial price log for every listing (mirrors the pipeline
        # logging the initial price on insert), plus a price CHANGE row for listing 1
        # (which carries price_change_count=1). listing_id is the 1-based rowid.
        for listing_id, row in enumerate(_ROWS, start=1):
            initial_price = dict(zip(_COLS, row))["price_pcm"]
            cur.execute(
                "INSERT INTO price_history (listing_id, price_pcm, recorded_at) VALUES (?, ?, ?)",
                (listing_id, initial_price, _FIRST_SEEN),
            )
        # The one real price change: listing 1 moved 5400 -> 5200.
        cur.execute(
            "INSERT INTO price_history (listing_id, price_pcm, recorded_at) VALUES (?, ?, ?)",
            (1, 5400, _FIRST_SEEN),
        )

        # scrape_runs: one finished run per source (no 'running' zombie rows — those
        # are a known hazard the audit logger reaps; the fixture stays clean).
        sources = sorted({dict(zip(_COLS, r))["source"] for r in _ROWS})
        for i, src in enumerate(sources):
            cur.execute(
                "INSERT INTO scrape_runs (run_id, spider_name, started_at, finished_at, "
                "duration_seconds, status, items_scraped, error_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"fixture-run-{i}", src, _RUN_STARTED, _RUN_FINISHED, 2520.0,
                 "finished", 1, 0),
            )

        conn.commit()
    finally:
        conn.close()
    return db_path


if __name__ == "__main__":
    out = build_mini_rentals()
    # Tiny summary so a human running this sees what it produced.
    conn = sqlite3.connect(str(out))
    try:
        (n_listings,) = conn.execute("SELECT COUNT(*) FROM listings").fetchone()
        (n_active,) = conn.execute("SELECT COUNT(*) FROM listings WHERE is_active=1").fetchone()
        (n_ph,) = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()
        (n_runs,) = conn.execute("SELECT COUNT(*) FROM scrape_runs").fetchone()
    finally:
        conn.close()
    print(f"Built {out}")
    print(f"  listings={n_listings} active={n_active} price_history={n_ph} scrape_runs={n_runs}")
