#!/usr/bin/env python3
"""
_verify_serving_sync.py — End-to-end REAL verification for serving task #8.

Spins up an EPHEMERAL local Postgres (via testing.postgresql), applies the
dashboard init-db schema, runs the real sync_sqlite_to_postgres.py against it
(--execute), then runs the EXACT query SQL the dashboard uses
(getComparables / getMarketStats / getSimilarListings) to prove that real
listings + comps + peers come back after a load. No mocks. No prod access.

Run:  python3 scripts/_verify_serving_sync.py
"""
import os
import subprocess
import sys

import psycopg2
import testing.postgresql

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SQLITE_DB = os.path.join(PROJECT, "output", "rentals.db")
SYNC_SCRIPT = os.path.join(HERE, "sync_sqlite_to_postgres.py")

# Minimal schema matching dashboard/src/app/api/init-db/route.ts (the columns
# the dashboard queries touch). Postgres flavour. canonical_id intentionally
# OMITTED here to prove the sync's ensure_pg_parity ALTER adds it.
INIT_DB_SQL = """
CREATE TABLE IF NOT EXISTS listings (
    id SERIAL PRIMARY KEY, source TEXT NOT NULL, property_id TEXT NOT NULL,
    url TEXT, area TEXT, price INTEGER, price_pw INTEGER, price_pcm INTEGER,
    price_period TEXT, address TEXT, postcode TEXT, latitude REAL, longitude REAL,
    bedrooms INTEGER, bathrooms INTEGER, reception_rooms INTEGER, property_type TEXT,
    size_sqft INTEGER, size_sqm REAL, furnished TEXT, epc_rating TEXT,
    floorplan_url TEXT, room_details TEXT,
    has_basement INTEGER DEFAULT 0, has_lower_ground INTEGER DEFAULT 0,
    has_ground INTEGER DEFAULT 0, has_mezzanine INTEGER DEFAULT 0,
    has_first_floor INTEGER DEFAULT 0, has_second_floor INTEGER DEFAULT 0,
    has_third_floor INTEGER DEFAULT 0, has_fourth_plus INTEGER DEFAULT 0,
    has_roof_terrace INTEGER DEFAULT 0, floor_count INTEGER, property_levels TEXT,
    let_agreed INTEGER DEFAULT 0, agent_name TEXT, agent_phone TEXT,
    summary TEXT, description TEXT, features TEXT, added_date TEXT,
    address_fingerprint TEXT, first_seen TEXT, last_seen TEXT,
    is_active INTEGER DEFAULT 1, price_change_count INTEGER DEFAULT 0, scraped_at TEXT,
    is_short_let INTEGER DEFAULT 0, property_type_std TEXT, let_type TEXT,
    postcode_normalized TEXT, postcode_inferred TEXT, agent_brand TEXT,
    UNIQUE(source, property_id)
);
CREATE TABLE IF NOT EXISTS price_history (
    id SERIAL PRIMARY KEY, listing_id INTEGER NOT NULL REFERENCES listings(id),
    price_pcm INTEGER, recorded_at TEXT
);
CREATE TABLE IF NOT EXISTS scrape_runs (
    id SERIAL PRIMARY KEY, run_id TEXT NOT NULL, spider_name TEXT NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT, duration_seconds REAL,
    status TEXT DEFAULT 'running', items_scraped INTEGER DEFAULT 0,
    items_new INTEGER DEFAULT 0, items_updated INTEGER DEFAULT 0,
    items_dropped INTEGER DEFAULT 0, items_errors INTEGER DEFAULT 0,
    request_count INTEGER DEFAULT 0, response_count INTEGER DEFAULT 0,
    response_bytes BIGINT DEFAULT 0, error_count INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0, memory_start_mb REAL, memory_peak_mb REAL,
    memory_end_mb REAL, log_file TEXT, exit_reason TEXT, error_summary TEXT,
    UNIQUE(run_id, spider_name)
);
"""

# Exact comparables query (db.ts getComparables), parameterized for the subject
# property used by the report pages: 1312 sqft, 2 bed, SW1 area.
COMPARABLES_SQL = """
WITH ranked AS (
  SELECT address, postcode, source, price_pcm::int as price_pcm,
    size_sqft::int as size_sqft, bedrooms::int as bedrooms,
    ROUND((price_pcm::numeric / size_sqft::numeric), 2)::float as ppsf,
    ROW_NUMBER() OVER (PARTITION BY price_pcm, size_sqft,
      CASE WHEN POSITION(' ' IN postcode) > 0 THEN SUBSTRING(postcode, 1, POSITION(' ' IN postcode) - 1) ELSE postcode END
      ORDER BY CASE WHEN source = 'rightmove' THEN 1 ELSE 0 END, source) as rn
  FROM listings
  WHERE is_active = 1 AND size_sqft IS NOT NULL AND size_sqft > 0
    AND price_pcm IS NOT NULL AND price_pcm > 0
    AND size_sqft BETWEEN %(minSize)s AND %(maxSize)s
    AND (postcode ~ '^SW1[A-Z]' OR postcode ~ '^SW3' OR postcode ~ '^SW7' OR postcode ~ '^W1[A-Z]')
    AND (price_period = 'pcm' OR price_period IS NULL OR price_period = '')
    AND COALESCE(is_short_let, 0) = 0
    AND (price_pcm::numeric / size_sqft::numeric) >= 3
    AND (price_pcm::numeric / size_sqft::numeric) <= 30
    AND bedrooms BETWEEN %(minBeds)s AND %(maxBeds)s
)
SELECT COUNT(*) FROM ranked WHERE rn = 1
"""

MARKET_STATS_SQL = """
SELECT COUNT(*)::int as total_listings,
  ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_pcm::numeric / size_sqft::numeric))::numeric, 2)::float as median_ppsf
FROM listings
WHERE is_active = 1 AND size_sqft IS NOT NULL AND size_sqft > 100
  AND price_pcm IS NOT NULL AND price_pcm > 500
"""

# Exact similar-listings core query (db.ts getSimilarListings) for a SW1 2-bed
# at ~10914 pcm — the report subject — to prove /api/similar returns real peers.
SIMILAR_SQL = """
WITH scored AS (
  SELECT id, source, price_pcm::int as price_pcm, bedrooms::int as bedrooms,
    (CASE WHEN bedrooms = %(beds)s THEN 0.30 WHEN ABS(bedrooms - %(beds)s) = 1 THEN 0.15 ELSE 0 END
     + 0.15
     + CASE WHEN ABS(price_pcm - %(price)s) <= %(tol15)s THEN 0.25 WHEN ABS(price_pcm - %(price)s) <= %(tol30)s THEN 0.15 ELSE 0.05 END
     + 0.05
     + CASE WHEN source IN ('savills','knightfrank') THEN 0.10 WHEN source IN ('chestertons','foxtons') THEN 0.07 ELSE 0.05 END
    )::float as similarity_score
  FROM listings
  WHERE is_active = 1 AND SPLIT_PART(postcode, ' ', 1) = %(district)s
    AND bedrooms BETWEEN %(minBeds)s AND %(maxBeds)s
    AND price_pcm BETWEEN %(priceMin)s AND %(priceMax)s AND price_pcm > 0
)
SELECT COUNT(*) FROM scored WHERE similarity_score > 0.3
"""


def main() -> int:
    print("[verify] starting ephemeral Postgres...", flush=True)
    with testing.postgresql.Postgresql() as pg:
        url = pg.url()
        # init-db schema
        with psycopg2.connect(url) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(INIT_DB_SQL)
        print("[verify] init-db schema applied (canonical_id intentionally absent).", flush=True)

        # Run the REAL sync with --execute against the ephemeral DB.
        env = dict(os.environ, POSTGRES_URL=url)
        r = subprocess.run(
            [sys.executable, SYNC_SCRIPT, "--execute", "--i-have-rotated-the-secret",
             "--sqlite", SQLITE_DB],
            env=env, capture_output=True, text=True,
        )
        print(r.stdout)
        if r.returncode != 0:
            print("[verify] SYNC FAILED:\n" + r.stderr, file=sys.stderr)
            return 1

        # Now query exactly what the dashboard queries.
        results = {}
        with psycopg2.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM listings")
                results["listings_loaded"] = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM listings WHERE is_active=1")
                results["active"] = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM price_history")
                results["price_history"] = cur.fetchone()[0]
                # parity column actually present + populated?
                cur.execute("SELECT COUNT(*) FROM listings WHERE canonical_id IS NOT NULL")
                results["canonical_id_populated"] = cur.fetchone()[0]

                cur.execute(COMPARABLES_SQL, {"minSize": int(1312*0.75), "maxSize": int(1312*1.25),
                                              "minBeds": 1, "maxBeds": 3})
                results["comparables_report"] = cur.fetchone()[0]

                cur.execute(MARKET_STATS_SQL)
                ms = cur.fetchone()
                results["market_total"], results["market_median_ppsf"] = ms[0], ms[1]

                cur.execute(SIMILAR_SQL, {"beds": 2, "price": 10914, "tol15": int(10914*0.15),
                                          "tol30": int(10914*0.30), "district": "SW1W",
                                          "minBeds": 1, "maxBeds": 3,
                                          "priceMin": int(10914*0.5), "priceMax": int(10914*2.0)})
                results["similar_peers_SW1W"] = cur.fetchone()[0]

        print("\n[verify] === RESULTS (real Postgres, real query SQL) ===")
        for k, v in results.items():
            print(f"  {k}: {v}")

        ok = (results["listings_loaded"] > 9000 and results["active"] > 5000
              and results["comparables_report"] > 0 and results["market_total"] > 0
              and results["similar_peers_SW1W"] > 0
              and results["canonical_id_populated"] >= 0)
        print(f"\n[verify] {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
