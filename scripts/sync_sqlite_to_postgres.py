#!/usr/bin/env python3
"""
sync_sqlite_to_postgres.py — Load the canonical SQLite DB into Neon Postgres.

Mirrors `output/rentals.db` (the canonical store, owned by dataeng — see
DATA_LAYER_CONTRACT.md) into the Vercel/Neon Postgres that the dashboard reads
through `@vercel/postgres`. This is the prod-data-sync for serving task #8.

Design / safety:
  * DRY-RUN BY DEFAULT. Without --execute the script connects, validates schema
    parity, and prints row-count deltas, but performs NO writes (rolls back).
  * --execute is GATED. It will refuse to run unless --i-have-rotated-the-secret
    is ALSO passed, so an accidental run cannot write prod. The lead runs the
    real load only AFTER the POSTGRES_URL secret has been rotated.
  * Reads a COPY of the canonical DB (never opens the live file directly), per
    DATA_LAYER_CONTRACT.md §8.
  * Preserves listings.id so price_history.listing_id foreign keys line up.
  * Idempotent: TRUNCATE + reload inside one transaction (atomic; readers never
    see a half-loaded table). Re-running is safe and converges to the SQLite state.
  * Ensures Postgres schema parity (adds canonical_id, price_pw) idempotently —
    these are the drift items handed to serving in DATA_LAYER_CONTRACT.md §3. The
    authoritative schema is created by the dashboard `init-db` route; this script
    only ADD COLUMN IF NOT EXISTS for the two parity columns so a load never
    fails on a column the canonical DB has but Postgres lacks.

Usage:
  # Dry run (default) — safe, no writes. Uses POSTGRES_URL from env.
  python3 scripts/sync_sqlite_to_postgres.py

  # Dry run against an explicit DB / URL
  python3 scripts/sync_sqlite_to_postgres.py --sqlite output/rentals.db --postgres-url "$POSTGRES_URL"

  # REAL prod load (GATED — lead runs this AFTER secret rotation):
  python3 scripts/sync_sqlite_to_postgres.py --execute --i-have-rotated-the-secret
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from contextlib import closing

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    sys.exit("psycopg2 is required: pip3 install psycopg2-binary")

DEFAULT_SQLITE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "rentals.db"
)

# Columns synced for `listings`. These exist in BOTH the canonical SQLite schema
# (DATA_LAYER_CONTRACT.md §2, 54 cols) and the Postgres schema created by
# dashboard/src/app/api/init-db/route.ts. `id` is included first so we can
# preserve primary keys (price_history.listing_id references them).
LISTINGS_COLUMNS = [
    "id", "source", "property_id", "url", "area",
    "price", "price_pw", "price_pcm", "price_period",
    "address", "postcode", "latitude", "longitude",
    "bedrooms", "bathrooms", "reception_rooms", "property_type", "property_type_std",
    "size_sqft", "size_sqm", "furnished", "epc_rating",
    "floorplan_url", "room_details",
    "has_basement", "has_lower_ground", "has_ground", "has_mezzanine",
    "has_first_floor", "has_second_floor", "has_third_floor",
    "has_fourth_plus", "has_roof_terrace", "floor_count", "property_levels",
    "let_agreed", "let_type", "is_short_let",
    "agent_name", "agent_phone", "agent_brand",
    "summary", "description", "features",
    "added_date", "scraped_at",
    "postcode_normalized", "postcode_inferred",
    "address_fingerprint", "canonical_id",
    "first_seen", "last_seen", "is_active", "price_change_count",
]

PRICE_HISTORY_COLUMNS = ["id", "listing_id", "price_pcm", "price_pw", "recorded_at"]

SCRAPE_RUNS_COLUMNS = [
    "id", "run_id", "spider_name", "started_at", "finished_at",
    "duration_seconds", "status", "items_scraped", "items_new", "items_updated",
    "items_dropped", "items_errors", "request_count", "response_count",
    "response_bytes", "error_count", "retry_count",
    "memory_start_mb", "memory_peak_mb", "memory_end_mb",
    "log_file", "exit_reason", "error_summary",
]

BATCH = 500


def log(msg: str) -> None:
    print(f"[sync] {msg}", flush=True)


def get_postgres_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url
    for var in ("POSTGRES_URL", "DATABASE_URL", "POSTGRES_URL_NON_POOLING"):
        url = os.environ.get(var)
        if url:
            return url
    sys.exit(
        "No Postgres URL. Pass --postgres-url or set POSTGRES_URL.\n"
        "  (prod value lives in dashboard/.env.prod)"
    )


def open_sqlite_copy(src: str) -> tuple[sqlite3.Connection, str]:
    """Copy the canonical DB to a temp file and open the copy read-only.

    Per DATA_LAYER_CONTRACT.md §8: validate/read a COPY, never the live file.
    """
    if not os.path.exists(src):
        sys.exit(f"SQLite DB not found: {src}")
    tmp_dir = tempfile.mkdtemp(prefix="rentals_sync_")
    tmp_db = os.path.join(tmp_dir, "rentals_copy.db")
    shutil.copy2(src, tmp_db)
    # Copy WAL/SHM siblings if present so the copy reflects committed + WAL state.
    for ext in ("-wal", "-shm"):
        sib = src + ext
        if os.path.exists(sib):
            shutil.copy2(sib, tmp_db + ext)
    conn = sqlite3.connect(tmp_db)
    conn.row_factory = sqlite3.Row
    return conn, tmp_dir


def sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def ensure_pg_parity(pg_cur) -> list[str]:
    """Add the two parity columns the canonical DB has but init-db may lack.

    DATA_LAYER_CONTRACT.md §3 drift items #1 and #2. Idempotent.
    """
    actions = []
    pg_cur.execute("ALTER TABLE listings ADD COLUMN IF NOT EXISTS canonical_id INTEGER")
    actions.append("listings.canonical_id ensured")
    pg_cur.execute("ALTER TABLE price_history ADD COLUMN IF NOT EXISTS price_pw INTEGER")
    actions.append("price_history.price_pw ensured")
    return actions


def pg_table_exists(pg_cur, table: str) -> bool:
    pg_cur.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name=%s)",
        (table,),
    )
    return pg_cur.fetchone()[0]


def load_table(pg_cur, sqlite_conn, table: str, columns: list[str]) -> int:
    """TRUNCATE + bulk-insert `table` from SQLite. Returns rows inserted."""
    # Only sync columns that actually exist in the SQLite source (defensive).
    src_cols = sqlite_columns(sqlite_conn, table)
    cols = [c for c in columns if c in src_cols]
    missing = [c for c in columns if c not in src_cols]
    if missing:
        log(f"  {table}: SQLite missing {missing} — syncing {len(cols)} cols")

    col_list = ", ".join(cols)
    rows = sqlite_conn.execute(f"SELECT {col_list} FROM {table}").fetchall()
    if not rows:
        log(f"  {table}: 0 source rows — skipping")
        return 0

    # RESTART IDENTITY so SERIAL sequences reset; CASCADE because price_history
    # FKs listings. Order of load (listings first) keeps FKs satisfiable.
    pg_cur.execute(f"TRUNCATE {table} RESTART IDENTITY CASCADE")

    insert_sql = f"INSERT INTO {table} ({col_list}) VALUES %s"
    payload = [tuple(r[c] for c in cols) for r in rows]
    for i in range(0, len(payload), BATCH):
        execute_values(pg_cur, insert_sql, payload[i : i + BATCH], page_size=BATCH)

    # Re-sync the id sequence to MAX(id) so future inserts don't collide.
    pg_cur.execute(
        f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
        f"COALESCE((SELECT MAX(id) FROM {table}), 1), true)"
    )
    return len(payload)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sqlite", default=DEFAULT_SQLITE, help="Path to canonical SQLite DB")
    ap.add_argument("--postgres-url", default=None, help="Override POSTGRES_URL")
    ap.add_argument("--execute", action="store_true", help="ACTUALLY write to Postgres (GATED)")
    ap.add_argument(
        "--i-have-rotated-the-secret",
        action="store_true",
        help="Required with --execute. Confirms the POSTGRES_URL secret was rotated first.",
    )
    args = ap.parse_args()

    # ---- Gate: --execute requires the rotation confirmation flag ----
    if args.execute and not args.i_have_rotated_the_secret:
        sys.exit(
            "REFUSING to write prod. --execute requires --i-have-rotated-the-secret.\n"
            "The lead runs the real load ONLY after rotating the POSTGRES_URL secret."
        )

    dry_run = not args.execute
    mode = "DRY-RUN (no writes)" if dry_run else "EXECUTE (writing prod)"
    log(f"mode: {mode}")
    log(f"sqlite source: {args.sqlite}")

    pg_url = get_postgres_url(args.postgres_url)
    # Redact creds in any echo of the URL.
    safe_url = pg_url.split("@")[-1] if "@" in pg_url else pg_url
    log(f"postgres target: ...@{safe_url[:60]}")

    sqlite_conn, tmp_dir = open_sqlite_copy(args.sqlite)
    try:
        # Source counts
        src_counts = {}
        for t in ("listings", "price_history", "scrape_runs"):
            try:
                src_counts[t] = sqlite_conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except sqlite3.OperationalError:
                src_counts[t] = None
        log(f"source rows: {src_counts}")

        pg_conn = psycopg2.connect(pg_url)
        pg_conn.autocommit = False
        started = time.time()
        try:
            with pg_conn.cursor() as cur:
                # Verify the schema was initialized (init-db must have run first).
                for t in ("listings", "price_history"):
                    if not pg_table_exists(cur, t):
                        sys.exit(
                            f"Postgres table '{t}' does not exist. Run the dashboard "
                            f"init-db route first (GET /api/init-db), then re-run this sync."
                        )

                parity = ensure_pg_parity(cur)
                for a in parity:
                    log(f"  parity: {a}")

                # Before counts
                before = {}
                for t in ("listings", "price_history", "scrape_runs"):
                    if pg_table_exists(cur, t):
                        cur.execute(f"SELECT COUNT(*) FROM {t}")
                        before[t] = cur.fetchone()[0]
                log(f"postgres BEFORE: {before}")

                # Load in FK-safe order: listings -> price_history -> scrape_runs
                inserted = {}
                inserted["listings"] = load_table(cur, sqlite_conn, "listings", LISTINGS_COLUMNS)
                inserted["price_history"] = load_table(
                    cur, sqlite_conn, "price_history", PRICE_HISTORY_COLUMNS
                )
                if pg_table_exists(cur, "scrape_runs") and src_counts.get("scrape_runs"):
                    inserted["scrape_runs"] = load_table(
                        cur, sqlite_conn, "scrape_runs", SCRAPE_RUNS_COLUMNS
                    )

                # After counts (within the open txn, before commit/rollback)
                after = {}
                for t in inserted:
                    cur.execute(f"SELECT COUNT(*) FROM {t}")
                    after[t] = cur.fetchone()[0]
                log(f"postgres AFTER (in-txn): {after}")
                log(f"rows inserted: {inserted}")

                # Sanity: active prime-central comps the dashboard needs
                cur.execute(
                    "SELECT COUNT(*) FROM listings WHERE is_active=1 AND size_sqft>0 "
                    "AND price_pcm>0 AND (postcode ~ '^SW1[A-Z]' OR postcode ~ '^SW3' "
                    "OR postcode ~ '^SW7' OR postcode ~ '^W1[A-Z]')"
                )
                comps = cur.fetchone()[0]
                log(f"sanity: active prime-central comps now queryable = {comps}")

            if dry_run:
                pg_conn.rollback()
                log("DRY-RUN: rolled back — NO changes written to prod.")
            else:
                pg_conn.commit()
                log(f"EXECUTE: committed in {time.time()-started:.1f}s.")
        finally:
            pg_conn.close()
    finally:
        sqlite_conn.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
