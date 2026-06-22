#!/usr/bin/env python3
"""
sync_sales_to_postgres.py — Load the for-SALE SQLite DB into prod Neon Postgres.

The for-sale vertical's standalone, safety-hardened prod-write sync. Mirrors
`output/sales.db` (the runner-side scrape output of the for-sale spiders) into the
Vercel/Neon Postgres `sale_listings` table that the /api/similar-sale route reads.

This is a SINGLE-TABLE near-clone of scripts/sync_sqlite_to_postgres.py (the rental
sync — left BYTE-UNCHANGED by contract). It reuses that script's full prod-write
safety machinery VERBATIM, scoped to the one `sale_listings` table:
  * DRY-RUN BY DEFAULT. Without --execute the script connects, ensures the schema,
    prints row-count deltas, but performs NO writes (rolls back).
  * --execute is GATED. It REFUSES unless --i-have-rotated-the-secret is ALSO passed,
    so an accidental run cannot write prod (the codebase's standard execute gate).
  * Reads a COPY of sales.db (never the live file) — open_sqlite_copy.
  * NON-DESTRUCTIVE UPSERT: INSERT ... ON CONFLICT (source, property_id) DO UPDATE.
    ADDS/UPDATES rows; NEVER deletes prod-only rows. NO TRUNCATE, NO setval.
  * Fail-closed full pg_dump backup (scripts/backup_neon.sh) + an in-txn JSON snapshot
    BEFORE any write; ABORTS+rolls back if the row count would shrink >5%. Idempotent.

The ONE addition vs the rental sync: it CREATEs the prod `sale_listings` table (+ its
indexes) idempotently — ensure_sale_table() — instead of refusing when the table is
absent. This is what makes /api/similar-sale light up on the first scheduled run (the
route graceful-empties on the missing table today). The prod-shaped DDL (BIGINT
asking_price, INTEGER flags, UNIQUE(source, property_id)) matches the canonical DDL the
sale route is proven against (dashboard/test/similar_sale_query_test.mjs) and passes
scripts/check_prod_schema_drift.py's SALE_ASSERTIONS.

Usage:
  # Dry run (default) — safe, no writes. Uses POSTGRES_URL from env.
  python3 scripts/sync_sales_to_postgres.py --sqlite output/sales.db

  # REAL prod load (GATED — the scheduled workflow / lead runs this):
  python3 scripts/sync_sales_to_postgres.py --sqlite output/sales.db --execute --i-have-rotated-the-secret
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# scripts/ is not a package; make the for_sale package importable when run as
# `python3 scripts/sync_sales_to_postgres.py`.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    sys.exit("psycopg2 is required: pip3 install psycopg2-binary")

# SINGLE-SOURCE the column list from the for-sale data layer (never re-list — prevents
# drift between the runner-side schema and the prod schema). SALE_COLUMNS has no `id`.
from for_sale.sale_data import SALE_COLUMNS

DEFAULT_SQLITE = os.path.join(PROJECT_ROOT, "output", "sales.db")

# Business unique key the UPSERT conflicts ON — the real-world identity of a sale row
# (NOT the surrogate SERIAL `id`). Single key for the single table.
CONFLICT_KEY = ["source", "property_id"]
# "first_seen" is set once on INSERT, never updated (CLAUDE.md contract); EXCLUDED could
# move it backward. last_seen DOES refresh (the cycle-relative freshness window needs it).
NEVER_UPDATE = {"first_seen"}
# Columns carrying a local SQLite row id — dropped so we never force a local id onto a
# populated prod table (Postgres owns its SERIAL). Sale has no canonical_id / no child
# history table, so this is simpler than the rental sync (no FK remap, no DO_NOTHING).
DROP_IDLIKE = {"id"}

# An UPSERT must NEVER reduce the table's row count by more than this fraction. A bigger
# shrink means something is wrong (bad/empty source, schema mismatch) — abort + roll back.
MAX_ROWCOUNT_DROP_FRACTION = 0.05  # >5% shrink = hard stop
BATCH = 500


def log(msg: str) -> None:
    print(f"[sale-sync] {msg}", flush=True)


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
    """Copy sales.db to a temp file and open the copy (never the live file)."""
    if not os.path.exists(src):
        sys.exit(f"SQLite DB not found: {src}")
    tmp_dir = tempfile.mkdtemp(prefix="sales_sync_")
    tmp_db = os.path.join(tmp_dir, "sales_copy.db")
    shutil.copy2(src, tmp_db)
    for ext in ("-wal", "-shm"):
        sib = src + ext
        if os.path.exists(sib):
            shutil.copy2(sib, tmp_db + ext)
    conn = sqlite3.connect(tmp_db)
    conn.row_factory = sqlite3.Row
    return conn, tmp_dir


def sqlite_columns(conn: sqlite3.Connection, t: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({t})")}


def pg_table_exists(c, t: str) -> bool:
    c.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name=%s)",
        (t,),
    )
    return c.fetchone()[0]


# The prod-shaped DDL. BIGINT asking_price (a £>2.147B asking price would overflow an
# INTEGER column / the saleSimilarQuery `asking_price::int` cast), INTEGER flags. Matches
# dashboard/test/similar_sale_query_test.mjs and passes check_prod_schema_drift SALE_ASSERTIONS.
# Every statement is CREATE ... IF NOT EXISTS → safe to run on EVERY scheduled invocation.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sale_listings (
    id                  SERIAL PRIMARY KEY,
    source              TEXT NOT NULL,
    property_id         TEXT NOT NULL,
    url                 TEXT,
    area                TEXT,
    asking_price        BIGINT,
    price_qualifier     TEXT,
    address             TEXT,
    postcode            TEXT,
    latitude            REAL,
    longitude           REAL,
    bedrooms            INTEGER,
    bathrooms           INTEGER,
    property_type       TEXT,
    size_sqft           INTEGER,
    is_new_build        INTEGER DEFAULT 0,
    is_under_offer      INTEGER DEFAULT 0,
    agent_name          TEXT,
    agent_phone         TEXT,
    summary             TEXT,
    added_date          TEXT,
    address_fingerprint TEXT,
    first_seen          TEXT,
    last_seen           TEXT,
    is_active           INTEGER DEFAULT 1,
    scraped_at          TEXT,
    UNIQUE(source, property_id)
)
"""

_CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_sale_source_prop  ON sale_listings(source, property_id)",
    "CREATE INDEX IF NOT EXISTS idx_sale_postcode     ON sale_listings(postcode)",
    "CREATE INDEX IF NOT EXISTS idx_sale_fingerprint  ON sale_listings(address_fingerprint)",
    "CREATE INDEX IF NOT EXISTS idx_sale_active       ON sale_listings(is_active)",
]


def ensure_sale_table(pg_cur) -> None:
    """Idempotently CREATE the prod sale_listings table + its 4 indexes (the ONE true
    addition vs the rental sync, which refuses when the table is absent). Safe on every
    scheduled run. Runs BEFORE counts/backup so the very first sync populates from zero.

    If a pre-existing prod table somehow lacks the UNIQUE(source, property_id) constraint
    (the ON CONFLICT target), add it under a pg_constraint catalog guard so the upsert
    always has a constraint to conflict on.
    """
    pg_cur.execute(_CREATE_TABLE_SQL)
    for idx_sql in _CREATE_INDEX_SQL:
        pg_cur.execute(idx_sql)
    # Belt-and-braces: guarantee the ON CONFLICT target exists even on a legacy table
    # that predates the UNIQUE(...) in the CREATE. ADD CONSTRAINT has no IF NOT EXISTS,
    # so guard with a catalog check (mirror ensure_pg_parity in the rental sync).
    pg_cur.execute(
        "SELECT 1 FROM pg_constraint c JOIN pg_class t ON c.conrelid = t.oid "
        "WHERE t.relname = 'sale_listings' AND c.contype = 'u'"
    )
    if not pg_cur.fetchone():
        pg_cur.execute(
            "ALTER TABLE sale_listings "
            "ADD CONSTRAINT sale_listings_source_property_uq UNIQUE (source, property_id)"
        )


def backup_prod(pg_cur, tables: list[str], out_path: str) -> dict:
    """Pre-write safety backup: dump every prod row of `tables` to a JSON file BEFORE any
    write, so a bad sync is always recoverable from disk (independent of Neon PITR).
    Returns {table: row_count}. Read-only; runs inside the txn."""
    backup = {}
    counts = {}
    for t in tables:
        if not pg_table_exists(pg_cur, t):
            continue
        pg_cur.execute(f"SELECT * FROM {t}")
        col_names = [d[0] for d in pg_cur.description]
        rows = pg_cur.fetchall()
        backup[t] = {"columns": col_names, "rows": [list(r) for r in rows]}
        counts[t] = len(rows)
    with open(out_path, "w") as f:
        json.dump(backup, f, default=str)
    return counts


def load_sale_table(pg_cur, sqlite_conn) -> dict:
    """NON-DESTRUCTIVE UPSERT of sale_listings from the local sales.db into Postgres.

    INSERTs new rows and UPDATEs existing ones keyed on UNIQUE(source, property_id); it
    NEVER deletes prod rows absent from the local snapshot. The surrogate `id` is dropped
    so we never force a local id onto a populated prod table. Returns {processed, table}.
    """
    src_cols = sqlite_columns(sqlite_conn, "sale_listings")
    cols = [c for c in SALE_COLUMNS if c in src_cols and c not in DROP_IDLIKE]
    missing = [c for c in SALE_COLUMNS if c not in src_cols]
    if missing:
        log(f"  sale_listings: sales.db missing {missing} — syncing {len(cols)} cols")

    # Refuse on EMPTY *or PARTIAL* conflict key. A partial key (e.g. only "source" when
    # the real constraint is UNIQUE(source, property_id)) makes ON CONFLICT error at
    # execute time ("no unique constraint matching"). Fail early + clearly — never emit a
    # write that targets the wrong/no constraint.
    conflict = [c for c in CONFLICT_KEY if c in cols]
    if conflict != CONFLICT_KEY:
        raise RuntimeError(
            f"sale_listings: conflict key {CONFLICT_KEY} not fully present in synced "
            f"columns {cols} (have {conflict}) — refusing to write (UPSERT would target "
            f"the wrong/no constraint). Fix the source schema."
        )

    update_cols = [c for c in cols if c not in conflict and c not in NEVER_UPDATE]
    set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
    action = f"DO UPDATE SET {set_clause}" if update_cols else "DO NOTHING"

    col_list = ", ".join(cols)
    conflict_list = ", ".join(conflict)
    insert_sql = (
        f"INSERT INTO sale_listings ({col_list}) VALUES %s "
        f"ON CONFLICT ({conflict_list}) {action}"
    )

    rows = sqlite_conn.execute(f"SELECT {col_list} FROM sale_listings").fetchall()
    if not rows:
        log("  sale_listings: 0 source rows — skipping")
        return {"processed": 0, "table": "sale_listings"}

    payload = [tuple(r[c] for c in cols) for r in rows]
    for i in range(0, len(payload), BATCH):
        execute_values(pg_cur, insert_sql, payload[i : i + BATCH], page_size=BATCH)

    # NOTE: no TRUNCATE, no setval. The prod SERIAL sequence is owned by prod; we never
    # reset it. Existing prod-only rows are PRESERVED (that's the whole point).
    return {"processed": len(payload), "table": "sale_listings"}


def run_guarded_upsert(pg_conn, pg_cur, sqlite_conn, backup_path: str) -> dict:
    """Run the guarded prod write within the caller's open transaction: count BEFORE,
    backup (if non-empty), UPSERT, count AFTER, then the >5% shrink delta-abort guard.

    On a shrink the WHOLE transaction is rolled back (pg_conn.rollback()) and the process
    exits non-zero with the backup path — a sync must NEVER reduce prod. Returns
    {before_n, after_n, processed, backup_path}. The caller commits on success.
    """
    pg_cur.execute("SELECT COUNT(*) FROM sale_listings")
    before_n = pg_cur.fetchone()[0]
    if before_n > 0:
        backup_prod(pg_cur, ["sale_listings"], backup_path)
        log(f"SAFETY BACKUP written: {backup_path} ({before_n} prod rows)")
    else:
        log("prod sale_listings is empty — no pre-write backup needed.")

    result = load_sale_table(pg_cur, sqlite_conn)

    pg_cur.execute("SELECT COUNT(*) FROM sale_listings")
    after_n = pg_cur.fetchone()[0]
    log(f"sale_listings BEFORE={before_n} AFTER={after_n} processed={result['processed']}")

    if before_n > 0 and after_n < before_n * (1 - MAX_ROWCOUNT_DROP_FRACTION):
        pg_conn.rollback()
        sys.exit(
            f"ABORTED + ROLLED BACK: sale_listings would shrink {before_n} -> {after_n} "
            f"(>{MAX_ROWCOUNT_DROP_FRACTION:.0%}). A sync must never reduce row count. "
            f"Backup at {backup_path}. Investigate the source DB before re-running."
        )
    return {
        "before_n": before_n,
        "after_n": after_n,
        "processed": result["processed"],
        "backup_path": backup_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sqlite", default=DEFAULT_SQLITE, help="Path to the for-sale SQLite DB")
    ap.add_argument("--postgres-url", default=None, help="Override POSTGRES_URL")
    ap.add_argument("--execute", action="store_true", help="ACTUALLY write to Postgres (GATED)")
    ap.add_argument(
        "--i-have-rotated-the-secret",
        action="store_true",
        help="Required with --execute. Confirms the prod-write was deliberately authorised.",
    )
    ap.add_argument(
        "--skip-pg-dump",
        action="store_true",
        help="Skip the fail-closed pg_dump backup (scripts/backup_neon.sh) before write. "
        "Only for environments without pg_dump; the in-txn JSON snapshot still runs.",
    )
    args = ap.parse_args()

    # ---- Gate: --execute requires the rotation confirmation flag ----
    if args.execute and not args.i_have_rotated_the_secret:
        sys.exit(
            "REFUSING to write prod. --execute requires --i-have-rotated-the-secret.\n"
            "The scheduled workflow / lead runs the real load only with both tokens."
        )

    dry_run = not args.execute
    mode = "DRY-RUN (no writes)" if dry_run else "EXECUTE (writing prod)"
    log(f"mode: {mode}")
    log(f"sqlite source: {args.sqlite}")

    pg_url = get_postgres_url(args.postgres_url)
    safe_url = pg_url.split("@")[-1] if "@" in pg_url else pg_url
    log(f"postgres target: ...@{safe_url[:60]}")

    # ---- FAIL-CLOSED full pg_dump backup before any prod write ----
    # scripts/backup_neon.sh exits non-zero if the dump fails/empty -> check=True raises ->
    # the load NEVER runs without a verified backup on disk. The whole-DB dump covers
    # sale_listings once it exists. Skipped on dry-run (no write to protect against).
    if not dry_run and not args.skip_pg_dump:
        backup_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup_neon.sh")
        if os.path.exists(backup_sh):
            log("running fail-closed pg_dump backup (scripts/backup_neon.sh) before write...")
            # pg_dump cannot use the Neon pgbouncer "-pooler" endpoint; strip it for the
            # dump (the sync's own writes still use the pooler URL).
            dump_url = pg_url.replace("-pooler.", ".")
            env = dict(os.environ, POSTGRES_URL=dump_url)
            try:
                subprocess.run(["bash", backup_sh], check=True, env=env, timeout=600)
            except subprocess.TimeoutExpired:
                sys.exit("REFUSING to write prod: pg_dump backup timed out (>600s). No write performed.")
            log("pg_dump backup OK — proceeding.")
        else:
            sys.exit(
                "REFUSING to write prod: scripts/backup_neon.sh not found and "
                "--skip-pg-dump not set. A full backup must precede any prod write."
            )

    sqlite_conn, tmp_dir = open_sqlite_copy(args.sqlite)
    try:
        try:
            src_n = sqlite_conn.execute("SELECT COUNT(*) FROM sale_listings").fetchone()[0]
        except sqlite3.OperationalError:
            src_n = None
        log(f"source sale_listings rows: {src_n}")

        pg_conn = psycopg2.connect(pg_url)
        pg_conn.autocommit = False
        started = time.time()
        try:
            with pg_conn.cursor() as cur:
                # The ONE addition vs the rental sync: CREATE the table (+ indexes)
                # idempotently rather than refusing when it is absent.
                ensure_sale_table(cur)

                backup_path = os.path.join(
                    tempfile.gettempdir(),
                    f"sale_prod_backup_{time.strftime('%Y%m%dT%H%M%S')}.json",
                )
                # Guarded write: count BEFORE -> backup -> UPSERT -> count AFTER ->
                # delta-abort (rolls back + exits on a >5% shrink).
                summary = run_guarded_upsert(pg_conn, cur, sqlite_conn, backup_path)

                # Post-write sanity: active sale comps the dashboard needs.
                cur.execute(
                    "SELECT COUNT(*) FROM sale_listings "
                    "WHERE is_active=1 AND asking_price>0 "
                    "AND (is_under_offer IS NULL OR is_under_offer=0)"
                )
                comps = cur.fetchone()[0]
                log(f"sanity: active priced (not under-offer) sale comps queryable = {comps}")
                log(f"summary: {summary}")

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
