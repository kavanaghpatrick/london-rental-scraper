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
  * NON-DESTRUCTIVE UPSERT (since 2026-06-16): INSERT ... ON CONFLICT DO UPDATE/
    NOTHING keyed on each table's business unique key. ADDS/UPDATES rows; NEVER
    deletes prod-only rows. (Replaces the old TRUNCATE+reload, which destroyed
    ~12,269 real prod listings — see NEON_RECOVERY_RUNBOOK.md / the incident.)
  * Safety: takes a full pg_dump backup (scripts/backup_neon.sh, fail-closed) +
    an in-txn JSON snapshot BEFORE any write, and ABORTS+rolls back if a table's
    row count would shrink >5% (a sync must never reduce prod). Idempotent.
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
import json
import os
import shutil
import sqlite3
import subprocess
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
    """Add the parity columns + the price_history unique key the UPSERT needs.

    Columns: DATA_LAYER_CONTRACT.md §3 drift items #1 and #2. Idempotent.
    Constraint: price_history has no natural unique key in init-db (only id PK),
    so the non-destructive UPSERT (ON CONFLICT (listing_id, recorded_at) DO
    NOTHING) needs one. We add it idempotently here — it also de-dupes history.
    """
    actions = []
    pg_cur.execute("ALTER TABLE listings ADD COLUMN IF NOT EXISTS canonical_id INTEGER")
    actions.append("listings.canonical_id ensured")
    pg_cur.execute("ALTER TABLE price_history ADD COLUMN IF NOT EXISTS price_pw INTEGER")
    actions.append("price_history.price_pw ensured")
    # ADD CONSTRAINT has no IF NOT EXISTS — guard with a catalog check. Required
    # for the price_history ON CONFLICT target. De-dupe first so the constraint
    # can be created even if duplicate (listing_id, recorded_at) pairs exist.
    pg_cur.execute(
        "SELECT 1 FROM pg_constraint WHERE conname = 'price_history_listing_recorded_uq'"
    )
    if not pg_cur.fetchone():
        # Remove any pre-existing duplicate snapshots (keep lowest id) so the
        # UNIQUE constraint can be added without error.
        pg_cur.execute(
            "DELETE FROM price_history a USING price_history b "
            "WHERE a.id > b.id AND a.listing_id = b.listing_id "
            "AND a.recorded_at IS NOT DISTINCT FROM b.recorded_at"
        )
        pg_cur.execute(
            "ALTER TABLE price_history ADD CONSTRAINT price_history_listing_recorded_uq "
            "UNIQUE (listing_id, recorded_at)"
        )
        actions.append("price_history (listing_id, recorded_at) UNIQUE added")
    else:
        actions.append("price_history unique key present")
    return actions


def pg_table_exists(pg_cur, table: str) -> bool:
    pg_cur.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name=%s)",
        (table,),
    )
    return pg_cur.fetchone()[0]


# Safety guard: an UPSERT should never REDUCE a table's row count. If post-write
# counts are below pre-write by more than this fraction, something is wrong
# (e.g. a wrong/empty source, a schema mismatch) — abort and roll back.
MAX_ROWCOUNT_DROP_FRACTION = 0.05  # >5% shrink = hard stop


def backup_prod(pg_cur, tables: list[str], out_path: str) -> dict:
    """Pre-write safety backup: dump every prod row of `tables` to a JSON file
    BEFORE any write, so a bad sync is always recoverable from disk (independent
    of Neon PITR). Returns {table: row_count}. Read-only; runs inside the txn."""
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


# Business unique keys per table — what an UPSERT conflicts ON. These are the
# real-world identity of a row (NOT the surrogate SERIAL `id`), so a sync MATCHES
# existing prod rows instead of duplicating or clobbering them.
#   listings:      UNIQUE(source, property_id)            -> DO UPDATE (refresh fields)
#   price_history: one snapshot per (listing_id, recorded_at) -> DO NOTHING (append-only history; never overwrite)
#   scrape_runs:   UNIQUE(run_id, spider_name)            -> DO UPDATE (refresh telemetry)
CONFLICT_KEYS = {
    "listings": ["source", "property_id"],
    "price_history": ["listing_id", "recorded_at"],
    "scrape_runs": ["run_id", "spider_name"],
}
# Tables where a conflict should UPDATE the existing row vs leave it untouched.
DO_NOTHING_ON_CONFLICT = {"price_history"}  # history is append-only — never rewrite a past snapshot


def load_table(pg_cur, sqlite_conn, table: str, columns: list[str]) -> dict:
    """NON-DESTRUCTIVE UPSERT of `table` from SQLite into Postgres.

    Replaces the old TRUNCATE+reload (which destroyed prod-only rows — the
    2026-06-16 incident). This INSERTs new rows and UPDATEs existing ones keyed
    on the business unique key; it NEVER deletes prod rows that aren't in the
    local snapshot. The surrogate `id` is dropped from the column set so we never
    force a local id onto a populated prod table (which would collide with or
    overwrite prod-only rows). Returns {processed, table}.
    """
    src_cols = sqlite_columns(sqlite_conn, table)
    # Drop columns that hold SQLITE row ids — under UPSERT, Postgres assigns its
    # own SERIAL ids, so any column carrying a local id would be WRONG in prod:
    #   - "id":          surrogate PK; let Postgres keep/assign its own.
    #   - "canonical_id" (listings): a self-reference to another listings.id. Copying
    #     the SQLite value would point at the wrong/nonexistent prod listing. We DROP
    #     it (→ NULL in prod) rather than corrupt the self-FK; dataeng re-runs dedupe
    #     to repopulate it against prod ids. (price_history.listing_id is handled by
    #     an explicit remap below, not dropped, because the FK is NOT NULL.)
    DROP_IDLIKE = {"id", "canonical_id"}
    cols = [c for c in columns if c in src_cols and c not in DROP_IDLIKE]
    missing = [c for c in columns if c not in src_cols]
    if missing:
        log(f"  {table}: SQLite missing {missing} — syncing {len(cols)} cols")

    conflict_spec = CONFLICT_KEYS.get(table, [])
    conflict = [c for c in conflict_spec if c in cols]
    # Refuse on EMPTY *or PARTIAL* conflict key — a partial key (e.g. only "source"
    # when the real constraint is UNIQUE(source, property_id)) makes ON CONFLICT
    # error at execute time ("no unique constraint matching"). Fail early + clearly.
    if conflict_spec and conflict != conflict_spec:
        raise RuntimeError(
            f"{table}: conflict key {conflict_spec} not fully present in synced "
            f"columns {cols} (have {conflict}) — refusing to write (UPSERT would "
            f"target the wrong/no constraint). Fix the source schema."
        )
    if not conflict:
        # No usable conflict key -> we cannot safely upsert; refuse rather than
        # risk duplicates or a destructive fallback.
        raise RuntimeError(
            f"{table}: no conflict key in synced columns {cols} — refusing to "
            f"write (would risk duplicates). Conflict key needed: {CONFLICT_KEYS.get(table)}"
        )

    col_list = ", ".join(cols)
    rows = sqlite_conn.execute(f"SELECT {col_list} FROM {table}").fetchall()
    if not rows:
        log(f"  {table}: 0 source rows — skipping")
        return {"processed": 0, "table": table}

    conflict_list = ", ".join(conflict)
    if table in DO_NOTHING_ON_CONFLICT:
        action = "DO NOTHING"
    else:
        # Never overwrite "first_seen" on an existing row — the contract is
        # "set once on INSERT, never updated" (CLAUDE.md). EXCLUDED.first_seen
        # could move it backward. All other non-key columns refresh from local.
        NEVER_UPDATE = {"first_seen"}
        update_cols = [c for c in cols if c not in conflict and c not in NEVER_UPDATE]
        set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
        action = f"DO UPDATE SET {set_clause}" if update_cols else "DO NOTHING"

    insert_sql = (
        f"INSERT INTO {table} ({col_list}) VALUES %s "
        f"ON CONFLICT ({conflict_list}) {action}"
    )

    if table == "price_history":
        # CRITICAL FK REMAP: price_history.listing_id in SQLite references SQLITE
        # listing ids, which do NOT match Postgres ids under UPSERT (we no longer
        # force local ids). Remap each SQLite listing_id -> its (source,property_id)
        # -> the Postgres listing id. Rows whose listing can't be resolved (orphans
        # or listings not yet loaded) are SKIPPED rather than violating the FK.
        # (The old TRUNCATE+preserve-id path hid this; UPSERT requires the remap.)
        sl_id_to_key = {
            r["id"]: (r["source"], r["property_id"])
            for r in sqlite_conn.execute("SELECT id, source, property_id FROM listings")
        }
        pg_cur.execute("SELECT source, property_id, id FROM listings")
        key_to_pg_id = {(s, p): i for (s, p, i) in pg_cur.fetchall()}
        lid_idx = cols.index("listing_id")
        remapped, skipped = [], 0
        for r in rows:
            key = sl_id_to_key.get(r["listing_id"])
            pg_id = key_to_pg_id.get(key) if key else None
            if pg_id is None:
                skipped += 1
                continue
            vals = [r[c] for c in cols]
            vals[lid_idx] = pg_id
            remapped.append(tuple(vals))
        payload = remapped
        if skipped:
            log(f"  price_history: skipped {skipped} rows with unresolvable listing_id (orphans)")
    else:
        payload = [tuple(r[c] for c in cols) for r in rows]

    for i in range(0, len(payload), BATCH):
        execute_values(pg_cur, insert_sql, payload[i : i + BATCH], page_size=BATCH)

    # NOTE: no TRUNCATE, no setval. The SERIAL sequence is owned by prod; we never
    # reset it. Existing prod-only rows are PRESERVED (that's the whole point).
    return {"processed": len(payload), "table": table}


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

    # ---- FAIL-CLOSED full pg_dump backup before any prod write (security #38) ----
    # scripts/backup_neon.sh exits non-zero if the dump fails/empty → check=True
    # raises → the load NEVER runs without a verified backup on disk. This is the
    # authoritative, restorable backup (the in-txn JSON snapshot below is a second
    # belt-and-braces net). Skipped on dry-run (no write to protect against).
    if not dry_run and not args.skip_pg_dump:
        backup_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup_neon.sh")
        if os.path.exists(backup_sh):
            log("running fail-closed pg_dump backup (scripts/backup_neon.sh) before write...")
            # pg_dump CANNOT use the Neon pgbouncer "-pooler" endpoint (transaction
            # pooling drops the session state pg_dump relies on). Strip "-pooler"
            # so the dump hits the DIRECT endpoint (Neon's documented fix); the
            # sync's own writes still use the pooler pg_url — only the dump reroutes.
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
                            f"init-db route first (POST /api/init-db), then re-run this sync."
                        )

                parity = ensure_pg_parity(cur)
                for a in parity:
                    log(f"  parity: {a}")

                sync_tables = ("listings", "price_history", "scrape_runs")

                # Before counts
                before = {}
                for t in sync_tables:
                    if pg_table_exists(cur, t):
                        cur.execute(f"SELECT COUNT(*) FROM {t}")
                        before[t] = cur.fetchone()[0]
                log(f"postgres BEFORE: {before}")

                # ---- PRE-WRITE SAFETY BACKUP (always, even dry-run) ----
                # Dump prod to disk BEFORE touching it, so a bad sync is
                # recoverable independent of Neon PITR (the 2026-06-16 incident
                # had no such backup). Skipped only if prod is empty.
                backup_path = os.path.join(
                    tempfile.gettempdir(),
                    f"prod_backup_{time.strftime('%Y%m%dT%H%M%S')}.json",
                )
                if any(before.values()):
                    bcounts = backup_prod(cur, list(before.keys()), backup_path)
                    log(f"SAFETY BACKUP written: {backup_path} (rows: {bcounts})")
                else:
                    log("prod is empty — no pre-write backup needed.")

                # Load in FK-safe order: listings -> price_history -> scrape_runs
                inserted = {}
                inserted["listings"] = load_table(cur, sqlite_conn, "listings", LISTINGS_COLUMNS)["processed"]
                inserted["price_history"] = load_table(
                    cur, sqlite_conn, "price_history", PRICE_HISTORY_COLUMNS
                )["processed"]
                if pg_table_exists(cur, "scrape_runs") and src_counts.get("scrape_runs"):
                    inserted["scrape_runs"] = load_table(
                        cur, sqlite_conn, "scrape_runs", SCRAPE_RUNS_COLUMNS
                    )["processed"]

                # After counts (within the open txn, before commit/rollback)
                after = {}
                for t in inserted:
                    cur.execute(f"SELECT COUNT(*) FROM {t}")
                    after[t] = cur.fetchone()[0]
                log(f"postgres AFTER (in-txn): {after}")
                log(f"rows processed (upserted): {inserted}")

                # ---- ROW-COUNT DELTA SAFETY GUARD ----
                # A non-destructive UPSERT must NEVER reduce a table's row count.
                # If it did, something is wrong (bad source / schema) — abort.
                for t, after_n in after.items():
                    before_n = before.get(t, 0)
                    if before_n > 0 and after_n < before_n * (1 - MAX_ROWCOUNT_DROP_FRACTION):
                        pg_conn.rollback()
                        sys.exit(
                            f"ABORTED + ROLLED BACK: {t} would shrink {before_n} -> "
                            f"{after_n} (>{MAX_ROWCOUNT_DROP_FRACTION:.0%}). A sync must "
                            f"never reduce row count. Backup at {backup_path}. "
                            f"Investigate the source DB before re-running."
                        )

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
