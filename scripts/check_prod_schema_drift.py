#!/usr/bin/env python3
"""
check_prod_schema_drift.py — READ-ONLY prod-Postgres column-TYPE drift detector.

================================================================================
WHY THIS EXISTS — the blind spot the PR suite STRUCTURALLY cannot cover.
================================================================================
The dashboard's hot queries embed Postgres TYPE assumptions in raw SQL:

  * `WHERE is_active = 1`            (similarQuery.js, saleSimilarQuery.js, db.ts)
        -> assumes is_active is an INTEGER. If prod ever migrates the column to
           BOOLEAN, `is_active = 1` is a runtime ERROR in Postgres
           ("operator does not exist: boolean = integer") and /api/similar 500s.
  * `asking_price::int`             (saleSimilarQuery.js, saleDb.ts)
        -> assumes asking_price fits a 4-byte int. If prod stored it as BIGINT
           and a real listing exceeds 2,147,483,647 (£2.1bn — implausible) OR,
           far more realistically, the ::int cast silently truncates / the value
           overflows, the comp math is wrong. We assert the DECLARED type AND
           probe for any value that would overflow a signed int32.
  * `price_pcm::int / size_sqft::numeric` arithmetic (similar + ppsf stats)
        -> assumes both are numeric-castable. A TEXT migration would 500 these.
  * `is_under_offer = 0`            (saleSimilarQuery.js)
        -> integer assumption, same failure mode as is_active.

WHY PR CI CANNOT SEE THIS (the whole point of this scheduled job):
  ci.yml seeds its OWN ephemeral schema from dashboard/src/app/api/init-db/
  route.ts (and the for-sale path's for_sale/sale_data.py CREATE TABLE), where
  every column is declared INTEGER / REAL by construction. So the PR suite tests
  the SQL against types it itself just created — they ALWAYS agree. The REAL
  prod Neon schema was created long ago and is mutated out-of-band (manual ALTER
  TABLE, a Neon console change, a migration tool, a `@vercel/postgres` auto-init
  that diverged). NOTHING in the PR path ever connects to prod, so a prod-only
  type migration (INTEGER -> BOOLEAN, INTEGER -> BIGINT, a column renamed/dropped)
  is invisible until a live /api/similar request 500s in production. This job is
  the only thing that reads prod's information_schema and compares it to what the
  shipped SQL casts assume.

SAFETY / SCOPE:
  * READ-ONLY. Touches ONLY information_schema (catalog views) plus, for the
    asking_price overflow probe, a single bounded `SELECT ... LIMIT 1` against
    sale_listings. NO DDL, NO writes, NO transaction that mutates anything.
  * Connects via POSTGRES_URL (same resolution order as backup_neon.sh /
    sync_sqlite_to_postgres.py: POSTGRES_URL, DATABASE_URL,
    POSTGRES_URL_NON_POOLING). Strips the Vercel trailing-newline artifact.
  * GRACEFUL SKIP (exit 0) when no URL is present — e.g. a fork without the
    secret. Absence of a secret is NOT a drift signal; it must never open an
    issue or fail the scheduled job on forks.
  * Exit codes:  0 = OK or skipped (no URL).  1 = drift detected.  2 = could not
    connect / unexpected error (operational, NOT drift — caller can treat as a
    transient and not file a drift issue, or surface separately).

USAGE:
  POSTGRES_URL=... python3 scripts/check_prod_schema_drift.py
  python3 scripts/check_prod_schema_drift.py --url 'postgresql://...'
  python3 scripts/check_prod_schema_drift.py --json   # machine-readable report
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    import psycopg2
except ImportError:  # pragma: no cover - import guard
    sys.exit("psycopg2 is required: pip3 install psycopg2-binary")


# Signed 4-byte int range — the boundary `::int` casts respect. A value outside
# this overflows / is the BIGINT cast risk called out for asking_price.
INT32_MAX = 2_147_483_647
INT32_MIN = -2_147_483_648

# Postgres type names we treat as "integer-compatible" for an `= 1` / `= 0`
# int-literal comparison and an `::int` cast target. smallint/int/bigint all
# compare cleanly to an int literal; only the BIGINT *cast overflow* is a
# separate, value-level concern handled below.
INTEGER_TYPES = {"smallint", "integer", "bigint"}
# Types that survive `::int` / `::numeric` arithmetic without a 500. numeric/
# real/double are castable to int and participate in the ppsf division.
NUMERIC_CASTABLE_TYPES = {
    "smallint", "integer", "bigint", "numeric", "decimal", "real",
    "double precision",
}


# ---- the exact assertions, derived from the shipped SQL --------------------
#
# Each entry: (table, column, kind, why). `kind` selects the type rule:
#   "int_eq"        column is compared to an int literal (is_active = 1 etc.)
#                   -> declared type MUST be in INTEGER_TYPES (a BOOLEAN here is
#                      the INTEGER->BOOLEAN migration that breaks `= 1`).
#   "int_cast"      column is `::int`-cast in the SELECT (asking_price::int etc.)
#                   -> must be integer-compatible AND (if bigint) we run the
#                      overflow probe to flag any value > INT32_MAX.
#   "numeric_math"  column participates in `::numeric` division / ::int arithmetic
#                   (price_pcm/size_sqft ppsf) -> must be NUMERIC_CASTABLE_TYPES.
#   "exists"        column is read in SELECT/WHERE; type is flexible (cast to
#                   ::text or used as a string) -> presence is the only assertion.
#
# Sources of truth (read at implementation time):
#   dashboard/src/lib/similarQuery.js      (/api/similar — rental peers)
#   dashboard/src/lib/saleSimilarQuery.js  (/api/similar-sale — saleSimilar)
#   dashboard/src/lib/db.ts                (is_active = 1 portfolio/stats)
RENTAL_ASSERTIONS = [
    ("listings", "is_active", "int_eq",
     "similarQuery.js / db.ts: `WHERE is_active = 1` — int-literal compare; "
     "a BOOLEAN migration breaks it (operator boolean = integer)."),
    ("listings", "price_pcm", "numeric_math",
     "similarQuery.js: `price_pcm::int` + `price_pcm::numeric / size_sqft::numeric` ppsf."),
    ("listings", "size_sqft", "numeric_math",
     "similarQuery.js: `size_sqft::int` + ppsf denominator `size_sqft::numeric`."),
    ("listings", "bedrooms", "int_cast",
     "similarQuery.js: `bedrooms::int` + `bedrooms BETWEEN $12 AND $13`."),
    ("listings", "property_id", "exists",
     "similarQuery.js: `property_id != $16` exclude key."),
    ("listings", "postcode", "exists",
     "similarQuery.js: `SPLIT_PART(postcode, ' ', 1)` district gate."),
    ("listings", "source", "exists", "similarQuery.js: source-weighting CASE."),
    ("listings", "property_type", "exists", "similarQuery.js: LOWER(property_type) match."),
    ("listings", "address", "exists", "similarQuery.js: SELECTed for the card."),
    ("listings", "url", "exists", "similarQuery.js: SELECTed for the card."),
    ("listings", "last_seen", "exists",
     "similarQuery.js: `last_seen::text` — any text-castable type (timestamp/date/text)."),
    ("listings", "id", "exists", "similarQuery.js: SELECTed; PK."),
]

SALE_ASSERTIONS = [
    ("sale_listings", "is_active", "int_eq",
     "saleSimilarQuery.js / saleDb.ts: `WHERE is_active = 1` — int-literal compare; "
     "a BOOLEAN migration breaks it."),
    ("sale_listings", "is_under_offer", "int_eq",
     "saleSimilarQuery.js: `(is_under_offer IS NULL OR is_under_offer = 0)` SSTC filter — "
     "int-literal compare; a BOOLEAN migration breaks it."),
    ("sale_listings", "asking_price", "int_cast",
     "saleSimilarQuery.js / saleDb.ts: `asking_price::int`. If the column is BIGINT "
     "and a value exceeds int32 (2,147,483,647) the ::int cast overflows at runtime."),
    ("sale_listings", "size_sqft", "numeric_math",
     "saleSimilarQuery.js: `size_sqft::int` + ppsf denominator `size_sqft::numeric`."),
    ("sale_listings", "bedrooms", "int_cast",
     "saleSimilarQuery.js: `bedrooms::int` + `bedrooms BETWEEN $12 AND $13`."),
    ("sale_listings", "property_id", "exists",
     "saleSimilarQuery.js: `property_id != $16` exclude key."),
    ("sale_listings", "postcode", "exists",
     "saleSimilarQuery.js: `SPLIT_PART(postcode, ' ', 1)` district gate."),
    ("sale_listings", "source", "exists", "saleSimilarQuery.js: source-weighting CASE."),
    ("sale_listings", "property_type", "exists", "saleSimilarQuery.js: LOWER(property_type) match."),
    ("sale_listings", "address", "exists", "saleSimilarQuery.js: SELECTed for the card."),
    ("sale_listings", "url", "exists", "saleSimilarQuery.js: SELECTed for the card."),
    ("sale_listings", "last_seen", "exists",
     "saleSimilarQuery.js: `last_seen::text` — any text-castable type."),
    ("sale_listings", "id", "exists", "saleSimilarQuery.js: SELECTed; PK."),
]

ALL_ASSERTIONS = RENTAL_ASSERTIONS + SALE_ASSERTIONS


def log(msg: str) -> None:
    print(f"[schema-drift] {msg}", flush=True)


def resolve_url(cli_url: str | None) -> str | None:
    """Same precedence as backup_neon.sh / sync_sqlite_to_postgres.py."""
    if cli_url:
        return _clean_url(cli_url)
    for var in ("POSTGRES_URL", "DATABASE_URL", "POSTGRES_URL_NON_POOLING"):
        val = os.environ.get(var)
        if val:
            return _clean_url(val)
    return None


def _clean_url(u: str) -> str:
    """Strip wrapping quotes and the Vercel-export trailing '\\n' / CR/LF artifact."""
    u = u.strip()
    if u.startswith('"') and u.endswith('"'):
        u = u[1:-1]
    if u.endswith("\\n"):
        u = u[:-2]
    return u.replace("\n", "").replace("\r", "")


def fetch_columns(cur, table: str) -> dict[str, dict]:
    """Read information_schema.columns for one table (current schema search_path).

    Returns {column_name: {data_type, is_nullable, ...}}. READ-ONLY catalog read.
    Restricts to table_schema='public' (where Neon/Vercel create app tables) to
    avoid colliding with same-named catalog/temp tables.
    """
    cur.execute(
        """
        SELECT column_name, data_type, is_nullable, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = %s
          AND table_schema = 'public'
        """,
        (table,),
    )
    out: dict[str, dict] = {}
    for name, data_type, is_nullable, charlen in cur.fetchall():
        out[name] = {
            "data_type": data_type,
            "is_nullable": is_nullable,
            "char_max_len": charlen,
        }
    return out


def table_exists(cur, table: str) -> bool:
    cur.execute(
        """
        SELECT 1 FROM information_schema.tables
        WHERE table_name = %s AND table_schema = 'public'
        """,
        (table,),
    )
    return cur.fetchone() is not None


def probe_asking_price_overflow(cur) -> tuple[bool, int | None]:
    """Bounded read-only probe: does any sale_listings.asking_price exceed int32?

    Only meaningful when asking_price is BIGINT (an INTEGER column physically
    cannot hold a value > INT32_MAX). Returns (overflow_found, sample_value).
    Single-row, indexed-friendly, read-only SELECT.
    """
    cur.execute(
        "SELECT asking_price FROM sale_listings "
        "WHERE asking_price > %s OR asking_price < %s LIMIT 1",
        (INT32_MAX, INT32_MIN),
    )
    row = cur.fetchone()
    if row is None:
        return (False, None)
    return (True, int(row[0]))


def check(cur) -> list[dict]:
    """Run every assertion. Returns a list of failure dicts (empty == healthy)."""
    failures: list[dict] = []

    # Cache columns per table (one catalog read each).
    tables = {t for (t, _c, _k, _w) in ALL_ASSERTIONS}
    cols_by_table: dict[str, dict] = {}
    for t in tables:
        if not table_exists(cur, t):
            failures.append({
                "table": t,
                "column": "*",
                "kind": "table_missing",
                "detail": f"table `{t}` does not exist in prod (public schema). "
                          f"The /api/similar{'-sale' if t == 'sale_listings' else ''} "
                          f"query reads it; a missing table 500s the route.",
            })
            cols_by_table[t] = {}
        else:
            cols_by_table[t] = fetch_columns(cur, t)

    for table, column, kind, why in ALL_ASSERTIONS:
        cols = cols_by_table.get(table, {})
        if not cols and not table_exists(cur, table):
            # Table-missing already recorded once above; don't spam per column.
            continue
        info = cols.get(column)
        if info is None:
            failures.append({
                "table": table, "column": column, "kind": "column_missing",
                "detail": f"column `{table}.{column}` is absent in prod. {why}",
            })
            continue

        dtype = (info["data_type"] or "").lower()

        if kind == "int_eq":
            # `col = 1` / `col = 0`. boolean is the migration that breaks it.
            if dtype not in INTEGER_TYPES:
                failures.append({
                    "table": table, "column": column, "kind": "type_mismatch",
                    "actual": dtype, "expected": "integer (one of %s)" % sorted(INTEGER_TYPES),
                    "detail": f"`{table}.{column}` is `{dtype}`, but the shipped SQL "
                              f"compares it to an int literal. {why}",
                })

        elif kind == "int_cast":
            if dtype not in INTEGER_TYPES:
                failures.append({
                    "table": table, "column": column, "kind": "type_mismatch",
                    "actual": dtype, "expected": "integer-compatible",
                    "detail": f"`{table}.{column}` is `{dtype}`; the SQL does `::int`. {why}",
                })
            elif dtype == "bigint" and table == "sale_listings" and column == "asking_price":
                # Declared-bigint is allowed, but a value > int32 overflows ::int.
                try:
                    overflow, sample = probe_asking_price_overflow(cur)
                except Exception as e:  # pragma: no cover - prod-only path
                    log(f"asking_price overflow probe skipped (non-fatal): {e}")
                    overflow, sample = (False, None)
                if overflow:
                    failures.append({
                        "table": table, "column": column, "kind": "int32_overflow",
                        "actual": f"bigint value {sample}", "expected": "<= %d" % INT32_MAX,
                        "detail": f"`sale_listings.asking_price` is BIGINT and holds "
                                  f"{sample}, which exceeds int32 ({INT32_MAX}). The "
                                  f"shipped `asking_price::int` cast OVERFLOWS at runtime.",
                    })

        elif kind == "numeric_math":
            if dtype not in NUMERIC_CASTABLE_TYPES:
                failures.append({
                    "table": table, "column": column, "kind": "type_mismatch",
                    "actual": dtype, "expected": "numeric-castable (%s)" % sorted(NUMERIC_CASTABLE_TYPES),
                    "detail": f"`{table}.{column}` is `{dtype}`; the SQL does numeric "
                              f"arithmetic (`::int` / `::numeric` ppsf). {why}",
                })

        elif kind == "exists":
            # Presence asserted above; any type is acceptable (string / ::text use).
            pass

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="Explicit Postgres URL (else POSTGRES_URL/DATABASE_URL).")
    ap.add_argument("--json", action="store_true", help="Emit a machine-readable JSON report.")
    args = ap.parse_args()

    url = resolve_url(args.url)
    if not url:
        # GRACEFUL SKIP — no secret (fork / unconfigured). NOT a drift signal.
        log("No POSTGRES_URL/DATABASE_URL set — skipping prod schema-drift check "
            "(expected on forks). Exit 0.")
        if args.json:
            print(json.dumps({"status": "skipped", "reason": "no_postgres_url"}))
        return 0

    # Vercel/Neon pooled '-pooler' endpoint is fine for these tiny read-only
    # catalog queries (no session-state dependency like pg_dump has), so we do
    # NOT rewrite it here.
    try:
        conn = psycopg2.connect(url, connect_timeout=20)
    except Exception as e:
        log(f"could not connect to prod Postgres: {e}")
        if args.json:
            print(json.dumps({"status": "error", "reason": "connect_failed", "error": str(e)}))
        # Operational failure, NOT drift. Exit 2 so the workflow can distinguish.
        return 2

    try:
        conn.set_session(readonly=True, autocommit=True)  # belt-and-braces read-only
        with conn.cursor() as cur:
            failures = check(cur)
    except Exception as e:
        log(f"unexpected error during catalog read: {e}")
        if args.json:
            print(json.dumps({"status": "error", "reason": "query_failed", "error": str(e)}))
        return 2
    finally:
        conn.close()

    if args.json:
        print(json.dumps({
            "status": "drift" if failures else "ok",
            "checked": len(ALL_ASSERTIONS),
            "failures": failures,
        }, indent=2))

    log(f"checked {len(ALL_ASSERTIONS)} column-type assertions across "
        f"listings + sale_listings.")
    if failures:
        log(f"PROD SCHEMA DRIFT — {len(failures)} mismatch(es):")
        for f in failures:
            log(f"  DRIFT [{f.get('kind')}] {f['table']}.{f['column']}: {f['detail']}")
        log("These are PROD-ONLY type/shape divergences from what the shipped SQL "
            "casts assume. PR CI cannot see them (it seeds INTEGER/REAL schemas from "
            "init-db/sale_data and tests the SQL against types it just created).")
        return 1

    log("OK — all prod column types match the SQL cast assumptions. No drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
