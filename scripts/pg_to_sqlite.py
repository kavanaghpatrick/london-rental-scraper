#!/usr/bin/env python3
"""
pg_to_sqlite.py — Materialize the Postgres `listings` table into a local SQLite DB.

WHY (#9): the daily-scrape CI job writes scraped data to Postgres (`--postgres`),
but the canonical trainer `retrain_canonical.py` reads a SQLite `--db`. A fresh CI
checkout has no local SQLite (output/rentals.db is gitignored). This script bridges
that gap: it pulls `listings` (and price_history when present) out of Postgres into
a SQLite file that `retrain_canonical.py` can train from. Run it AFTER the scrape and
BEFORE the retrain step.

Only the columns the trainer reads need to be faithful; we copy the full set the
trainer + feature pipeline touch, including `is_short_let` (used in its WHERE).

Usage (CI):
    python3 scripts/pg_to_sqlite.py --out output/rentals.db        # uses POSTGRES_URL
    python3 scripts/pg_to_sqlite.py --out /tmp/ci.db --postgres-url "$POSTGRES_URL"
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

try:
    import psycopg2
except ImportError:
    sys.exit("psycopg2 is required: pip3 install psycopg2-binary")

# Columns retrain_canonical.py + engineer_features_v20 read. Kept as a superset so
# the trainer never hits a missing column. `is_short_let` and `price_pcm` are load-
# bearing for the trainer's WHERE/quality filters.
LISTINGS_COLUMNS = [
    "id", "source", "property_id",
    "bedrooms", "bathrooms", "size_sqft",
    "postcode", "area", "property_type", "address",
    "price_pcm", "latitude", "longitude", "features", "description",
    "property_type_std", "let_type", "postcode_normalized", "postcode_inferred",
    "agent_brand", "floor_count",
    "has_roof_terrace", "has_basement", "has_ground",
    "has_first_floor", "has_second_floor", "has_third_floor", "has_fourth_plus",
    "is_short_let", "is_active",
]


def get_pg_url(cli: str | None) -> str:
    if cli:
        return cli
    for var in ("POSTGRES_URL", "DATABASE_URL", "POSTGRES_URL_NON_POOLING"):
        if os.environ.get(var):
            return os.environ[var]
    sys.exit("No Postgres URL. Pass --postgres-url or set POSTGRES_URL.")


def existing_pg_columns(pg_cur, table: str) -> set[str]:
    pg_cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name=%s",
        (table,),
    )
    return {r[0] for r in pg_cur.fetchall()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/rentals.db", help="SQLite output path")
    ap.add_argument("--postgres-url", default=None)
    args = ap.parse_args()

    pg_url = get_pg_url(args.postgres_url)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()  # rebuild fresh; this SQLite is a disposable CI artifact

    pg = psycopg2.connect(pg_url)
    try:
        with pg.cursor() as cur:
            have = existing_pg_columns(cur, "listings")
            cols = [c for c in LISTINGS_COLUMNS if c in have]
            missing = [c for c in LISTINGS_COLUMNS if c not in have]
            if missing:
                print(f"[pg_to_sqlite] Postgres listings missing {missing} — copying {len(cols)} cols")

            cur.execute(f"SELECT {', '.join(cols)} FROM listings")
            rows = cur.fetchall()
            print(f"[pg_to_sqlite] pulled {len(rows)} listings from Postgres")

        sq = sqlite3.connect(str(out))
        try:
            coldefs = ", ".join(f"{c} {'INTEGER PRIMARY KEY' if c == 'id' else ''}".strip() for c in cols)
            sq.execute(f"CREATE TABLE listings ({coldefs})")
            placeholders = ", ".join(["?"] * len(cols))
            sq.executemany(
                f"INSERT INTO listings ({', '.join(cols)}) VALUES ({placeholders})",
                rows,
            )
            sq.commit()
            n = sq.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
            trainable = sq.execute(
                "SELECT COUNT(*) FROM listings WHERE size_sqft > 0 AND bedrooms IS NOT NULL "
                "AND price_pcm > 0 AND (is_short_let = 0 OR is_short_let IS NULL)"
            ).fetchone()[0]
            print(f"[pg_to_sqlite] wrote {n} rows to {out} ({trainable} trainable)")
        finally:
            sq.close()
    finally:
        pg.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
