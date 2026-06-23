#!/usr/bin/env python3
"""Guarded, reusable one-time cleanup: NULL economically-bad size_sqft values.

WHY: floorplan-OCR writers (now gated — see scripts/ocr_enrich.py.sqft_passes_sanity_gate)
historically persisted square-METRES magnitudes (sub-150) and max()-of-garbage (>10000)
as if they were a flat's size_sqft. Those become bad model inputs and absurd £/sqft on
the served /api/similar peers. This script NULLs them so they are treated as "unknown"
(re-OCR can recover the true value later from the preserved size_sqm / floorplan_url).

ECONOMICS-AWARE: a flat 10,000 cliff would wrongly NULL real prime-London mega-mansions
(verified in prod: e.g. 12,415 sqft / 8 bed / £150,000 pcm = £12.08/sqft — genuine). The
discriminator between a real mansion and OCR garbage above 10,000 is £/sqft/month: real
London is £3-30; the garbage rows are all < £3/sqft. So the bad-row predicate is:

    size_sqft < 150                                          # sqm-as-sqft / room dim
    OR size_sqft > 14000                                     # above any real London home
    OR (size_sqft > 10000 AND price_pcm > 0                  # in the 10k-14k mansion band
        AND price_pcm / size_sqft < 3.0)                     #   but uneconomic -> garbage

This KEEPS the £>=3/sqft mansions in [10000,14000] and NULLs the sub-£3/sqft garbage,
while still unconditionally NULLing everything <150 or >14000.

SAFETY CONTRACT:
  * DRY-RUN BY DEFAULT. Mutates only with explicit --execute.
  * NEVER deletes a row. Only sets size_sqft = NULL (and size_source = NULL IFF that
    column exists in the schema — the live rentals.db has no such column, so only
    size_sqft is nulled).
  * Does NOT touch size_sqm, floorplan_url, room_details — those let re-OCR recover.
  * Takes a timestamped file-copy backup of the DB BEFORE any --execute mutation.
  * Aborts if the candidate count exceeds --max-rows (default 500). The known bad set
    is ~222 sub-150 + a handful of >10000 garbage; a 10x blowup means a bad predicate.
  * Idempotent: a second run finds 0 candidates.

Usage:
    python3 scripts/null_bad_sqft.py --db output/rentals.db                 # dry-run, all sources
    python3 scripts/null_bad_sqft.py --db output/rentals.db --execute       # mutate (backup first)
    python3 scripts/null_bad_sqft.py --db output/rentals.db --source rightmove --execute

PROD (NOT run by this script — describable guarded path, see the FIX SPEC §5).
    backup_neon.sh ; then on Neon (parameterized; %(src)s optional, omit for all sources):
    UPDATE listings SET size_sqft = NULL
     WHERE size_sqft IS NOT NULL AND size_sqft > 0
       AND ( size_sqft < 150
             OR size_sqft > 14000
             OR ( size_sqft > 10000 AND price_pcm IS NOT NULL AND price_pcm > 0
                  AND price_pcm::float / size_sqft < 3.0 ) )
       AND ( %(src)s IS NULL OR source = %(src)s );
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Bounds — kept in sync with scripts/ocr_enrich.py SQFT_SANITY_MIN/MAX and the G5 gate.
# SQFT_MAX is the COARSE outer rail (14000) admitting real ~13,246 sqft mansions; the
# economic discrimination above 10,000 is by £/sqft (PPSF_MIN), matching the write-time
# gate's PPSF_MIN of 3.
SQFT_MIN = 150
SQFT_MAX = 14000
SQFT_ECON_FLOOR = 10000   # above this, a sub-£3/sqft reading is OCR garbage, not a mansion
PPSF_MIN = 3.0            # monthly £/sqft below which a >10k reading is uneconomic garbage
DEFAULT_MAX_ROWS = 500

# The clause that defines an economically-bad size_sqft. Bind nothing — pure SQL.
# Branch 1: below the absolute floor (sqm-as-sqft / room dim).
# Branch 2: above the coarse outer rail (no real London home is this big).
# Branch 3: in the 10k-14k mega-mansion band BUT uneconomic (£/sqft < 3) -> garbage.
#   CAST to REAL so the division is float (SQLite integer division would floor to 0).
_BAD_CLAUSE = (
    "size_sqft IS NOT NULL AND size_sqft > 0 "
    f"AND (size_sqft < {SQFT_MIN} "
    f"OR size_sqft > {SQFT_MAX} "
    f"OR (size_sqft > {SQFT_ECON_FLOOR} AND price_pcm IS NOT NULL AND price_pcm > 0 "
    f"AND CAST(price_pcm AS REAL) / size_sqft < {PPSF_MIN}))"
)


class CleanupAborted(Exception):
    """Raised when the candidate count exceeds the --max-rows safety cap."""


def _table_columns(conn: sqlite3.Connection) -> list[str]:
    return [r[1] for r in conn.execute("PRAGMA table_info(listings)").fetchall()]


def _source_column(cols: list[str]) -> str | None:
    """The live schema has no per-measurement source column. Some variants might name
    it size_source / sqft_source — null it too if present. Returns None if absent."""
    for c in ("size_source", "sqft_source", "size_sqft_source"):
        if c in cols:
            return c
    return None


def _backup_db(db_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = db_path.parent / "sqft_cleanup_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{db_path.stem}_pre_null_bad_sqft_{ts}{db_path.suffix}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def run_cleanup(
    db_path: str,
    execute: bool = False,
    source: str | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    verbose: bool = True,
) -> dict:
    """NULL out-of-range size_sqft. Returns a result dict.

    Keys: candidates, nulled, executed, backup_path, per_source, before_total, after_total.
    Raises CleanupAborted if candidates > max_rows (before any mutation).
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    where = f"WHERE {_BAD_CLAUSE}"
    params: list = []
    if source:
        where += " AND source = ?"
        params.append(source)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        cols = _table_columns(conn)
        src_col = _source_column(cols)

        before_total = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]

        # n_insane BEFORE (rightmove-only, the G5 metric) — for operator visibility.
        n_insane_before = conn.execute(
            f"SELECT COUNT(*) FROM listings WHERE source='rightmove' AND {_BAD_CLAUSE}"
        ).fetchone()[0]

        candidate_rows = conn.execute(
            f"SELECT id, source, size_sqft, bedrooms, price_pcm FROM listings {where} "
            "ORDER BY id",
            params,
        ).fetchall()
        candidates = len(candidate_rows)

        # Per-source report.
        per_source: dict[str, int] = {}
        for r in candidate_rows:
            per_source[r["source"]] = per_source.get(r["source"], 0) + 1

        if verbose:
            print("=" * 70)
            print("NULL BAD SQFT — economics-aware (<150, >14000, or >10000 & £/sqft<3) cleanup")
            print("=" * 70)
            print(f"[DB]        {path}")
            print(f"[SOURCE]    {source or 'ALL (reporting per-source)'}")
            print(f"[MODE]      {'EXECUTE' if execute else 'DRY-RUN (no mutation)'}")
            print(f"[PREDICATE] {_BAD_CLAUSE}")
            print(f"[CANDIDATES] {candidates} row(s)")
            print(f"[n_insane rightmove BEFORE] {n_insane_before}")
            if src_col:
                print(f"[source-col] will also NULL '{src_col}'")
            else:
                print("[source-col] none in schema -> nulling size_sqft only")
            print("-" * 70)
            for r in candidate_rows:
                print(
                    f"  id={r['id']:<7} {r['source']:<11} size_sqft={r['size_sqft']:<7} "
                    f"beds={r['bedrooms']} price_pcm={r['price_pcm']}"
                )
            print("-" * 70)
            print("[BY SOURCE] " + (
                ", ".join(f"{s}={n}" for s, n in sorted(per_source.items(), key=lambda x: -x[1]))
                or "(none)"))

        # Safety cap — check BEFORE mutating.
        if candidates > max_rows:
            raise CleanupAborted(
                f"candidate count {candidates} exceeds --max-rows {max_rows}; "
                "refusing to mutate (suspect predicate)"
            )

        result = {
            "candidates": candidates,
            "nulled": 0,
            "executed": False,
            "backup_path": None,
            "per_source": per_source,
            "before_total": before_total,
            "after_total": before_total,
            "n_insane_before": n_insane_before,
            "source_column": src_col,
        }

        if not execute:
            if verbose:
                print(f"\n[DRY-RUN] Would NULL {candidates} row(s). Re-run with --execute to apply.")
            return result

        if candidates == 0:
            if verbose:
                print("\n[EXECUTE] 0 candidates — nothing to do (idempotent).")
            result["executed"] = True
            return result

        # Backup BEFORE mutating.
        backup_path = _backup_db(path)
        result["backup_path"] = str(backup_path)
        if verbose:
            print(f"\n[BACKUP] {backup_path}")

        ids = [r["id"] for r in candidate_rows]
        placeholders = ",".join("?" for _ in ids)
        set_clause = "size_sqft = NULL"
        if src_col:
            set_clause += f", {src_col} = NULL"
        cur = conn.execute(
            f"UPDATE listings SET {set_clause} WHERE id IN ({placeholders})", ids
        )
        conn.commit()
        result["nulled"] = cur.rowcount
        result["executed"] = True

        after_total = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        result["after_total"] = after_total
        n_insane_after = conn.execute(
            f"SELECT COUNT(*) FROM listings WHERE source='rightmove' AND {_BAD_CLAUSE}"
        ).fetchone()[0]
        result["n_insane_after"] = n_insane_after

        if verbose:
            print(f"[EXECUTE] NULLed size_sqft on {result['nulled']} row(s).")
            print(f"[ROWS]    total {before_total} -> {after_total} (must be EQUAL — no deletions)")
            print(f"[n_insane rightmove] {n_insane_before} -> {n_insane_after}")
            if before_total != after_total:
                print("[!!] ROW COUNT CHANGED — investigate (cleanup must never delete).")
        return result
    finally:
        conn.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="NULL economically-bad size_sqft (<150, >14000, or >10000 & £/sqft<3) (guarded).")
    ap.add_argument("--db", default="output/rentals.db", help="SQLite DB path")
    ap.add_argument("--source", default=None, help="restrict to one source (default: all)")
    ap.add_argument("--execute", action="store_true", help="apply the NULLs (default: dry-run)")
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                    help=f"abort if candidates exceed this (default {DEFAULT_MAX_ROWS})")
    args = ap.parse_args(argv)
    try:
        res = run_cleanup(args.db, execute=args.execute, source=args.source,
                          max_rows=args.max_rows)
    except CleanupAborted as e:
        print(f"\n[ABORTED] {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        return 1
    if args.execute and res["before_total"] != res["after_total"]:
        return 3  # row-count changed: hard failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
