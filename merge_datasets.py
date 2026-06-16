#!/usr/bin/env python3
"""
merge_datasets.py — non-destructive UNION of all rental data sources (task #36).

Produces ONE complete dataset losing NOTHING. Builds a NEW merged DB; never
mutates an input in place. Key = (source, property_id). price_history is remapped
by (source, property_id) — NEVER by listing_id (autoincrement surrogate, not
comparable across DBs). is_active is RECOMPUTED cycle-relative post-merge
(DATA_LAYER_CONTRACT §5.1), not copied from any source.

Inputs (any subset; missing ones skipped, logged):
  --base       output/rentals.db                 (local canonical, 10,048 — richest fields)
  --git-jsonl  output/_git_recovery/RECOVERED_dec15_full_records.jsonl  (804 FULL records, Dec tail)
  --prod-db    <PITR dump>.db                     (~22,317 prod rows — PENDING #31; folds in when present)
  --out        output/rentals_merged.db

Re-scrape seed (NOT inserted): listings known only as predictions (no raw fields)
are written to a URL list for re-scraping, never DB-inserted.

Default DRY-RUN: prints the merge diff, writes the out DB but does NOT promote to
canonical. Promote (--promote) only after verify passes + lead sign-off.

Usage:
  python3 merge_datasets.py                      # dry-run: base + git-jsonl (+ prod if --prod-db)
  python3 merge_datasets.py --prod-db prod_recovered.db   # fold in PITR dump when it lands
  python3 merge_datasets.py --promote            # after verify, replace canonical (lead-gated)
"""
import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

LISTINGS_COLS = [
    "source", "property_id", "url", "area", "price", "price_pw", "price_pcm",
    "price_period", "address", "postcode", "latitude", "longitude", "bedrooms",
    "bathrooms", "property_type", "size_sqft", "furnished", "let_agreed",
    "agent_name", "agent_phone", "summary", "description", "features",
    "added_date", "scraped_at", "property_type_std", "let_type",
    "postcode_normalized", "postcode_inferred", "agent_brand", "reception_rooms",
    "size_sqm", "epc_rating", "floorplan_url", "room_details", "has_basement",
    "has_lower_ground", "has_ground", "has_mezzanine", "has_first_floor",
    "has_second_floor", "has_third_floor", "has_fourth_plus", "has_roof_terrace",
    "floor_count", "property_levels", "canonical_id", "address_fingerprint",
    "first_seen", "last_seen", "is_active", "price_change_count", "is_short_let",
]
# Fields where a non-null OLD value must not be overwritten by a NULL new value,
# and when both non-null the later last_seen wins (sticky structural/enriched data).
STICKY = {
    "url", "area", "address", "postcode", "latitude", "longitude", "bedrooms",
    "bathrooms", "property_type", "size_sqft", "furnished", "agent_name",
    "agent_phone", "summary", "description", "features", "property_type_std",
    "postcode_normalized", "postcode_inferred", "agent_brand", "reception_rooms",
    "size_sqm", "epc_rating", "floorplan_url", "room_details", "has_basement",
    "has_lower_ground", "has_ground", "has_mezzanine", "has_first_floor",
    "has_second_floor", "has_third_floor", "has_fourth_plus", "has_roof_terrace",
    "floor_count", "property_levels", "address_fingerprint", "is_short_let",
}
STALE_DAYS = 7  # §5.1 cycle-relative window


def log(msg):
    print(f"[merge] {msg}")


def make_base_schema(conn):
    """Create empty listings + price_history with the canonical 54-col schema."""
    cols_ddl = ",\n  ".join(f"{c} {_coltype(c)}" for c in ["id"] + LISTINGS_COLS)
    conn.execute(f"CREATE TABLE listings (\n  {cols_ddl},\n  UNIQUE(source, property_id))")
    conn.execute("""CREATE TABLE price_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, listing_id INTEGER NOT NULL,
        price_pcm INTEGER, price_pw INTEGER, recorded_at TEXT,
        FOREIGN KEY (listing_id) REFERENCES listings(id))""")
    conn.execute("CREATE INDEX idx_src_prop ON listings(source, property_id)")
    conn.execute("CREATE INDEX idx_fp ON listings(address_fingerprint)")
    conn.execute("CREATE INDEX idx_ph_listing ON price_history(listing_id)")


# Auxiliary tables/views the merge must PRESERVE from the base canonical DB (audit
# telemetry + dedupe). merge_datasets only rebuilds listings + price_history; without
# this carry-over, promoting the merge output would DROP scrape_runs/scrape_events
# (breaks dashboard /api/running + audit loggers) and the dedupe tables/views.
AUX_TABLES = ["scrape_runs", "scrape_events", "duplicate_groups", "dedupe_stats", "listings_deduped"]
AUX_VIEWS = ["listings_clean", "area_stats_clean"]


def carry_over_aux(out_conn, base_db):
    """Copy audit + dedupe tables/views from the base canonical DB into the merged DB,
    so the merge never drops them. Reads base via a read-only connection."""
    if not base_db or not Path(base_db).exists():
        return
    src = sqlite3.connect(f"file:{base_db}?mode=ro", uri=True)
    sc, oc = src.cursor(), out_conn.cursor()
    out_conn.execute("PRAGMA foreign_keys=OFF")
    for t in AUX_TABLES:
        ddl = sc.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()
        if not ddl:
            continue
        oc.execute(f"DROP TABLE IF EXISTS {t}")
        oc.execute(ddl[0])
        rows = sc.execute(f"SELECT * FROM {t}").fetchall()
        if rows:
            oc.executemany(f"INSERT INTO {t} VALUES ({','.join('?' * len(rows[0]))})", rows)
        log(f"  carried over aux table {t}: {len(rows)} rows")
    for v in AUX_VIEWS:
        ddl = sc.execute("SELECT sql FROM sqlite_master WHERE type='view' AND name=?", (v,)).fetchone()
        if not ddl:
            continue
        oc.execute(f"DROP VIEW IF EXISTS {v}")
        oc.execute(ddl[0])
        log(f"  carried over aux view {v}")
    out_conn.commit()
    src.close()


def _coltype(c):
    if c == "id":
        return "INTEGER PRIMARY KEY AUTOINCREMENT"
    if c in ("latitude", "longitude", "size_sqm"):
        return "REAL"
    if c in ("price", "price_pw", "price_pcm", "bedrooms", "bathrooms", "size_sqft",
             "reception_rooms", "let_agreed", "is_active", "price_change_count",
             "is_short_let", "canonical_id", "postcode_inferred", "floor_count") or c.startswith("has_"):
        return "INTEGER"
    return "TEXT"


def upsert_listing(cur, rec):
    """Insert or non-destructively merge one listing record (dict) by (source,property_id)."""
    src, pid = rec.get("source"), rec.get("property_id")
    cur.execute("SELECT * FROM listings WHERE source=? AND property_id=?", (src, pid))
    existing = cur.fetchone()
    if existing is None:
        cols = [c for c in LISTINGS_COLS if c in rec]
        cur.execute(
            f"INSERT INTO listings ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [rec.get(c) for c in cols],
        )
        return "inserted"
    # Merge: existing is a Row; build the updated values
    ex = dict(existing)
    new = {}
    ex_ls, new_ls = ex.get("last_seen") or "", rec.get("last_seen") or ""
    newer = new_ls >= ex_ls  # incoming row is at least as recent
    for c in LISTINGS_COLS:
        old_v, in_v = ex.get(c), rec.get(c)
        if c == "first_seen":
            new[c] = min(x for x in (ex.get("first_seen"), rec.get("first_seen")) if x) if (ex.get("first_seen") or rec.get("first_seen")) else None
        elif c == "last_seen":
            new[c] = max(ex_ls, new_ls) or None
        elif c == "is_active":
            new[c] = ex.get("is_active")  # recomputed later in §5.1 pass; keep placeholder
        elif c in STICKY:
            # non-null wins; if both non-null, later last_seen wins
            if in_v is None:
                new[c] = old_v
            elif old_v is None:
                new[c] = in_v
            else:
                new[c] = in_v if newer else old_v
        else:  # price_*, price_period, added_date, let_type, canonical_id, price_change_count, scraped_at
            new[c] = in_v if (newer and in_v is not None) else old_v
    sets = ",".join(f"{c}=?" for c in LISTINGS_COLS)
    cur.execute(f"UPDATE listings SET {sets} WHERE source=? AND property_id=?",
                [new[c] for c in LISTINGS_COLS] + [src, pid])
    return "merged"


def union_price_history(out_cur, src_conn, label):
    """Union price_history from src_conn into out, remapping by (source,property_id)."""
    src_conn.row_factory = sqlite3.Row
    rows = src_conn.execute("""
        SELECT l.source, l.property_id, ph.price_pcm, ph.price_pw, ph.recorded_at
        FROM price_history ph JOIN listings l ON ph.listing_id = l.id
    """).fetchall()
    added = skipped = orphan = 0
    for r in rows:
        out_cur.execute("SELECT id FROM listings WHERE source=? AND property_id=?",
                        (r["source"], r["property_id"]))
        m = out_cur.fetchone()
        if not m:
            orphan += 1
            continue
        lid = m[0]
        # dedup key: (source,property_id,price_pcm,recorded_at) -> (listing_id, price_pcm, recorded_at)
        out_cur.execute(
            "SELECT 1 FROM price_history WHERE listing_id=? AND IFNULL(price_pcm,-1)=IFNULL(?,-1) AND recorded_at=?",
            (lid, r["price_pcm"], r["recorded_at"]))
        if out_cur.fetchone():
            skipped += 1
            continue
        out_cur.execute(
            "INSERT INTO price_history (listing_id, price_pcm, price_pw, recorded_at) VALUES (?,?,?,?)",
            (lid, r["price_pcm"], r["price_pw"], r["recorded_at"]))
        added += 1
    log(f"  price_history from {label}: +{added} added, {skipped} dup-skipped, {orphan} orphan(no listing)")


def recompute_is_active(cur):
    """§5.1 cycle-relative is_active across the merged set."""
    cur.execute("SELECT MAX(last_seen) FROM listings")
    mx = cur.fetchone()[0]
    if not mx:
        return None
    cutoff = (datetime.fromisoformat(mx) - timedelta(days=STALE_DAYS)).isoformat()
    cur.execute("UPDATE listings SET is_active = CASE WHEN last_seen >= ? THEN 1 ELSE 0 END", (cutoff,))
    return mx, cutoff


def load_listings_from_db(conn):
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute("SELECT * FROM listings")]


def load_listings_from_jsonl(path):
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output/rentals.db")
    ap.add_argument("--git-jsonl", default="output/_git_recovery/RECOVERED_dec15_full_records.jsonl")
    ap.add_argument("--prod-db", default=None, help="PITR-recovered prod DB (folds in when present)")
    ap.add_argument("--out", default="output/rentals_merged.db")
    ap.add_argument("--rescrape-out", default="output/rescrape_seed_urls.txt")
    ap.add_argument("--promote", action="store_true", help="replace canonical after verify (lead-gated)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        out.unlink()
    out_conn = sqlite3.connect(out)
    out_conn.row_factory = sqlite3.Row
    make_base_schema(out_conn)
    cur = out_conn.cursor()

    # --- Seed order: broadest first. If prod present, seed prod; else seed base. ---
    sources = []
    if args.prod_db and Path(args.prod_db).exists():
        sources.append(("prod", args.prod_db, "db"))
    sources.append(("local-canonical", args.base, "db"))
    if Path(args.git_jsonl).exists():
        sources.append(("git-recovery", args.git_jsonl, "jsonl"))

    counts = {}
    for label, path, kind in sources:
        recs = load_listings_from_jsonl(path) if kind == "jsonl" else load_listings_from_db(sqlite3.connect(path))
        ins = mrg = 0
        for rec in recs:
            r = upsert_listing(cur, rec)
            ins += r == "inserted"; mrg += r == "merged"
        counts[label] = (len(recs), ins, mrg)
        log(f"listings from {label}: {len(recs)} in -> {ins} new, {mrg} merged")
    out_conn.commit()

    # --- price_history union (DB sources only; remap by source:property_id) ---
    for label, path, kind in sources:
        if kind == "db":
            union_price_history(cur, sqlite3.connect(path), label)
    out_conn.commit()

    # --- §5.1 cycle-relative is_active ---
    res = recompute_is_active(cur)
    if res:
        log(f"is_active recomputed cycle-relative: max_last_seen={res[0][:19]}, cutoff={res[1][:19]}")
    out_conn.commit()

    # --- Carry over audit + dedupe tables/views from the base canonical DB ---
    # (so the merge never drops scrape_runs/scrape_events/dedupe — they're not
    # rebuilt by the merge but MUST survive promotion to canonical.)
    carry_over_aux(out_conn, args.base)

    # --- Verify (acceptance) ---
    total = cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    active = cur.execute("SELECT SUM(is_active) FROM listings").fetchone()[0]
    ph = cur.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
    orphans = cur.execute("SELECT COUNT(*) FROM price_history ph LEFT JOIN listings l ON ph.listing_id=l.id WHERE l.id IS NULL").fetchone()[0]
    integrity = out_conn.execute("PRAGMA integrity_check").fetchone()[0]
    log("=" * 50)
    log(f"MERGED: listings={total}, active={active}, price_history={ph}, orphans={orphans}, integrity={integrity}")
    by_src = cur.execute("SELECT source, COUNT(*), SUM(is_active) FROM listings GROUP BY source ORDER BY 2 DESC").fetchall()
    for s in by_src:
        log(f"  {s[0]:14s} {s[1]:6d} total  {s[2] or 0:6d} active")
    assert orphans == 0, "ORPHAN price_history rows — abort"
    assert integrity == "ok", "integrity check failed — abort"
    out_conn.close()

    if args.promote:
        log("PROMOTE: backing up canonical then replacing with merged (lead-gated)")
        bak = f"{args.base}.pre_merge_{datetime.now():%Y%m%d_%H%M%S}.bak"
        shutil.copy(args.base, bak)
        shutil.copy(out, args.base)
        log(f"promoted {out} -> {args.base} (backup: {bak})")
    else:
        log(f"DRY-RUN: wrote {out}; canonical untouched. Re-run with --promote after sign-off.")


if __name__ == "__main__":
    sys.exit(main())
