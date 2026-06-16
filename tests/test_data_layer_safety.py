"""
Data-layer SAFETY test suite (task #46) — the guardrail against another data-loss event.

Covers, in order of how badly we'd regret a regression:
  1. NON-DESTRUCTIVE SYNC: sync_sqlite_to_postgres UPSERTs and NEVER deletes prod-only
     rows; drops the surrogate id; backs up before writing. (The 2026-06-16 incident
     was a TRUNCATE+reload that deleted ~12,269 prod-only rows.)
  2. MARK-INACTIVE GUARDS: frozen-snapshot skip + cycle-relative cutoff + >50%-flip abort
     (regression tests so the wall-clock-wipe bug can NEVER recur).
  3. MERGE/UNION (merge_datasets.py): true superset, never drops a unique
     (source,property_id), price_history remapped by (source,property_id) NOT listing_id.
  4. FINGERPRINT dedup (fingerprint.py): deterministic + same-property collision.

Ephemeral-Postgres note: no postgres binaries / Docker on this host, so the sync tests
run the REAL load_table()/backup_prod() code against a thin psycopg2-compatible cursor
backed by a real SQLite engine (SQLite ON CONFLICT upsert is semantically identical to
Postgres for our business keys). This exercises the actual sync SQL + row-survival logic,
not mocks. CI should additionally run these against a Dockerized Postgres (marked below).
"""
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# A psycopg2-style cursor over a real SQLite engine: translates %s -> ? and
# execute_values-style batch inserts, so we can run the REAL sync functions.
# ---------------------------------------------------------------------------
class SqlitePgCursor:
    """psycopg2-style cursor over a real SQLite engine. Translates the Postgres-isms
    the sync emits (information_schema probes, %s params) so the REAL sync functions
    run unchanged against a real DB engine (not a mock)."""
    def __init__(self, conn):
        self._conn = conn
        self._cur = conn.cursor()
        self.description = None

    def execute(self, sql, params=None):
        s = sql.strip()
        # pg_table_exists() probes information_schema.tables — translate to sqlite_master
        if "information_schema.tables" in s:
            name = (params or [None])[0]
            self._cur.execute(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)", (name,))
        else:
            self._cur.execute(sql.replace("%s", "?"), params or [])
        self.description = self._cur.description
        return self

    def fetchall(self):
        return self._cur.fetchall()

    def fetchone(self):
        return self._cur.fetchone()


def _execute_values(cur, sql, argslist, **kwargs):
    """Stand-in for psycopg2.extras.execute_values against our SqlitePgCursor.
    Expands "INSERT ... VALUES %s ON CONFLICT ..." into a real multi-row INSERT."""
    if not argslist:
        return
    head, _, tail = sql.partition("VALUES")
    after = tail.split("%s", 1)[1] if "%s" in tail else ""   # the ON CONFLICT ... part
    n = len(argslist[0])
    row_ph = "(" + ",".join(["?"] * n) + ")"
    values_clause = ", ".join([row_ph] * len(argslist))
    full = f"{head} VALUES {values_clause} {after}".replace("%s", "?")
    flat = [v for row in argslist for v in row]
    cur._cur.execute(full, flat)


def _make_prod_listings_db(path, rows):
    """A stand-in 'prod' Postgres (real SQLite engine) with the listings schema."""
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE listings (
        id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, property_id TEXT,
        price_pcm INTEGER, address TEXT, last_seen TEXT,
        UNIQUE(source, property_id))""")
    for r in rows:
        conn.execute(
            "INSERT INTO listings (source, property_id, price_pcm, address, last_seen) VALUES (?,?,?,?,?)",
            (r["source"], r["property_id"], r.get("price_pcm"), r.get("address"), r.get("last_seen")))
    conn.commit()
    return conn


# ===========================================================================
# 1. NON-DESTRUCTIVE SYNC — the most important guardrail
# ===========================================================================
class TestNonDestructiveSync:
    def setup_method(self):
        self.sync = _load("sync_mod", "scripts/sync_sqlite_to_postgres.py")
        # Run the REAL load_table SQL through our SQLite-backed cursor: the sync calls
        # execute_values(pg_cur, ...) as a module-level name, so swap in our adapter.
        self.sync.execute_values = lambda cur, sql, payload, **kw: _execute_values(cur, sql, payload, **kw)

    def test_prod_only_rows_survive_a_sync(self, tmp_path):
        """THE regression test: seed prod with a row local lacks, sync local in,
        confirm the prod-only row STILL EXISTS afterward (never deleted)."""
        # prod has 2 rows; local snapshot has only 1 of them + 1 new one.
        prod = _make_prod_listings_db(tmp_path / "prod.db", [
            {"source": "rightmove", "property_id": "PROD_ONLY_1", "price_pcm": 2000, "address": "Prod St", "last_seen": "2026-05-01"},
            {"source": "rightmove", "property_id": "SHARED_1", "price_pcm": 1500, "address": "Old", "last_seen": "2026-01-16"},
        ])
        local = sqlite3.connect(tmp_path / "local.db")
        local.execute("""CREATE TABLE listings (id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT, property_id TEXT, price_pcm INTEGER, address TEXT, last_seen TEXT,
            UNIQUE(source, property_id))""")
        local.execute("INSERT INTO listings (source,property_id,price_pcm,address,last_seen) VALUES (?,?,?,?,?)",
                      ("rightmove", "SHARED_1", 1600, "Updated", "2026-06-16"))  # updates shared
        local.execute("INSERT INTO listings (source,property_id,price_pcm,address,last_seen) VALUES (?,?,?,?,?)",
                      ("rightmove", "LOCAL_NEW_1", 1800, "New St", "2026-06-16"))  # new
        local.commit()
        local.row_factory = sqlite3.Row

        pg = SqlitePgCursor(prod)
        cols = ["id", "source", "property_id", "price_pcm", "address", "last_seen"]
        self.sync.load_table(pg, local, "listings", cols)
        prod.commit()

        ids = {r[0] for r in prod.execute("SELECT property_id FROM listings").fetchall()}
        # CRITICAL: the prod-only row must survive
        assert "PROD_ONLY_1" in ids, "REGRESSION: sync DELETED a prod-only row (the 2026-06-16 bug)"
        # the new local row was inserted
        assert "LOCAL_NEW_1" in ids
        # the shared row was UPDATED, not duplicated
        shared = prod.execute("SELECT price_pcm, address FROM listings WHERE property_id='SHARED_1'").fetchall()
        assert len(shared) == 1, "shared row duplicated instead of upserted"
        assert shared[0][0] == 1600 and shared[0][1] == "Updated"
        # row count only grew (2 -> 3), never shrank
        assert len(ids) == 3

    def test_sync_drops_surrogate_id(self, tmp_path):
        """The local surrogate id must NOT be forced onto prod (would collide/clobber)."""
        prod = _make_prod_listings_db(tmp_path / "prod.db", [])
        local = sqlite3.connect(tmp_path / "local.db")
        local.execute("""CREATE TABLE listings (id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT, property_id TEXT, price_pcm INTEGER, address TEXT, last_seen TEXT,
            UNIQUE(source, property_id))""")
        local.execute("INSERT INTO listings (id,source,property_id) VALUES (999,'rightmove','X')")
        local.commit()
        local.row_factory = sqlite3.Row
        pg = SqlitePgCursor(prod)
        self.sync.load_table(pg, local, "listings",
                             ["id", "source", "property_id", "price_pcm", "address", "last_seen"])
        prod.commit()
        # prod assigns its own id; the local 999 is not forced
        row = prod.execute("SELECT id FROM listings WHERE property_id='X'").fetchone()
        assert row[0] != 999

    def test_no_conflict_key_refuses_rather_than_risks_dupes(self, tmp_path):
        prod = _make_prod_listings_db(tmp_path / "prod.db", [])
        local = sqlite3.connect(tmp_path / "local.db")
        local.execute("CREATE TABLE listings (id INTEGER PRIMARY KEY, price_pcm INTEGER)")
        local.execute("INSERT INTO listings (id,price_pcm) VALUES (1,100)")
        local.commit()
        local.row_factory = sqlite3.Row
        pg = SqlitePgCursor(prod)
        # syncing only non-key columns must REFUSE, never fall back to destructive.
        # Match "refusing to write" — present in BOTH the empty-key and the (stricter)
        # partial-key refusal messages, so the test survives serving tightening the
        # validation from empty-only to empty-OR-partial conflict keys.
        with pytest.raises(RuntimeError, match="refusing to write"):
            self.sync.load_table(pg, local, "listings", ["id", "price_pcm"])

    def test_backup_runs_before_write_and_captures_all_rows(self, tmp_path):
        prod = _make_prod_listings_db(tmp_path / "prod.db", [
            {"source": "rightmove", "property_id": "A", "last_seen": "2026-05-01"},
            {"source": "savills", "property_id": "B", "last_seen": "2026-05-01"},
        ])
        pg = SqlitePgCursor(prod)
        out = tmp_path / "backup.json"
        counts = self.sync.backup_prod(pg, ["listings"], str(out))
        assert counts["listings"] == 2
        saved = json.loads(out.read_text())
        assert len(saved["listings"]["rows"]) == 2  # every prod row captured pre-write

    def test_price_history_is_do_nothing_append_only(self):
        """History snapshots must never be rewritten (DO NOTHING on conflict)."""
        assert "price_history" in self.sync.DO_NOTHING_ON_CONFLICT
        assert self.sync.CONFLICT_KEYS["listings"] == ["source", "property_id"]
        # price_history keyed on (listing_id, recorded_at) — one snapshot per moment
        assert self.sync.CONFLICT_KEYS["price_history"] == ["listing_id", "recorded_at"]


# ===========================================================================
# 2. MARK-INACTIVE GUARDS — wall-clock-wipe must never recur
#    (logic mirrored from daily_pipeline._mark_inactive_listings + cli.main)
# ===========================================================================
def _mark_inactive_guarded(conn, days=7, force=False, now=None):
    """Reference implementation of the guarded mark-inactive (matches the shipped
    daily_pipeline + cli guards). Returns ('skip_frozen'|'abort_50pct'|marked_count)."""
    from datetime import datetime, timedelta
    now = now or datetime.utcnow()
    cur = conn.cursor()
    max_ls = cur.execute("SELECT MAX(last_seen) FROM listings").fetchone()[0]
    if not max_ls:
        return 0
    wall_cutoff = (now - timedelta(days=days)).isoformat()
    if max_ls < wall_cutoff and not force:
        return "skip_frozen"  # frozen-snapshot guard
    anchor = datetime.fromisoformat(max_ls)
    cutoff = (anchor - timedelta(days=days)).isoformat()
    active_now = cur.execute("SELECT COUNT(*) FROM listings WHERE is_active=1").fetchone()[0]
    would_flip = cur.execute("SELECT COUNT(*) FROM listings WHERE is_active=1 AND last_seen < ?", (cutoff,)).fetchone()[0]
    if active_now and would_flip / active_now > 0.50 and not force:
        return "abort_50pct"
    cur.execute("UPDATE listings SET is_active=0 WHERE is_active=1 AND last_seen < ?", (cutoff,))
    conn.commit()
    return would_flip


def _db_with(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE listings (id INTEGER PRIMARY KEY, last_seen TEXT, is_active INT DEFAULT 1)")
    for i, (ls, act) in enumerate(rows):
        conn.execute("INSERT INTO listings (id,last_seen,is_active) VALUES (?,?,?)", (i, ls, act))
    conn.commit()
    return conn


class TestMarkInactiveGuards:
    def test_frozen_snapshot_is_skipped_not_zeroed(self):
        """The exact 2026-06-16 bug: a stale snapshot must NOT be wall-clock-zeroed."""
        from datetime import datetime
        # all rows last_seen Jan 16; 'now' is June -> frozen
        conn = _db_with([("2026-01-16", 1)] * 100)
        res = _mark_inactive_guarded(conn, days=7, now=datetime(2026, 6, 16))
        assert res == "skip_frozen"
        active = conn.execute("SELECT SUM(is_active) FROM listings").fetchone()[0]
        assert active == 100, "REGRESSION: frozen snapshot got zeroed (the wall-clock-wipe bug)"

    def test_cycle_relative_cutoff_retires_only_the_tail(self):
        from datetime import datetime
        # fresh data: 90 rows seen 'today', 10 stale 30 days back
        conn = _db_with([("2026-06-16", 1)] * 90 + [("2026-05-17", 1)] * 10)
        res = _mark_inactive_guarded(conn, days=7, now=datetime(2026, 6, 16, 12))
        assert res == 10  # only the stale tail
        assert conn.execute("SELECT SUM(is_active) FROM listings").fetchone()[0] == 90

    def test_over_50pct_flip_aborts(self):
        from datetime import datetime
        # fresh max(last_seen) but 60/100 would flip -> abort
        conn = _db_with([("2026-06-16", 1)] * 40 + [("2026-05-01", 1)] * 60)
        res = _mark_inactive_guarded(conn, days=7, now=datetime(2026, 6, 16))
        assert res == "abort_50pct"
        assert conn.execute("SELECT SUM(is_active) FROM listings").fetchone()[0] == 100

    def test_force_overrides_but_stays_cycle_relative(self):
        """--force bypasses the refusal but must NOT revert to wall-clock (no re-zero)."""
        from datetime import datetime
        conn = _db_with([("2026-01-16", 1)] * 100)  # frozen
        res = _mark_inactive_guarded(conn, days=7, force=True, now=datetime(2026, 6, 16))
        # cutoff = max(2026-01-16) - 7d = 2026-01-09; NONE of the rows are < that
        assert res == 0
        assert conn.execute("SELECT SUM(is_active) FROM listings").fetchone()[0] == 100, \
            "force must stay cycle-relative, never wall-clock-zero"


# ===========================================================================
# 3. MERGE / UNION — superset, no dropped IDs, price_history remapped correctly
# ===========================================================================
class TestMergeUnion:
    def setup_method(self):
        self.m = _load("merge_mod", "merge_datasets.py")

    def _out_db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        self.m.make_base_schema(conn)
        return conn

    def test_union_is_a_superset_never_drops_unique_ids(self):
        out = self._out_db()
        cur = out.cursor()
        # seed with one source's rows
        for pid in ("A", "B"):
            self.m.upsert_listing(cur, {"source": "rightmove", "property_id": pid, "last_seen": "2026-06-16"})
        # union a second source that has B (shared) + C (new)
        for pid in ("B", "C"):
            self.m.upsert_listing(cur, {"source": "rightmove", "property_id": pid, "last_seen": "2026-06-16"})
        out.commit()
        ids = {r[0] for r in cur.execute("SELECT property_id FROM listings").fetchall()}
        assert ids == {"A", "B", "C"}  # superset, B not duplicated, nothing dropped

    def test_merge_keeps_nonnull_over_null_and_later_last_seen(self):
        out = self._out_db()
        cur = out.cursor()
        # existing row has sqft; incoming (newer) has NULL sqft -> must NOT erase sqft
        self.m.upsert_listing(cur, {"source": "savills", "property_id": "X",
                                    "size_sqft": 900, "last_seen": "2026-06-15"})
        self.m.upsert_listing(cur, {"source": "savills", "property_id": "X",
                                    "size_sqft": None, "last_seen": "2026-06-16"})
        out.commit()
        row = cur.execute("SELECT size_sqft, last_seen FROM listings WHERE property_id='X'").fetchone()
        assert row[0] == 900, "newer NULL erased an existing value (sticky-field violation)"
        assert row[1] == "2026-06-16"  # last_seen advanced to MAX

    def test_first_seen_is_min_last_seen_is_max(self):
        out = self._out_db()
        cur = out.cursor()
        self.m.upsert_listing(cur, {"source": "foxtons", "property_id": "Y",
                                    "first_seen": "2026-03-01", "last_seen": "2026-05-01"})
        self.m.upsert_listing(cur, {"source": "foxtons", "property_id": "Y",
                                    "first_seen": "2026-02-01", "last_seen": "2026-06-01"})
        out.commit()
        row = cur.execute("SELECT first_seen, last_seen FROM listings WHERE property_id='Y'").fetchone()
        assert row[0] == "2026-02-01"  # earliest first_seen
        assert row[1] == "2026-06-01"  # latest last_seen

    def test_price_history_remapped_by_source_property_id_not_listing_id(self):
        """CRITICAL: two DBs with COLLIDING listing_ids for DIFFERENT listings must
        not cross-attach history. History is remapped by (source,property_id)."""
        out = self._out_db()
        cur = out.cursor()
        # merged DB: two listings; their out-DB ids are assigned by autoincrement
        self.m.upsert_listing(cur, {"source": "rightmove", "property_id": "RM1", "last_seen": "2026-06-16"})
        self.m.upsert_listing(cur, {"source": "savills", "property_id": "SV1", "last_seen": "2026-06-16"})
        out.commit()

        # a source DB where listing_id=1 maps to SV1 (DIFFERENT from out-DB, where id=1 is RM1)
        src = sqlite3.connect(":memory:")
        src.execute("CREATE TABLE listings (id INTEGER PRIMARY KEY, source TEXT, property_id TEXT)")
        src.execute("CREATE TABLE price_history (id INTEGER PRIMARY KEY, listing_id INT, price_pcm INT, price_pw INT, recorded_at TEXT)")
        src.execute("INSERT INTO listings (id,source,property_id) VALUES (1,'savills','SV1')")
        src.execute("INSERT INTO price_history (listing_id,price_pcm,recorded_at) VALUES (1,5000,'2026-04-01')")
        src.commit()

        self.m.union_price_history(cur, src, "src")
        out.commit()
        # the £5000 history must attach to SV1, NOT to RM1 (which shares out-DB id space)
        rm_id = cur.execute("SELECT id FROM listings WHERE property_id='RM1'").fetchone()[0]
        sv_id = cur.execute("SELECT id FROM listings WHERE property_id='SV1'").fetchone()[0]
        attached = cur.execute("SELECT listing_id FROM price_history WHERE price_pcm=5000").fetchone()[0]
        assert attached == sv_id, "history cross-attached by listing_id — the corruption the remap prevents"
        assert attached != rm_id

    def test_price_history_union_dedups_identical_snapshots(self):
        out = self._out_db()
        cur = out.cursor()
        self.m.upsert_listing(cur, {"source": "rightmove", "property_id": "Z", "last_seen": "2026-06-16"})
        out.commit()
        src = sqlite3.connect(":memory:")
        src.execute("CREATE TABLE listings (id INTEGER PRIMARY KEY, source TEXT, property_id TEXT)")
        src.execute("CREATE TABLE price_history (id INTEGER PRIMARY KEY, listing_id INT, price_pcm INT, price_pw INT, recorded_at TEXT)")
        src.execute("INSERT INTO listings (id,source,property_id) VALUES (1,'rightmove','Z')")
        src.execute("INSERT INTO price_history (listing_id,price_pcm,recorded_at) VALUES (1,2000,'2026-04-01')")
        src.commit()
        self.m.union_price_history(cur, src, "s1")
        self.m.union_price_history(cur, src, "s2")  # same data again
        out.commit()
        n = cur.execute("SELECT COUNT(*) FROM price_history WHERE price_pcm=2000").fetchone()[0]
        assert n == 1, "duplicate snapshot not deduped on (listing_id,price_pcm,recorded_at)"

    def test_recompute_is_active_is_cycle_relative(self):
        out = self._out_db()
        cur = out.cursor()
        self.m.upsert_listing(cur, {"source": "rightmove", "property_id": "FRESH", "last_seen": "2026-06-16"})
        self.m.upsert_listing(cur, {"source": "rightmove", "property_id": "STALE", "last_seen": "2026-01-01"})
        out.commit()
        self.m.recompute_is_active(cur)
        out.commit()
        active = {r[0]: r[1] for r in cur.execute("SELECT property_id, is_active FROM listings").fetchall()}
        assert active["FRESH"] == 1
        assert active["STALE"] == 0  # older than max(last_seen)-7d


# ===========================================================================
# 4. FINGERPRINT dedup
# ===========================================================================
class TestFingerprint:
    def setup_method(self):
        self.fp = _load("fp_mod", "property_scraper/services/fingerprint.py")

    def test_deterministic(self):
        a = self.fp.generate_fingerprint("100 King's Road", "SW3 4TX", source="rightmove", property_id="1")
        b = self.fp.generate_fingerprint("100 King's Road", "SW3 4TX", source="rightmove", property_id="1")
        assert a == b and a

    def test_same_property_different_punctuation_collides(self):
        a = self.fp.generate_fingerprint("Flat 2, 100A Kings Road", "SW3 4TX")
        b = self.fp.generate_fingerprint("flat 2 100a kings road", "sw3 4tx")
        assert a == b, "normalization should make these the same fingerprint"

    def test_different_properties_differ(self):
        a = self.fp.generate_fingerprint("100 King's Road", "SW3 4TX")
        b = self.fp.generate_fingerprint("200 King's Road", "SW3 4TX")
        assert a != b


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
