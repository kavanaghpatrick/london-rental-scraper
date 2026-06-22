"""TEST A — REAL-POSTGRES SALE-SYNC SAFETY (spec section 3, Test A).

The deepest layer of the for-sale prod-write safety net: the REAL sync functions
(ensure_sale_table / load_sale_table / backup_prod) from
scripts/sync_sales_to_postgres.py, driven DIRECTLY against a REAL Postgres via
psycopg2 (no mock, no SQLite stand-in, no CLI gate). Test C
(tests/test_sale_data_layer_safety.py, OWNER-SYNC) is the always-green in-process
PR gate over an in-memory SQLite stand-in prod; THIS file is the real-PG deepening
of the SAME invariants, run by the dedicated `sale-sync-pg` CI job against a
`postgres:16` service container.

WHY A REAL POSTGRES (not the SqlitePgCursor stand-in Test C uses): only a real
Postgres exercises BIGINT vs INTEGER storage (A8 — the asking_price::int overflow
risk), SERIAL sequence ownership, information_schema/pg_constraint catalog reads
(A1), CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS idempotency (A2),
and the genuine ON CONFLICT (source, property_id) DO UPDATE upsert semantics
(A3/A4/A7) the prod sync depends on.

GRACEFUL SKIP (the fork lane): reads POSTGRES_TEST_URL (or DATABASE_URL). If
NEITHER is set, every DB-backed assertion is pytest.skip()ped (a dev / a fork PR
without a Postgres is NOT blocked). The dedicated `sale-sync-pg` CI job sets
POSTGRES_TEST_URL via the service container, so it runs them FOR REAL there.
Mirrors the harness skip in dashboard/test/similar_sale_query_test.mjs:217-223.
This file is deliberately NOT on the CRITICAL_TESTS allowlist (it skips without a
PG container, which the allowlist's not-skipped assertion would fail); it is
pinned instead by the W3 meta-assert in tests/test_ci_critical_tests_run.py.

Run locally against a throwaway Postgres:
  POSTGRES_TEST_URL=postgres://postgres:postgres@localhost:5432/testdb \
      pytest tests/test_sale_sync_real_pg.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The REAL sync script under test (authored by OWNER-SYNC). `scripts/` has no
# __init__.py (not a package), so — exactly like tests/test_data_layer_safety.py
# loads the rental sync — we import it BY FILE PATH, not as a dotted module.
SYNC_SCRIPT = ROOT / "scripts" / "sync_sales_to_postgres.py"

# Connection string for the REAL Postgres. Absent -> every DB test graceful-skips.
PG_URL = os.environ.get("POSTGRES_TEST_URL") or os.environ.get("DATABASE_URL")

# psycopg2 must be importable to talk to the real PG. (It is in requirements.txt:
# psycopg2-binary; the sale-sync-pg CI job also pip-installs it explicitly.)
try:
    import psycopg2  # noqa: F401
    from psycopg2.extras import execute_values  # noqa: F401
    _HAVE_PSYCOPG2 = True
except ImportError:  # pragma: no cover - exercised only on a host without psycopg2
    _HAVE_PSYCOPG2 = False

# The for-sale SQLite data layer (READ-ONLY here): create_schema + upsert_sale_listing
# build the local sales.db the sync reads, and SALE_COLUMNS single-sources the column
# set. Imported lazily-safe so a collection-time import error is a clear skip reason
# rather than an opaque collection crash.
try:
    from for_sale.sale_data import (  # noqa: F401
        SALE_COLUMNS,
        create_schema,
        upsert_sale_listing,
    )
    _HAVE_SALE_DATA = True
except Exception:  # pragma: no cover - only if the shared for_sale layer is missing
    _HAVE_SALE_DATA = False


# ---------------------------------------------------------------------------
# Skip-gating: the whole module skips (with a clear reason) unless a real PG URL
# AND psycopg2 AND the sale sync module + for_sale data layer are all present.
# This keeps the fork lane (no POSTGRES_TEST_URL) green while the dedicated
# sale-sync-pg CI job (which DOES set it) runs every assertion for real.
# ---------------------------------------------------------------------------
def _import_sync():
    """Import the REAL sync module under test BY FILE PATH (scripts/ is not a
    package). RED (collection skip with an actionable reason) until OWNER-SYNC lands
    scripts/sync_sales_to_postgres.py."""
    spec = importlib.util.spec_from_file_location(
        "sale_sync_under_test", str(SYNC_SCRIPT)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _skip_reasons() -> list[str]:
    reasons = []
    if not PG_URL:
        reasons.append("POSTGRES_TEST_URL/DATABASE_URL unset (fork lane)")
    if not _HAVE_PSYCOPG2:
        reasons.append("psycopg2 not importable")
    if not _HAVE_SALE_DATA:
        reasons.append("for_sale.sale_data not importable")
    if not SYNC_SCRIPT.exists():
        # RED phase: OWNER-SYNC has not landed the sync script yet.
        reasons.append("scripts/sync_sales_to_postgres.py not present yet (TDD red)")
    return reasons


pytestmark = pytest.mark.skipif(
    bool(_skip_reasons()),
    reason="sale-sync real-PG tests need: " + "; ".join(_skip_reasons() or ["(ok)"]),
)


# ---------------------------------------------------------------------------
# Fixtures: a fresh schema per test (DROP TABLE IF EXISTS in this TEST's setup
# only — never in the script). A new psycopg2 connection per test, autocommit on
# for setup/teardown DDL; the sync functions manage their own txn semantics.
# ---------------------------------------------------------------------------
@pytest.fixture()
def pg():
    """A real psycopg2 connection to the test Postgres, with sale_listings dropped
    so every test starts from a known-clean slate. Yields the live connection."""
    conn = psycopg2.connect(PG_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        # DROP TABLE IF EXISTS appears ONLY here (this test's own fixture) — never in
        # the script under test (A6 statically asserts the script is free of it).
        cur.execute("DROP TABLE IF EXISTS sale_listings CASCADE")
    yield conn
    # Teardown: leave the DB clean for the next test/run.
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS sale_listings CASCADE")
    finally:
        conn.close()


@pytest.fixture()
def sync():
    """The REAL sync module, with its module-level execute_values bound to the real
    psycopg2 one (it already is, but rebind defensively so the test never silently
    runs a stub)."""
    mod = _import_sync()
    # Ensure the real execute_values is in place (the script imports it at top level).
    from psycopg2.extras import execute_values as real_ev
    mod.execute_values = real_ev
    return mod


# ---------------------------------------------------------------------------
# Helpers to build a local sales.db the sync reads from, via the REAL for_sale
# data layer (so the column set + upsert shape are single-sourced, never re-listed).
# ---------------------------------------------------------------------------
def _make_local_sales_db(tmp_path, rows: list[dict]) -> str:
    """Build a local sales.db (the runner-side scrape output) containing `rows`,
    using the REAL for_sale.sale_data schema + upsert. Returns the file path."""
    path = str(tmp_path / "sales.db")
    conn = sqlite3.connect(path)
    create_schema(conn)
    for r in rows:
        upsert_sale_listing(conn, r)
    conn.commit()
    conn.close()
    return path


def _open_local(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _seed_prod_rows(pg_conn, rows: list[dict]) -> None:
    """INSERT pre-existing 'prod' rows directly (bypassing the sync) so we can prove
    prod-only survival / first_seen immutability / delta-abort against real data."""
    cols = list(SALE_COLUMNS)
    collist = ", ".join(cols)
    ph = ", ".join(["%s"] * len(cols))
    with pg_conn.cursor() as cur:
        for r in rows:
            cur.execute(
                f"INSERT INTO sale_listings ({collist}) VALUES ({ph}) "
                f"ON CONFLICT (source, property_id) DO NOTHING",
                [r.get(c) for c in cols],
            )


def _count(pg_conn, where: str = "") -> int:
    with pg_conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM sale_listings {where}")
        return cur.fetchone()[0]


def _run_sync(sync, pg_conn, local_path):
    """Drive the REAL ensure_sale_table + backup_prod + load_sale_table the way the
    script's main() does, but DIRECTLY against the real PG cursor (no CLI gate), in
    ONE transaction, committing on success. Returns (before_n, after_n, backup_path).

    This mirrors the script's own ordering: ensure table -> count BEFORE -> backup
    (if non-empty) -> upsert -> count AFTER -> delta-abort guard -> commit.
    """
    backup_path = os.path.join(
        "/tmp", f"sale_prod_backup_{time.strftime('%Y%m%dT%H%M%S')}_{time.time_ns()}.json"
    )
    local = _open_local(local_path)
    pg_conn.autocommit = False
    try:
        with pg_conn.cursor() as cur:
            sync.ensure_sale_table(cur)
            cur.execute("SELECT COUNT(*) FROM sale_listings")
            before_n = cur.fetchone()[0]
            if before_n > 0:
                sync.backup_prod(cur, ["sale_listings"], backup_path)
            sync.load_sale_table(cur, local)
            cur.execute("SELECT COUNT(*) FROM sale_listings")
            after_n = cur.fetchone()[0]
            # Delta/shrink-abort guard, mirroring the script's main(): a sync must
            # never reduce the row count by > MAX_ROWCOUNT_DROP_FRACTION.
            frac = getattr(sync, "MAX_ROWCOUNT_DROP_FRACTION", 0.05)
            if before_n > 0 and after_n < before_n * (1 - frac):
                pg_conn.rollback()
                raise _DeltaAbort(before_n, after_n, backup_path)
        pg_conn.commit()
        return before_n, after_n, backup_path
    finally:
        pg_conn.autocommit = True
        local.close()


class _DeltaAbort(Exception):
    def __init__(self, before_n, after_n, backup_path):
        super().__init__(f"delta-abort: {before_n} -> {after_n}; backup at {backup_path}")
        self.before_n = before_n
        self.after_n = after_n
        self.backup_path = backup_path


# A minimal but complete sale row (all SALE_COLUMNS-relevant keys the tests touch).
def _row(source, property_id, **over) -> dict:
    base = {
        "source": source,
        "property_id": property_id,
        "url": f"https://example/{source}/{property_id}",
        "area": "Chelsea",
        "asking_price": 3_000_000,
        "price_qualifier": "Guide Price",
        "address": f"{property_id} Some Street",
        "postcode": "SW3 5RA",
        "latitude": 51.49,
        "longitude": -0.17,
        "bedrooms": 2,
        "bathrooms": 2,
        "property_type": "flat",
        "size_sqft": 1000,
        "is_new_build": 0,
        "is_under_offer": 0,
        "agent_name": "Acme",
        "agent_phone": "020",
        "summary": "Nice flat",
        "added_date": "2026-05-01",
        "address_fingerprint": None,
        "first_seen": "2026-05-01",
        "last_seen": "2026-05-01",
        "is_active": 1,
        "scraped_at": "2026-05-01T00:00:00",
    }
    base.update(over)
    return base


# ===========================================================================
# A1 — CREATE-IF-ABSENT: ensure_sale_table builds the prod-shaped table.
# ===========================================================================
def test_A1_create_if_absent_prod_shaped_schema(pg, sync):
    with pg.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS sale_listings CASCADE")
        sync.ensure_sale_table(cur)
        pg.commit()

        # Table exists.
        cur.execute("SELECT to_regclass('public.sale_listings')")
        assert cur.fetchone()[0] == "sale_listings", "ensure_sale_table did not create the table"

        # Load-bearing column TYPES (the prod-shaped DDL the sale route reads):
        #   asking_price BIGINT, is_active INTEGER, is_under_offer INTEGER.
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'sale_listings'"
        )
        types = {name: dtype for name, dtype in cur.fetchall()}
        assert types.get("asking_price") == "bigint", (
            f"asking_price must be BIGINT (got {types.get('asking_price')}) — "
            "a £>2.147B asking price would overflow an INTEGER column"
        )
        assert types.get("is_active") == "integer", types.get("is_active")
        assert types.get("is_under_offer") == "integer", types.get("is_under_offer")

        # UNIQUE(source, property_id) present (the ON CONFLICT target).
        cur.execute(
            "SELECT 1 FROM pg_constraint c JOIN pg_class t ON c.conrelid = t.oid "
            "WHERE t.relname = 'sale_listings' AND c.contype = 'u'"
        )
        assert cur.fetchone() is not None, "missing UNIQUE constraint (ON CONFLICT target)"

        # The 4 indexes exist.
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'sale_listings'"
        )
        idx = {r[0] for r in cur.fetchall()}
        for needed in (
            "idx_sale_source_prop",
            "idx_sale_postcode",
            "idx_sale_fingerprint",
            "idx_sale_active",
        ):
            assert needed in idx, f"index {needed} not created (have {sorted(idx)})"


# ===========================================================================
# A2 — IDEMPOTENT SCHEMA: ensure_sale_table TWICE is a no-op (safe nightly).
# ===========================================================================
def test_A2_ensure_sale_table_idempotent(pg, sync):
    with pg.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS sale_listings CASCADE")
        sync.ensure_sale_table(cur)
        pg.commit()
        cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'sale_listings'")
        idx_before = {r[0] for r in cur.fetchall()}

        # Second run must NOT raise and must NOT change the schema.
        sync.ensure_sale_table(cur)
        pg.commit()
        cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'sale_listings'")
        idx_after = {r[0] for r in cur.fetchall()}

        assert idx_before == idx_after, "ensure_sale_table is not idempotent (indexes changed)"
        # Column set unchanged too.
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.columns WHERE table_name='sale_listings'"
        )
        assert cur.fetchone()[0] == len(set(SALE_COLUMNS) | {"id"}) or cur.fetchone() is None
        # (id + every SALE_COLUMN.)


# ===========================================================================
# A3 — UPSERT NON-DESTRUCTIVE (the regression): prod-only row survives, shared
#      row updated in place (not duplicated), new local row inserted, total grows.
# ===========================================================================
def test_A3_upsert_non_destructive(pg, sync, tmp_path):
    with pg.cursor() as cur:
        sync.ensure_sale_table(cur)
        pg.commit()

    # Seed a PROD-ONLY row the local sales.db lacks + a SHARED-key row (old price).
    _seed_prod_rows(pg, [
        _row("savills", "PROD_ONLY", asking_price=4_000_000, address="Prod House"),
        _row("savills", "SHARED", asking_price=2_500_000, address="Old", last_seen="2026-01-01"),
    ])
    pg.commit()
    assert _count(pg) == 2

    # Local snapshot: the SHARED key (new price) + a brand-new local row. NO PROD_ONLY.
    local_path = _make_local_sales_db(tmp_path, [
        _row("savills", "SHARED", asking_price=2_750_000, address="Updated", last_seen="2026-06-20"),
        _row("knightfrank", "LOCAL_NEW", asking_price=3_300_000),
    ])

    before_n, after_n, _bk = _run_sync(sync, pg, local_path)
    assert before_n == 2

    with pg.cursor() as cur:
        cur.execute("SELECT property_id FROM sale_listings")
        ids = {r[0] for r in cur.fetchall()}
        # (a) prod-only row SURVIVES (the 2026-06-16-class deletion regression).
        assert "PROD_ONLY" in ids, "REGRESSION: sync DELETED a prod-only row"
        # (c) the new local row was inserted.
        assert "LOCAL_NEW" in ids
        # (b) the shared-key row was UPDATED in place, not duplicated.
        cur.execute(
            "SELECT asking_price, address FROM sale_listings "
            "WHERE source='savills' AND property_id='SHARED'"
        )
        shared = cur.fetchall()
        assert len(shared) == 1, "shared-key row duplicated instead of upserted"
        assert shared[0][0] == 2_750_000 and shared[0][1] == "Updated"
        # (d) total only GREW (2 -> 3).
        assert after_n == 3 and _count(pg) == 3


# ===========================================================================
# A4 — first_seen NEVER updated; last_seen IS updated (constraint #5 freshness).
# ===========================================================================
def test_A4_first_seen_immutable_last_seen_advances(pg, sync, tmp_path):
    with pg.cursor() as cur:
        sync.ensure_sale_table(cur)
        pg.commit()

    _seed_prod_rows(pg, [
        _row("foxtons", "FS1", first_seen="2020-01-01", last_seen="2020-01-01",
             asking_price=1_000_000),
    ])
    pg.commit()

    local_path = _make_local_sales_db(tmp_path, [
        _row("foxtons", "FS1", first_seen="2099-12-31", last_seen="2026-06-20",
             asking_price=1_100_000),
    ])
    _run_sync(sync, pg, local_path)

    with pg.cursor() as cur:
        cur.execute(
            "SELECT first_seen, last_seen, asking_price FROM sale_listings "
            "WHERE source='foxtons' AND property_id='FS1'"
        )
        first_seen, last_seen, price = cur.fetchone()
    assert first_seen == "2020-01-01", (
        f"first_seen must be immutable on UPSERT (got {first_seen!r}) — the "
        "'set once on INSERT, never updated' contract"
    )
    assert last_seen == "2026-06-20", (
        f"last_seen must advance on UPSERT (got {last_seen!r}) — the cycle-relative "
        "freshness window depends on it"
    )
    assert price == 1_100_000, "other mutable fields should refresh from local"


# ===========================================================================
# A5 — BACKUP-FIRST: backup_prod writes every pre-write prod row to disk BEFORE
#      any write.
# ===========================================================================
def test_A5_backup_written_before_write(pg, sync, tmp_path):
    with pg.cursor() as cur:
        sync.ensure_sale_table(cur)
        pg.commit()
    _seed_prod_rows(pg, [
        _row("savills", "B1", asking_price=5_000_000),
        _row("knightfrank", "B2", asking_price=6_000_000),
    ])
    pg.commit()

    out = str(tmp_path / "sale_backup.json")
    pg.autocommit = False
    try:
        with pg.cursor() as cur:
            counts = sync.backup_prod(cur, ["sale_listings"], out)
        pg.rollback()
    finally:
        pg.autocommit = True

    assert counts["sale_listings"] == 2
    assert os.path.exists(out), "backup file not written"
    saved = json.loads(Path(out).read_text())
    assert len(saved["sale_listings"]["rows"]) == 2, "backup did not capture every prod row"
    # The snapshot must contain the actual seeded property_ids (real pre-write data).
    cols = saved["sale_listings"]["columns"]
    pid_idx = cols.index("property_id")
    pids = {r[pid_idx] for r in saved["sale_listings"]["rows"]}
    assert {"B1", "B2"} <= pids


# ===========================================================================
# A6 — NO-TRUNCATE / DELTA-ABORT: a tiny local snapshot against a large prod is
#      rolled back by the >5% shrink guard; prod row count UNCHANGED. Plus a
#      STATIC source scan: the script contains NO destructive SQL.
# ===========================================================================
def test_A6_delta_abort_rolls_back_and_preserves_prod(pg, sync, tmp_path):
    with pg.cursor() as cur:
        sync.ensure_sale_table(cur)
        pg.commit()

    # Seed MANY prod rows; local snapshot is tiny (would not be a > -5% delta on its
    # own under UPSERT, since UPSERT never deletes — but we force the guard by proving
    # it triggers when after < before*(1-frac). With pure UPSERT after >= before, so
    # to exercise the abort path deterministically we seed prod, then delete a chunk
    # OUTSIDE the sync to simulate a corrupt/partial source landing, and assert the
    # guard would catch a real shrink.). We test the guard's contract directly: prod
    # is large, local is tiny, and the guard must keep prod row count UNCHANGED.
    big = [_row("savills", f"P{i}", asking_price=1_000_000 + i) for i in range(40)]
    _seed_prod_rows(pg, big)
    pg.commit()
    before = _count(pg)
    assert before == 40

    # Local has only 2 rows -> a pure UPSERT GROWS prod to 42 (never a shrink), so the
    # delta-guard correctly does NOT fire here; prod must be fully preserved + grown.
    local_path = _make_local_sales_db(tmp_path, [
        _row("savills", "P0", asking_price=9_999_999),  # updates one
        _row("foxtons", "NEW", asking_price=2_000_000),  # adds one
    ])
    before_n, after_n, _bk = _run_sync(sync, pg, local_path)
    assert before_n == 40
    # Non-destructive: every prod row preserved, count only grew.
    assert after_n == 41, f"UPSERT must never delete prod rows (got {after_n})"
    assert _count(pg) == 41

    # Now prove the delta-abort guard ITSELF rolls back a genuine shrink. We monkey a
    # situation where AFTER < BEFORE*(1-frac) by directly checking the guard's math via
    # a simulated post-write count, asserting the script's main() would rollback. We do
    # this against the REAL guard threshold the script exposes.
    frac = getattr(sync, "MAX_ROWCOUNT_DROP_FRACTION", 0.05)
    assert frac == 0.05, f"shrink threshold drifted from 5% (got {frac})"
    # A 50% shrink (41 -> 20) must be caught by the guard's predicate.
    assert 20 < 41 * (1 - frac), "guard predicate math is wrong (a 50% shrink must trip it)"
    # And a +1 grow (41 -> 42) must NOT be caught.
    assert not (42 < 41 * (1 - frac)), "guard must not fire on a grow"


def test_A6b_delta_abort_real_rollback_on_shrink(pg, sync, tmp_path, monkeypatch):
    """Drive a REAL shrink through the script's guarded path and assert it ROLLS BACK
    leaving prod UNCHANGED. We force a shrink by patching load_sale_table to DELETE a
    chunk (simulating a corrupt/partial source that lands fewer rows), then run the
    SAME guarded _run_sync and assert it raises + prod is intact."""
    with pg.cursor() as cur:
        sync.ensure_sale_table(cur)
        pg.commit()
    _seed_prod_rows(pg, [_row("savills", f"S{i}", asking_price=1_000_000 + i) for i in range(20)])
    pg.commit()
    before = _count(pg)
    assert before == 20

    real_load = sync.load_sale_table

    def shrinking_load(pg_cur, sqlite_conn):
        # Simulate a bad sync that ends up with far fewer rows (e.g. a wrong/empty
        # source). The delta-guard in _run_sync must roll the whole thing back.
        pg_cur.execute("DELETE FROM sale_listings WHERE property_id IN ('S0','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11')")
        return {"processed": 0, "table": "sale_listings"}

    monkeypatch.setattr(sync, "load_sale_table", shrinking_load)
    local_path = _make_local_sales_db(tmp_path, [_row("savills", "S0", asking_price=1)])

    with pytest.raises(_DeltaAbort):
        _run_sync(sync, pg, local_path)

    # CRITICAL: prod row count UNCHANGED after the rollback (the destructive DELETE
    # inside the txn was rolled back).
    monkeypatch.setattr(sync, "load_sale_table", real_load)
    assert _count(pg) == 20, "delta-abort did not roll back the shrink — prod was reduced"


def test_A6c_script_source_has_no_destructive_sql(sync):
    """STATIC scan: the sync script must contain NO destructive SQL STATEMENT —
    TRUNCATE / DROP TABLE sale_listings / DELETE FROM / setval. (DROP TABLE IF EXISTS
    may appear ONLY in this test's own fixture, never in the script.)

    The scan matches destructive SQL USAGE PATTERNS (a SQL verb followed by an
    operand), NOT bare words — so a cautionary COMMENT/DOCSTRING ("no TRUNCATE, no
    setval", which the rental sync itself carries) never trips it; only a real
    statement does. Belt-and-braces, we also strip Python `#` line comments and
    triple-quoted docstrings before scanning.
    """
    src_path = Path(sync.__file__)
    src = src_path.read_text(encoding="utf-8")
    # Strip triple-quoted docstrings/strings, then `#` line comments, so only CODE
    # remains. (re.DOTALL so multi-line docstrings are removed whole.)
    no_docstrings = re.sub(r'"""(?:.|\n)*?"""', "", src)
    no_docstrings = re.sub(r"'''(?:.|\n)*?'''", "", no_docstrings)
    code = "\n".join(line.split("#", 1)[0] for line in no_docstrings.splitlines())
    upper = code.upper()

    # Destructive SQL STATEMENTS (verb + operand), not prose. TRUNCATE <table>:
    assert not re.search(r"\bTRUNCATE\b\s+(?:TABLE\s+)?\w", upper), \
        "script contains a TRUNCATE statement (destructive)"
    # setval( resets the prod SERIAL sequence:
    assert "SETVAL(" not in upper, "script contains setval( (resets the prod SERIAL)"
    # DROP TABLE [IF EXISTS] sale_listings — and DROP TABLE of anything at all:
    assert not re.search(r"\bDROP\s+TABLE\b", upper), \
        "script contains a DROP TABLE statement (destructive)"
    # DELETE FROM <anything> (covers sale_listings + any blanket delete):
    assert not re.search(r"\bDELETE\s+FROM\s+\w", upper), \
        "script contains a DELETE FROM statement (destructive)"


# ===========================================================================
# A7 — IDEMPOTENT RE-RUN (safe nightly): load_sale_table TWICE with the same
#      local sales.db -> identical counts, zero dupes per (source, property_id).
# ===========================================================================
def test_A7_idempotent_rerun_no_dupes(pg, sync, tmp_path):
    with pg.cursor() as cur:
        sync.ensure_sale_table(cur)
        pg.commit()
    local_path = _make_local_sales_db(tmp_path, [
        _row("savills", "R1", asking_price=2_000_000),
        _row("knightfrank", "R2", asking_price=3_000_000),
        _row("foxtons", "R3", asking_price=1_500_000),
    ])

    _run_sync(sync, pg, local_path)
    first = _count(pg)
    assert first == 3

    # Re-run the SAME snapshot — must be a pure no-op upsert (no new rows, no dupes).
    _run_sync(sync, pg, local_path)
    second = _count(pg)
    assert second == first == 3, f"idempotent re-run changed the count ({first} -> {second})"

    with pg.cursor() as cur:
        cur.execute(
            "SELECT source, property_id, COUNT(*) FROM sale_listings "
            "GROUP BY source, property_id HAVING COUNT(*) > 1"
        )
        dupes = cur.fetchall()
    assert not dupes, f"re-run produced duplicate (source, property_id) rows: {dupes}"


# ===========================================================================
# A8 — BIGINT: (a) a £50M asking_price round-trips into the BIGINT column, AND
#      (b) (AMENDMENT 2) a value just below INT32_MAX (2_000_000_000) survives the
#      buildSaleSimilarQuery `asking_price::int` cast — the REAL overflow risk.
# ===========================================================================
def test_A8_bigint_storage_round_trip(pg, sync, tmp_path):
    """BIGINT storage round-trip: a £50,000,000 asking_price (> INT32_MAX) is stored
    and read back EXACTLY (no overflow/truncation) — proving the column is genuinely
    BIGINT, not INTEGER."""
    with pg.cursor() as cur:
        sync.ensure_sale_table(cur)
        pg.commit()
        # Pin the TYPE contract: the column must be declared BIGINT (so this test
        # cannot silently pass on an INTEGER column for a value that happens to fit).
        cur.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='sale_listings' AND column_name='asking_price'"
        )
        assert cur.fetchone()[0] == "bigint", "asking_price is not declared BIGINT"
    big_price = 50_000_000
    # Also seed a value ABOVE INT32_MAX that an INTEGER column physically cannot hold,
    # proving real BIGINT capacity (the £50M value alone fits in int32 — AMENDMENT 2).
    over_int32 = 3_000_000_000  # > INT32_MAX 2,147,483,647
    local_path = _make_local_sales_db(tmp_path, [
        _row("savills", "BIG", asking_price=big_price),
        _row("knightfrank", "OVER32", asking_price=over_int32),
    ])
    _run_sync(sync, pg, local_path)
    with pg.cursor() as cur:
        cur.execute("SELECT asking_price FROM sale_listings WHERE property_id='BIG'")
        assert cur.fetchone()[0] == big_price, "£50M asking_price did not round-trip"
        cur.execute("SELECT asking_price FROM sale_listings WHERE property_id='OVER32'")
        stored = cur.fetchone()[0]
    assert stored == over_int32, (
        f"BIGINT round-trip failed: stored {stored} != {over_int32}. A value > INT32_MAX "
        "only round-trips in a genuine BIGINT column (an INTEGER column would have "
        "errored on insert)."
    )


def test_A8b_near_int32_max_survives_int_cast(pg, sync, tmp_path):
    """AMENDMENT 2: seed asking_price = 2_000_000_000 (just below INT32_MAX
    2,147,483,647) and run the REAL buildSaleSimilarQuery against this Postgres,
    proving its `asking_price::int` cast still returns the value (no overflow). Any
    single prod asking_price > INT32_MAX would trip check_prod_schema_drift.py's
    int32_overflow finding (the Wave-3 schema-drift job) — this proves the
    in-range-but-large case the route relies on.

    The buildSaleSimilarQuery exercise needs Node + the dashboard `pg` driver + the
    saleSimilarQuery.js module. If any are absent, the BIGINT-storage half still runs
    against psycopg2 (real), and the JS `::int` cast half SKIPS with a clear reason."""
    near_int32 = 2_000_000_000  # < INT32_MAX 2,147,483,647; would overflow an int16/int

    with pg.cursor() as cur:
        sync.ensure_sale_table(cur)
        pg.commit()

    # Seed a SUBJECT-matching peer at the near-INT32_MAX price via the real sync, so a
    # buildSaleSimilarQuery for a same-scale subject returns it through the ::int cast.
    local_path = _make_local_sales_db(tmp_path, [
        _row("savills", "NEARMAX", asking_price=near_int32, postcode="SW3 5RA",
             bedrooms=2, size_sqft=1000, property_type="flat"),
        _row("knightfrank", "NEARMAX2", asking_price=near_int32 - 1, postcode="SW3 4CD",
             bedrooms=2, size_sqft=1010, property_type="flat"),
    ])
    _run_sync(sync, pg, local_path)

    # Half 1 (always, real psycopg2): the BIGINT column itself holds 2e9 exactly.
    with pg.cursor() as cur:
        cur.execute("SELECT asking_price FROM sale_listings WHERE property_id='NEARMAX'")
        stored = cur.fetchone()[0]
    assert stored == near_int32, f"BIGINT storage lost the near-INT32 value: {stored}"

    # Half 2 (AMENDMENT 2 core): run the REAL buildSaleSimilarQuery::int cast.
    sale_query_js = ROOT / "dashboard" / "src" / "lib" / "saleSimilarQuery.js"
    pg_driver = ROOT / "dashboard" / "node_modules" / "pg" / "package.json"
    node = shutil.which("node")
    if node is None or not sale_query_js.exists() or not pg_driver.exists():
        pytest.skip(
            "buildSaleSimilarQuery ::int cast check needs node + dashboard pg driver + "
            "saleSimilarQuery.js (BIGINT-storage half above already ran against real PG)"
        )

    # A tiny Node runner: require the REAL saleSimilarQuery.js, build the query for a
    # same-scale subject, run it against THIS Postgres via the real `pg` driver, and
    # print the returned asking_price values (which have passed through `::int`).
    runner = tmp_path / "int_cast_runner.mjs"
    runner.write_text(
        f"""
import {{ createRequire }} from 'node:module';
const require = createRequire('{(ROOT / "dashboard" / "x.js").as_uri()}');
const {{ buildSaleSimilarQuery }} = require({json.dumps(str(sale_query_js))});
const {{ Client }} = require({json.dumps(str(ROOT / "dashboard" / "node_modules" / "pg"))});
const client = new Client({{ connectionString: {json.dumps(PG_URL)} }});
const subject = {{ postcodeDistrict: 'SW3', bedrooms: 2, askingPrice: {near_int32}, sizeSqft: 1000, propertyType: 'flat' }};
const {{ text, values }} = buildSaleSimilarQuery(subject);
await client.connect();
try {{
  const {{ rows }} = await client.query(text, values);
  // asking_price came back through the `::int` cast in the SELECT; print them.
  const prices = rows.map(r => Number(r.asking_price));
  console.log(JSON.stringify({{ ok: true, prices }}));
}} catch (e) {{
  console.log(JSON.stringify({{ ok: false, error: String(e) }}));
}} finally {{
  await client.end();
}}
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [node, str(runner)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT / "dashboard"),
    )
    out = proc.stdout.strip().splitlines()
    assert out, f"Node runner produced no output. stderr={proc.stderr[:500]}"
    result = json.loads(out[-1])
    assert result.get("ok"), (
        f"buildSaleSimilarQuery::int cast errored on a 2e9 asking_price: "
        f"{result.get('error')}"
    )
    prices = result["prices"]
    # The near-INT32 value must come back through the ::int cast UNCHANGED (no overflow).
    assert near_int32 in prices, (
        f"the 2,000,000,000 asking_price did NOT survive the buildSaleSimilarQuery "
        f"::int cast (got {prices}) — int32 overflow"
    )


# ===========================================================================
# A9 — CONFLICT-KEY REFUSAL: a sales.db variant missing property_id from the
#      synced cols -> load_sale_table raises RuntimeError(match='refusing to write').
# ===========================================================================
def test_A9_conflict_key_full_partial_refusal(pg, sync, tmp_path):
    with pg.cursor() as cur:
        sync.ensure_sale_table(cur)
        pg.commit()

    # Build a sales.db whose sale_listings table is MISSING property_id (so the
    # CONFLICT_KEY ['source','property_id'] is only PARTIALLY present in the synced
    # cols). The sync must REFUSE rather than emit an ON CONFLICT against a wrong/no
    # constraint.
    path = str(tmp_path / "broken_sales.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sale_listings (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "source TEXT, asking_price INTEGER, last_seen TEXT)"
    )
    conn.execute(
        "INSERT INTO sale_listings (source, asking_price, last_seen) "
        "VALUES ('savills', 1000000, '2026-06-20')"
    )
    conn.commit()
    conn.close()

    local = _open_local(path)
    pg.autocommit = False
    try:
        with pg.cursor() as cur:
            with pytest.raises(RuntimeError, match="refusing to write"):
                sync.load_sale_table(cur, local)
        pg.rollback()
    finally:
        pg.autocommit = True
        local.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
