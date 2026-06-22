"""For-SALE data-layer SAFETY test suite (Test C / Layer-1) — the always-green PR gate
for the for-sale prod-write sync (scripts/sync_sales_to_postgres.py).

This is the in-process sibling of tests/test_sale_sync_real_pg.py (Test A, real Postgres).
It runs the REAL load_sale_table() / backup_prod() / ensure_sale_table() code against a
thin psycopg2-compatible cursor backed by a real SQLite engine — SQLite's
`INSERT ... ON CONFLICT(...) DO UPDATE` is semantically identical to Postgres for our
(source, property_id) business key. This exercises the actual sync SQL + row-survival
logic, NOT mocks. The same invariants are deepened against a Dockerized postgres:16 in
Test A (the dedicated `sale-sync-pg` CI job).

Covered invariants (mirror the rental Layer-1 suite + the spec's A3/A4/A5/A6/A7/A9):
  A3  UPSERT NON-DESTRUCTIVE: a prod-only sale row survives a sync; the shared-key row
      is UPDATED in place (not duplicated); the new local row inserts; count only grows.
  A4  first_seen NEVER updated / last_seen IS updated (the cycle-relative freshness clause).
  A5  BACKUP-FIRST: backup_prod writes a JSON snapshot of every prod row BEFORE any write.
  A6  DELTA/SHRINK-ABORT: a tiny local snapshot against a populated prod rolls back +
      SystemExits (the >5% shrink guard) AND a static source scan proves NO TRUNCATE /
      DROP TABLE sale_listings / DELETE FROM / setval in the sync script.
  A7  IDEMPOTENT RE-RUN: load_sale_table twice with the same local DB -> identical final
      counts, zero dupes per (source, property_id).
  A9  CONFLICT-KEY REFUSAL: a sales.db variant missing property_id -> RuntimeError.

Ephemeral-Postgres note: no postgres binaries / Docker on the PR-lane host, so these run
the REAL sync functions against the SqlitePgCursor adapter (the same pattern as the rental
tests/test_data_layer_safety.py). The DDL itself (ensure_sale_table) is Postgres-specific
(SERIAL/BIGINT/information_schema) and is asserted against real Postgres in Test A; here we
build the stand-in prod table with an equivalent SQLite schema so the UPSERT path is real.
"""
import importlib.util
import io
import json
import re
import sqlite3
import sys
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SYNC_REL = "scripts/sync_sales_to_postgres.py"
SYNC_ABS = ROOT / SYNC_REL


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _strip_python_prose(src: str) -> str:
    """Return `src` with COMMENTS and DOCSTRINGS removed, but ALL other string literals
    (where the real SQL lives) preserved. Lets the destructive-SQL scan assert on executable
    code: prose that merely names TRUNCATE/DELETE (the script documents that it avoids them)
    is dropped, while a genuine `cur.execute("DELETE FROM sale_listings")` would survive and
    trip the scan. Docstrings are identified via tokenize: a STRING token that begins a
    statement (preceded by NEWLINE/INDENT/DEDENT/ENCODING, i.e. not an assignment RHS)."""
    import ast

    # Collect (start, end) char spans of docstrings via AST, line/col -> abs offsets.
    lines = src.splitlines(keepends=True)
    line_starts = [0]
    for ln in lines:
        line_starts.append(line_starts[-1] + len(ln))

    def _off(lineno, col):
        return line_starts[lineno - 1] + col

    doc_spans = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ds = ast.get_docstring(node, clean=False)
            if ds is None:
                continue
            body0 = node.body[0]
            doc_spans.append((_off(body0.lineno, body0.col_offset),
                              _off(body0.end_lineno, body0.end_col_offset)))

    out = []
    prev = 0
    for s, e in sorted(doc_spans):
        out.append(src[prev:s])
        prev = e
    out.append(src[prev:])
    no_docs = "".join(out)

    # Drop comments with tokenize (string literals untouched).
    result = []
    toks = tokenize.generate_tokens(io.StringIO(no_docs).readline)
    try:
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                continue
            result.append(tok)
    except tokenize.TokenError:
        pass
    return tokenize.untokenize(result)


# ---------------------------------------------------------------------------
# A psycopg2-style cursor over a real SQLite engine: translates %s -> ? and the
# information_schema probe, so we can run the REAL sale-sync functions unchanged.
# (Mirrors tests/test_data_layer_safety.py:43-83.)
# ---------------------------------------------------------------------------
class SqlitePgCursor:
    def __init__(self, conn):
        self._conn = conn
        self._cur = conn.cursor()
        self.description = None

    def execute(self, sql, params=None):
        s = sql.strip()
        if "information_schema.tables" in s:
            name = (params or [None])[0]
            self._cur.execute(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)",
                (name,),
            )
        else:
            self._cur.execute(sql.replace("%s", "?"), params or [])
        self.description = self._cur.description
        return self

    def fetchall(self):
        return self._cur.fetchall()

    def fetchone(self):
        return self._cur.fetchone()


def _execute_values(cur, sql, argslist, **kwargs):
    """Stand-in for psycopg2.extras.execute_values against SqlitePgCursor.
    Expands "INSERT ... VALUES %s ON CONFLICT ..." into a real multi-row INSERT."""
    if not argslist:
        return
    head, _, tail = sql.partition("VALUES")
    after = tail.split("%s", 1)[1] if "%s" in tail else ""  # the ON CONFLICT ... part
    n = len(argslist[0])
    row_ph = "(" + ",".join(["?"] * n) + ")"
    values_clause = ", ".join([row_ph] * len(argslist))
    full = f"{head} VALUES {values_clause} {after}".replace("%s", "?")
    flat = [v for row in argslist for v in row]
    cur._cur.execute(full, flat)


# --- the stand-in "prod" sale_listings table (real SQLite engine) ----------
# Equivalent to the prod-shaped DDL (BIGINT->INTEGER affinity in sqlite, etc.).
_PROD_SALE_DDL = """
CREATE TABLE sale_listings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    property_id TEXT NOT NULL,
    url TEXT, area TEXT,
    asking_price INTEGER, price_qualifier TEXT,
    address TEXT, postcode TEXT, latitude REAL, longitude REAL,
    bedrooms INTEGER, bathrooms INTEGER, property_type TEXT, size_sqft INTEGER,
    is_new_build INTEGER DEFAULT 0, is_under_offer INTEGER DEFAULT 0,
    agent_name TEXT, agent_phone TEXT, summary TEXT, added_date TEXT,
    address_fingerprint TEXT, first_seen TEXT, last_seen TEXT,
    is_active INTEGER DEFAULT 1, scraped_at TEXT,
    UNIQUE(source, property_id)
)
"""


def _make_prod_sale_db(path, rows):
    """Stand-in 'prod' Postgres (real SQLite engine) with the sale_listings schema."""
    conn = sqlite3.connect(path)
    conn.execute(_PROD_SALE_DDL)
    for r in rows:
        keys = list(r.keys())
        ph = ", ".join("?" for _ in keys)
        conn.execute(
            f"INSERT INTO sale_listings ({', '.join(keys)}) VALUES ({ph})",
            [r[k] for k in keys],
        )
    conn.commit()
    return conn


def _make_local_sale_db(path, rows, *, drop_property_id=False):
    """Build a local sales.db the way the runner does: via for_sale.sale_data.create_schema
    + upsert_sale_listing (the REAL writer), so the synced column set is exactly SALE_COLUMNS.
    `drop_property_id` builds a degenerate schema missing property_id for the A9 refusal test."""
    conn = sqlite3.connect(path)
    if drop_property_id:
        # A sales.db whose sale_listings table LACKS property_id entirely (so the
        # conflict key cannot be fully present in the synced cols).
        conn.execute(
            "CREATE TABLE sale_listings (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "source TEXT, asking_price INTEGER, address TEXT, last_seen TEXT)"
        )
        for r in rows:
            conn.execute(
                "INSERT INTO sale_listings (source, asking_price, address, last_seen) "
                "VALUES (?,?,?,?)",
                (r.get("source"), r.get("asking_price"), r.get("address"), r.get("last_seen")),
            )
        conn.commit()
    else:
        sale_data = _load("sale_data_mod", "for_sale/sale_data.py")
        sale_data.create_schema(conn)
        for r in rows:
            sale_data.upsert_sale_listing(conn, r)
    conn.row_factory = sqlite3.Row
    return conn


# ===========================================================================
class _Base:
    def setup_method(self):
        self.sync = _load("sync_sales_mod", SYNC_REL)
        # The sync calls execute_values(pg_cur, ...) as a module-level name; swap the adapter.
        self.sync.execute_values = lambda cur, sql, payload, **kw: _execute_values(
            cur, sql, payload, **kw
        )


# ===========================================================================
# A3 — UPSERT NON-DESTRUCTIVE (THE regression)
# ===========================================================================
class TestUpsertNonDestructive(_Base):
    def test_prod_only_sale_row_survives_a_sync(self, tmp_path):
        prod = _make_prod_sale_db(
            tmp_path / "prod.db",
            [
                {"source": "savills", "property_id": "PROD_ONLY", "asking_price": 4_000_000,
                 "address": "Prod Mansion", "last_seen": "2026-05-01"},
                {"source": "savills", "property_id": "SHARED", "asking_price": 1_500_000,
                 "address": "Old", "last_seen": "2026-01-16"},
            ],
        )
        local = _make_local_sale_db(
            tmp_path / "local.db",
            [
                {"source": "savills", "property_id": "SHARED", "asking_price": 1_650_000,
                 "address": "Updated", "last_seen": "2026-06-16"},  # updates shared
                {"source": "savills", "property_id": "LOCAL_NEW", "asking_price": 2_200_000,
                 "address": "New Place", "last_seen": "2026-06-16"},  # new
            ],
        )

        pg = SqlitePgCursor(prod)
        self.sync.load_sale_table(pg, local)
        prod.commit()

        ids = {r[0] for r in prod.execute("SELECT property_id FROM sale_listings").fetchall()}
        # (a) prod-only row must survive
        assert "PROD_ONLY" in ids, "REGRESSION: sync DELETED a prod-only sale row"
        # (c) new local row inserted
        assert "LOCAL_NEW" in ids
        # (b) shared-key row UPDATED in place, not duplicated
        shared = prod.execute(
            "SELECT asking_price, address FROM sale_listings WHERE property_id='SHARED'"
        ).fetchall()
        assert len(shared) == 1, "shared row duplicated instead of upserted"
        assert shared[0][0] == 1_650_000 and shared[0][1] == "Updated"
        # (d) count only GREW (2 -> 3)
        assert len(ids) == 3


# ===========================================================================
# A4 — first_seen NEVER updated, last_seen IS updated
# ===========================================================================
class TestFirstSeenLastSeen(_Base):
    def test_first_seen_never_updated_last_seen_advances(self, tmp_path):
        prod = _make_prod_sale_db(
            tmp_path / "prod.db",
            [{"source": "knightfrank", "property_id": "KF1", "asking_price": 900_000,
              "first_seen": "2020-01-01", "last_seen": "2020-01-01"}],
        )
        local = _make_local_sale_db(
            tmp_path / "local.db",
            [{"source": "knightfrank", "property_id": "KF1", "asking_price": 950_000,
              "first_seen": "2026-06-16", "last_seen": "2026-06-16"}],
        )
        pg = SqlitePgCursor(prod)
        self.sync.load_sale_table(pg, local)
        prod.commit()

        row = prod.execute(
            "SELECT first_seen, last_seen, asking_price FROM sale_listings WHERE property_id='KF1'"
        ).fetchone()
        assert row[0] == "2020-01-01", "first_seen was overwritten (must be set-once)"
        assert row[1] == "2026-06-16", "last_seen did NOT advance (freshness clause broken)"
        assert row[2] == 950_000, "other mutable fields should refresh from local"


# ===========================================================================
# A5 — BACKUP-FIRST
# ===========================================================================
class TestBackupFirst(_Base):
    def test_backup_captures_all_prod_rows(self, tmp_path):
        prod = _make_prod_sale_db(
            tmp_path / "prod.db",
            [
                {"source": "savills", "property_id": "A", "last_seen": "2026-05-01"},
                {"source": "chestertons", "property_id": "B", "last_seen": "2026-05-01"},
            ],
        )
        pg = SqlitePgCursor(prod)
        out = tmp_path / "backup.json"
        counts = self.sync.backup_prod(pg, ["sale_listings"], str(out))
        assert counts["sale_listings"] == 2
        saved = json.loads(out.read_text())
        assert len(saved["sale_listings"]["rows"]) == 2  # every prod row captured pre-write


# ===========================================================================
# A6 — NO-TRUNCATE / DELTA-ABORT
# ===========================================================================
class TestDeltaAbortAndNoTruncate(_Base):
    def test_more_than_5pct_shrink_aborts_via_systemexit(self, tmp_path):
        """The >5% shrink guard must roll back + SystemExit and leave prod UNCHANGED.

        A pure non-destructive UPSERT can only grow or hold the row count, so to exercise
        the guard we simulate a destructive load (some future regression that deletes prod
        rows) by monkeypatching load_sale_table to wipe most of prod inside the txn. The
        guard must catch the shrink, roll back, and restore the original count."""
        prod_rows = [
            {"source": "savills", "property_id": f"P{i}", "asking_price": 1_000_000 + i,
             "last_seen": "2026-05-01"}
            for i in range(100)
        ]
        prod = _make_prod_sale_db(tmp_path / "prod.db", prod_rows)
        local = _make_local_sale_db(
            tmp_path / "local.db",
            [{"source": "savills", "property_id": "P0", "asking_price": 2_000_000,
              "last_seen": "2026-06-16"}],
        )
        before_n = prod.execute("SELECT COUNT(*) FROM sale_listings").fetchone()[0]

        # Inject a regression: a load that destructively shrinks prod (drops 90 of 100).
        def _destructive_load(pg_cur, sqlite_conn):
            pg_cur._cur.execute("DELETE FROM sale_listings WHERE property_id < 'P90'")
            return {"processed": 0, "table": "sale_listings"}

        self.sync.load_sale_table = _destructive_load

        with pytest.raises(SystemExit):
            self.sync.run_guarded_upsert(
                prod, SqlitePgCursor(prod), local, str(tmp_path / "bk.json")
            )

        # The guard called pg_conn.rollback() (prod is the connection) -> count restored.
        after_n = prod.execute("SELECT COUNT(*) FROM sale_listings").fetchone()[0]
        assert after_n == before_n, "delta-abort failed: prod row count was reduced"

    def test_source_has_no_destructive_sql(self):
        """Static scan of EXECUTABLE CODE (comments + docstrings stripped): the sync script
        must run NO TRUNCATE / DROP TABLE sale_listings / DELETE FROM sale_listings / setval.
        Stripping prose first means the contract is asserted on real SQL, not narrative that
        merely names the forbidden operations (the script documents that it avoids them)."""
        code = _strip_python_prose(SYNC_ABS.read_text())
        banned = [
            r"\bTRUNCATE\b",
            r"DROP\s+TABLE\s+(IF\s+EXISTS\s+)?sale_listings",
            r"DELETE\s+FROM\s+sale_listings",
            r"\bsetval\s*\(",
        ]
        for pat in banned:
            assert not re.search(pat, code, re.IGNORECASE), (
                f"sync_sales_to_postgres.py contains banned destructive SQL in code: {pat}"
            )


# ===========================================================================
# A7 — IDEMPOTENT RE-RUN ("safe nightly")
# ===========================================================================
class TestIdempotentRerun(_Base):
    def test_two_runs_same_local_db_no_dupes(self, tmp_path):
        prod = _make_prod_sale_db(tmp_path / "prod.db", [])
        local = _make_local_sale_db(
            tmp_path / "local.db",
            [
                {"source": "savills", "property_id": "S1", "asking_price": 1_000_000,
                 "last_seen": "2026-06-16"},
                {"source": "knightfrank", "property_id": "K1", "asking_price": 2_000_000,
                 "last_seen": "2026-06-16"},
            ],
        )
        pg = SqlitePgCursor(prod)
        self.sync.load_sale_table(pg, local)
        prod.commit()
        first = prod.execute("SELECT COUNT(*) FROM sale_listings").fetchone()[0]

        # second run, same local data
        self.sync.load_sale_table(SqlitePgCursor(prod), local)
        prod.commit()
        second = prod.execute("SELECT COUNT(*) FROM sale_listings").fetchone()[0]

        assert first == second == 2, "re-run changed counts (not idempotent)"
        dupes = prod.execute(
            "SELECT source, property_id, COUNT(*) c FROM sale_listings "
            "GROUP BY source, property_id HAVING c > 1"
        ).fetchall()
        assert not dupes, f"duplicates per (source, property_id): {dupes}"


# ===========================================================================
# A9 — CONFLICT-KEY REFUSAL
# ===========================================================================
class TestConflictKeyRefusal(_Base):
    def test_missing_property_id_refuses_to_write(self, tmp_path):
        prod = _make_prod_sale_db(tmp_path / "prod.db", [])
        local = _make_local_sale_db(
            tmp_path / "local.db",
            [{"source": "savills", "asking_price": 1_000_000, "address": "X",
              "last_seen": "2026-06-16"}],
            drop_property_id=True,
        )
        pg = SqlitePgCursor(prod)
        with pytest.raises(RuntimeError, match="refusing to write"):
            self.sync.load_sale_table(pg, local)


# ===========================================================================
# Importable-API contract (mirrors rental test_price_history_is_do_nothing_append_only)
# ===========================================================================
class TestApiContract(_Base):
    def test_constants_and_functions_present(self):
        assert self.sync.CONFLICT_KEY == ["source", "property_id"]
        assert "first_seen" in self.sync.NEVER_UPDATE
        assert self.sync.MAX_ROWCOUNT_DROP_FRACTION == 0.05
        for fn in ("ensure_sale_table", "load_sale_table", "backup_prod", "run_guarded_upsert"):
            assert callable(getattr(self.sync, fn)), f"missing importable fn: {fn}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
