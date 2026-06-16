"""
Unit tests for scripts/_safe_delete.py — the destructive-op guard (#37/#40).

Pure-unit: builds throwaway in-memory/temp SQLite DBs, no network, no canonical DB.
"""
import csv
import sqlite3
import sys
from pathlib import Path

import pytest

# Make scripts/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _safe_delete import guarded_delete, SafeDeleteAborted  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    """A 100-row listings table in a temp SQLite DB."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE listings (id INTEGER PRIMARY KEY, source TEXT, price_pcm INT)")
    cur.executemany(
        "INSERT INTO listings VALUES (?,?,?)",
        [(i, "rightmove", 1000 + i) for i in range(1, 101)],
    )
    conn.commit()
    yield conn, cur, tmp_path
    conn.close()


def _delete_cb(cur):
    def _do(ids):
        ph = ",".join("?" * len(ids))
        cur.execute(f"DELETE FROM listings WHERE id IN ({ph})", list(ids))
    return _do


def test_small_delete_proceeds_and_backs_up(db):
    conn, cur, tmp = db
    bp = guarded_delete(
        cur, "listings", [1, 2, 3, 4, 5],
        total_rows=100, do_delete=_delete_cb(cur),
        project_root=tmp, label="test-small",
    )
    assert cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 95
    # Backup file written with the 5 deleted rows + header.
    assert Path(bp).exists()
    with open(bp) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "id"           # header
    assert len(rows) == 1 + 5           # header + 5 rows


def test_over_threshold_aborts_and_deletes_nothing(db):
    conn, cur, tmp = db
    with pytest.raises(SafeDeleteAborted):
        guarded_delete(
            cur, "listings", list(range(1, 60)),   # 59/100 = 59% > 10%
            total_rows=100, do_delete=_delete_cb(cur),
            project_root=tmp, label="test-large",
        )
    # Nothing deleted.
    assert cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 100


def test_exactly_at_threshold_proceeds(db):
    conn, cur, tmp = db
    # 10/100 = exactly 10% — NOT > max_fraction, so it proceeds.
    guarded_delete(
        cur, "listings", list(range(1, 11)),
        total_rows=100, do_delete=_delete_cb(cur),
        project_root=tmp, label="test-boundary",
    )
    assert cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 90


def test_empty_ids_is_noop(db):
    conn, cur, tmp = db
    bp = guarded_delete(
        cur, "listings", [],
        total_rows=100, do_delete=_delete_cb(cur),
        project_root=tmp,
    )
    assert bp == ""
    assert cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 100


def test_custom_max_fraction(db):
    conn, cur, tmp = db
    # With a stricter 2% threshold, 5/100 = 5% should abort.
    with pytest.raises(SafeDeleteAborted):
        guarded_delete(
            cur, "listings", [1, 2, 3, 4, 5],
            total_rows=100, do_delete=_delete_cb(cur),
            project_root=tmp, max_fraction=0.02,
        )
    assert cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 100
