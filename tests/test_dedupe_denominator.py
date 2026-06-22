"""
T12 (A13) — delta-guard DENOMINATOR test for scripts/_safe_delete.guarded_delete.

The point
---------
The 10% delta-abort in guarded_delete is only as good as the `total_rows` denominator
the CALLER passes. A correct dedupe/retire retires a small tail of the RELEVANT
candidate set (active, priced rows). If a caller passes the WHOLE-TABLE COUNT(*) as the
denominator while only ever proposing to delete from the active subset, a large slice of
the *active* set can be wiped while the fraction-of-the-whole-table still looks small —
the guard never trips. This is "denominator dilution".

Concretely (the seed below): 800 inactive + 200 active rows; a buggy proposal deletes
25% of the ACTIVE set (50 rows). Measured against the active candidate set (50/200 = 25%)
that is WAY over the 10% threshold and MUST abort. Measured against the whole table
(50/1000 = 5%) it slips under 10% and the guard stays silent.

The daily-scrape.yml "Clean duplicate listings" step computes `total = COUNT(*) FROM
listings` (the WHOLE table, including inactive rows) but the dedupe candidate set it
draws `to_delete` from is `WHERE is_active = 1 AND price_pcm > 0` — so it has exactly this
dilution. That workflow fix is DEFERRED to a prod-workflow sign-off; here we:

  1. PROVE the guard logic itself is sound — when fed the ACTIVE candidate-set
     denominator it DOES abort on a 25%-of-active over-proposal (a passing gate); and
  2. XFAIL-DOCUMENT the dilution — when fed the WHOLE-TABLE denominator (as the
     daily-scrape inline SQL does) the guard does NOT abort. The xfail's assertion is the
     DESIRED behavior ("it should still abort"); it fails today because of the dilution,
     and FLIPS to a hard gate once the workflow passes the candidate-set denominator.

Pure-unit: temp in-memory SQLite, no network, no canonical DB.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

# Make scripts/ importable.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from _safe_delete import guarded_delete, SafeDeleteAborted  # noqa: E402


N_INACTIVE = 800
N_ACTIVE = 200
WHOLE_TABLE = N_INACTIVE + N_ACTIVE  # 1000


def _seed_conn():
    """800 inactive + 200 active rows. Active ids are 1..200; inactive ids 201..1000."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE listings (id INTEGER PRIMARY KEY, is_active INT, price_pcm INT)"
    )
    cur.executemany(
        "INSERT INTO listings VALUES (?,?,?)",
        [(i, 1, 2000 + i) for i in range(1, N_ACTIVE + 1)]
        + [(i, 0, 2000 + i) for i in range(N_ACTIVE + 1, WHOLE_TABLE + 1)],
    )
    conn.commit()
    return conn, cur


def _delete_cb(cur):
    def _do(ids):
        ph = ",".join("?" * len(ids))
        cur.execute(f"DELETE FROM listings WHERE id IN ({ph})", list(ids))
    return _do


def _active_candidate_count(cur) -> int:
    """The denominator the guard SHOULD use: the active, priced candidate set."""
    return cur.execute(
        "SELECT COUNT(*) FROM listings WHERE is_active = 1 AND price_pcm > 0"
    ).fetchone()[0]


def _over_proposal(cur):
    """A buggy proposal that would delete 25% of the ACTIVE set (50 active ids)."""
    n = int(0.25 * N_ACTIVE)  # 50
    ids = [r[0] for r in cur.execute(
        "SELECT id FROM listings WHERE is_active = 1 AND price_pcm > 0 "
        "ORDER BY id LIMIT ?", (n,)
    ).fetchall()]
    assert len(ids) == 50
    return ids


def test_guard_aborts_on_active_candidate_denominator(tmp_path):
    """The guard logic is SOUND: fed the ACTIVE candidate-set denominator, a proposal to
    delete 25% of the active set (>10%) ABORTS and deletes nothing. This is the behavior
    the daily-scrape step SHOULD get once it passes the right denominator."""
    conn, cur = _seed_conn()
    candidate_total = _active_candidate_count(cur)
    assert candidate_total == N_ACTIVE  # 200
    ids = _over_proposal(cur)  # 50  -> 50/200 = 25% > 10%

    with pytest.raises(SafeDeleteAborted):
        guarded_delete(
            cur, "listings", ids,
            total_rows=candidate_total,           # CORRECT denominator
            do_delete=_delete_cb(cur),
            project_root=tmp_path,
            label="active-candidate-denominator",
        )
    # Nothing deleted — the guard protected the active set.
    assert cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == WHOLE_TABLE
    conn.close()


@pytest.mark.xfail(
    reason="denominator dilution in daily-scrape.yml inline SQL (total = COUNT(*) whole "
    "table, but to_delete is drawn from WHERE is_active=1 AND price_pcm>0); fix deferred "
    "to prod-workflow sign-off. Flips to a hard gate once the workflow passes the active "
    "candidate-set denominator.",
    strict=True,
)
def test_whole_table_denominator_should_still_abort_but_does_not(tmp_path):
    """DOCUMENTING xfail pinning the real broken behavior.

    Reproduce the daily-scrape denominator EXACTLY: `total = COUNT(*) FROM listings`
    (whole table, 1000) while the over-proposal deletes 25% of the ACTIVE set (50 rows).
    The DESIRED behavior is that the guard STILL aborts (50 active rows is a bug-sized
    delete). It does NOT today, because 50/1000 = 5% < 10% dilutes the fraction. This
    assertion (that it raises) therefore FAILS -> xfail. When the workflow is fixed to
    pass the active candidate-set denominator, the guard WILL abort and this flips green
    (the strict xfail then reports XPASS and forces removing the marker)."""
    conn, cur = _seed_conn()
    whole_table_total = cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    assert whole_table_total == WHOLE_TABLE  # 1000 — the diluted denominator
    ids = _over_proposal(cur)  # 50 active rows

    # DESIRED: deleting 50 of the 200 active rows should abort regardless of how many
    # inactive rows pad the table. Today it does NOT (dilution) -> this raises-assert fails
    # -> xfail captures the bug.
    with pytest.raises(SafeDeleteAborted):
        guarded_delete(
            cur, "listings", ids,
            total_rows=whole_table_total,         # DILUTED whole-table denominator
            do_delete=_delete_cb(cur),
            project_root=tmp_path,
            label="whole-table-denominator (daily-scrape dilution)",
        )
    conn.close()


def test_dilution_is_real_not_a_test_artifact(tmp_path):
    """Belt-and-braces: prove the xfail above pins a REAL behavioral difference (not a
    flaky/vacuous xfail). With the whole-table denominator the SAME over-proposal that
    aborts under the candidate denominator instead PROCEEDS and silently deletes 50
    active rows. This is the concrete data loss the dilution bug allows."""
    conn, cur = _seed_conn()
    whole_table_total = cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    ids = _over_proposal(cur)

    # No abort under the diluted denominator: the delete goes through.
    bp = guarded_delete(
        cur, "listings", ids,
        total_rows=whole_table_total,
        do_delete=_delete_cb(cur),
        project_root=tmp_path,
        label="dilution-proceeds",
    )
    assert bp, "guard returned no backup path — it unexpectedly aborted"
    remaining = cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    assert remaining == WHOLE_TABLE - 50, "the 50 active rows were silently deleted"
    # And the active set really shrank from 200 -> 150 (the leak the guard didn't catch).
    active_left = cur.execute(
        "SELECT COUNT(*) FROM listings WHERE is_active = 1"
    ).fetchone()[0]
    assert active_left == N_ACTIVE - 50  # 150
    conn.close()
