"""
Cycle-relative mark-inactive logic — regression test for the recurring NIGHTLY
PROD is_active WIPE.

WHY THIS EXISTS
---------------
The "Mark stale listings inactive" step in .github/workflows/daily-scrape.yml used a
WALL-CLOCK cutoff:

    UPDATE listings SET is_active = 0
    WHERE last_seen::timestamp < NOW() - INTERVAL '2 days' AND is_active = 1

Prod is a FROZEN snapshot (MAX(last_seen) stops advancing once a real scrape finds
nothing fresh). Every night after the snapshot ages past 2 days, NOW() - 2d marches
PAST every row's last_seen, so 100% of listings flip is_active=0. /api/similar then
returns 0 peers and the dashboard "Compare" button breaks. A one-time re-sync set
~7,831 active; the next night's run wiped it again. This is the same footgun the
truncate-incident postmortem (§5.1) and the CLI `mark-inactive` command already guard
against by anchoring the cutoff to the DATA'S OWN clock.

THE FIX (cycle-relative): compare last_seen against the data's own MAX(last_seen),
not wall-clock NOW():

    UPDATE listings SET is_active = 0
    WHERE last_seen IS NOT NULL
      AND last_seen::timestamp
          < (SELECT MAX(last_seen::timestamp) FROM listings) - INTERVAL '2 days'
      AND is_active = 1

On a frozen snapshot (all last_seen ~= MAX) NOTHING is >2 days older than the
freshest, so ZERO rows flip — the synced active listings STAY active. When a real
scrape advances some last_seen to "now", MAX advances with it and genuinely-unseen
listings correctly age out.

SAFETY INVARIANTS THIS TEST PINS
--------------------------------
  * is_active is only ever FLIPPED (a reversible boolean) — NEVER a row delete.
    Every test asserts the total row COUNT is unchanged.
  * A frozen snapshot (all last_seen equal) marks ZERO rows.
  * A mixed set marks ONLY rows >2 days older than MAX; fresh rows stay active.
  * NULL last_seen is never marked (no cutoff comparison is true for NULL).
  * Empty table → MAX is NULL → marks nothing (no crash).

WHY SQLite HERE (CI-safe)
-------------------------
The prod step runs Postgres SQL. This test runs the SAME predicate against an
in-memory SQLite DB, which is the project's CI-safe pattern (no live Postgres, no
network, no canonical DB) — see tests/test_safe_delete.py. The two SQL dialects
differ only in DATE arithmetic syntax (`::timestamp - INTERVAL '2 days'` in PG vs
`datetime(..., '-2 days')` in SQLite); the LOGIC under test — "cutoff = MAX(last_seen)
- 2 days, anchored to the data, not NOW()" — is identical. A separate string-level
assert (test_workflow_sql_is_cycle_relative_not_wallclock) pins the exact Postgres
SQL shipped in the workflow so the two can't silently diverge.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).resolve().parent.parent
    / ".github"
    / "workflows"
    / "daily-scrape.yml"
)

# Mark-inactive horizon both the workflow SQL and this test use.
STALE_DAYS = 2


def _make_db(rows):
    """In-memory listings table seeded with (id, last_seen, is_active) rows.

    last_seen is stored as ISO text (exactly like the canonical schema), so the date
    comparison exercises the real text-timestamp shape.
    """
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE listings (id INTEGER PRIMARY KEY, last_seen TEXT, is_active INTEGER DEFAULT 1)"
    )
    cur.executemany("INSERT INTO listings (id, last_seen, is_active) VALUES (?,?,?)", rows)
    conn.commit()
    return conn, cur


# SQLite transliteration of the SHIPPED Postgres cycle-relative UPDATE. The predicate
# is byte-for-byte the same SHAPE:
#   - anchor the cutoff to (SELECT MAX(last_seen) FROM listings)  [NOT NOW()]
#   - subtract the stale horizon                                  [- INTERVAL '2 days']
#   - skip NULL last_seen                                         [last_seen IS NOT NULL]
#   - only touch currently-active rows                            [is_active = 1]
# Only the date-subtraction *syntax* changes (datetime(...,'-N days') vs ::timestamp -
# INTERVAL). guarded so a malformed/empty-table MAX (NULL) marks nothing.
_CYCLE_RELATIVE_SQLITE = f"""
    UPDATE listings SET is_active = 0
    WHERE last_seen IS NOT NULL
      AND last_seen < datetime(
            (SELECT MAX(last_seen) FROM listings), '-{STALE_DAYS} days'
          )
      AND is_active = 1
"""

# The ORIGINAL buggy wall-clock form, kept ONLY to PROVE (in a dedicated test) that it
# zeroes a frozen snapshot — i.e. to demonstrate the bug the fix removes. Never shipped.
_WALLCLOCK_SQLITE = f"""
    UPDATE listings SET is_active = 0
    WHERE last_seen < datetime('now', '-{STALE_DAYS} days')
      AND is_active = 1
"""


class MarkInactiveAborted(SystemExit):
    """Raised by the ported >50% abort. Subclasses SystemExit because the workflow raises
    SystemExit(1) on the over-flip; tests assert on SystemExit."""


def _mark_inactive_with_abort(cur, conn):
    """Python transliteration of the SHIPPED daily-scrape 'Mark stale listings inactive'
    step's executable logic: run the cycle-relative UPDATE, then compute the >50% abort
    against the ACTIVE set BEFORE committing. On trip -> rollback + SystemExit(1); else the
    flip stands. Mirrors .github/workflows/daily-scrape.yml + cli/main.py mark_inactive
    765-781. Returns the number of rows marked on success.

    is_active is reversible, so a trip rolls back (never commits a mass flip) — this catches
    a DIFFERENT class than the cycle-relative cutoff: a mis-dated MAX(last_seen) sweeping the
    cutoff past most rows. Measured against the ACTIVE set (same dilution lesson as A13)."""
    cur.execute(_CYCLE_RELATIVE_SQLITE)
    marked = cur.rowcount
    active_after = cur.execute(
        "SELECT COUNT(*) FROM listings WHERE is_active = 1"
    ).fetchone()[0]
    active_before = active_after + marked
    if active_before > 0 and marked > 0.5 * active_before:
        conn.rollback()
        raise MarkInactiveAborted(1)
    conn.commit()
    return marked


@pytest.mark.unit
def test_frozen_snapshot_marks_zero_rows():
    """All last_seen equal (frozen prod snapshot) -> ZERO rows flipped, however old
    the snapshot is. This is the exact prod scenario that was wiping is_active nightly."""
    # A snapshot frozen 90 days ago: every row shares the same (stale-by-wall-clock)
    # last_seen. Cycle-relative must still leave them ALL active.
    frozen = "2026-03-01T12:00:00"
    rows = [(i, frozen, 1) for i in range(1, 51)]  # 50 active rows
    conn, cur = _make_db(rows)
    try:
        cur.execute(_CYCLE_RELATIVE_SQLITE)
        marked = cur.rowcount
        active = cur.execute("SELECT COUNT(*) FROM listings WHERE is_active = 1").fetchone()[0]
        total = cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    finally:
        conn.close()

    assert marked == 0, "frozen snapshot must mark ZERO rows inactive"
    assert active == 50, "every active listing must STAY active on a frozen snapshot"
    assert total == 50, "mark-inactive must NEVER delete a row (count unchanged)"


@pytest.mark.unit
def test_mixed_set_marks_only_stale_relative_to_max():
    """Mixed last_seen: only rows >STALE_DAYS older than MAX(last_seen) flip; the
    freshest rows (and rows within the window) stay active. Row count unchanged."""
    # MAX(last_seen) = 2026-06-16. Cutoff = 2026-06-14T00:00:00.
    rows = [
        (1, "2026-06-16T10:00:00", 1),  # = MAX            -> stays active
        (2, "2026-06-15T23:00:00", 1),  # within 2d of MAX -> stays active
        (3, "2026-06-14T12:00:00", 1),  # within 2d of MAX -> stays active (cutoff is 06-14T00:00)
        (4, "2026-06-13T12:00:00", 1),  # >2d older        -> MARK inactive
        (5, "2026-06-01T00:00:00", 1),  # much older       -> MARK inactive
    ]
    conn, cur = _make_db(rows)
    try:
        cur.execute(_CYCLE_RELATIVE_SQLITE)
        marked = cur.rowcount
        still_active = {
            r[0] for r in cur.execute("SELECT id FROM listings WHERE is_active = 1").fetchall()
        }
        now_inactive = {
            r[0] for r in cur.execute("SELECT id FROM listings WHERE is_active = 0").fetchall()
        }
        total = cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    finally:
        conn.close()

    assert marked == 2, "only the 2 genuinely-stale rows should flip"
    assert now_inactive == {4, 5}, "only rows >2d older than MAX should be inactive"
    assert still_active == {1, 2, 3}, "MAX row + rows within the 2d window must stay active"
    assert total == 5, "no rows deleted — only is_active flipped"


@pytest.mark.unit
def test_null_last_seen_is_never_marked():
    """Rows with NULL last_seen must NOT be flipped (no cutoff comparison is true for
    NULL, and we explicitly require last_seen IS NOT NULL). They keep their state."""
    rows = [
        (1, "2026-06-16T10:00:00", 1),  # MAX, fresh   -> active
        (2, None, 1),                    # NULL, active -> stays active (untouched)
        (3, "2026-06-01T00:00:00", 1),  # stale        -> MARK inactive
    ]
    conn, cur = _make_db(rows)
    try:
        cur.execute(_CYCLE_RELATIVE_SQLITE)
        marked = cur.rowcount
        row2_active = cur.execute("SELECT is_active FROM listings WHERE id = 2").fetchone()[0]
        now_inactive = {
            r[0] for r in cur.execute("SELECT id FROM listings WHERE is_active = 0").fetchall()
        }
        total = cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    finally:
        conn.close()

    assert marked == 1, "only the genuinely-stale non-NULL row should flip"
    assert row2_active == 1, "NULL last_seen must never be marked inactive"
    assert now_inactive == {3}
    assert total == 3, "no rows deleted"


@pytest.mark.unit
def test_empty_table_marks_nothing_and_does_not_crash():
    """Empty table -> MAX(last_seen) is NULL -> cutoff comparison is NULL/false ->
    zero rows marked, no error (the empty-table / NULL-MAX edge case)."""
    conn, cur = _make_db([])
    try:
        cur.execute(_CYCLE_RELATIVE_SQLITE)  # must not raise
        marked = cur.rowcount
        total = cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    finally:
        conn.close()

    assert marked == 0
    assert total == 0


@pytest.mark.unit
def test_already_inactive_rows_are_left_alone():
    """The is_active = 1 guard means already-inactive rows aren't re-touched (rowcount
    counts only genuine transitions); idempotent re-runs flip nothing new."""
    rows = [
        (1, "2026-06-16T10:00:00", 1),  # fresh active
        (2, "2026-06-01T00:00:00", 0),  # stale BUT already inactive -> not in rowcount
        (3, "2026-06-01T00:00:00", 1),  # stale active -> flip
    ]
    conn, cur = _make_db(rows)
    try:
        first = cur.execute(_CYCLE_RELATIVE_SQLITE).rowcount
        # Second run: nothing left to transition.
        second = cur.execute(_CYCLE_RELATIVE_SQLITE).rowcount
        total = cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    finally:
        conn.close()

    assert first == 1, "only id=3 transitions (id=2 was already inactive)"
    assert second == 0, "re-running marks nothing new (idempotent)"
    assert total == 3


@pytest.mark.unit
def test_over_flip_aborts_and_rolls_back_active_set_unchanged():
    """>50% ABORT (defense-in-depth): when MAX(last_seen) is advanced by a small fresh
    cohort so the cycle-relative cutoff sweeps past MOST active rows, the ported abort
    refuses to commit. It rolls back -> the active COUNT is UNCHANGED, and it raises
    SystemExit. This catches a mis-dated MAX sweeping the cutoff past the bulk of the set.

    Seed: 2 fresh rows at MAX (= today) + 18 rows 30 days old. Cutoff = MAX - 2 days, so the
    18 old rows (90% of the 20 active) would flip — well over 50% -> MUST abort."""
    rows = (
        [(i, "2026-06-20T12:00:00", 1) for i in range(1, 3)]      # 2 fresh -> MAX
        + [(i, "2026-05-21T12:00:00", 1) for i in range(3, 21)]   # 18 old (30d) -> would flip
    )
    conn, cur = _make_db(rows)
    try:
        active_before = cur.execute(
            "SELECT COUNT(*) FROM listings WHERE is_active = 1"
        ).fetchone()[0]
        assert active_before == 20

        with pytest.raises(SystemExit):
            _mark_inactive_with_abort(cur, conn)

        # ROLLED BACK: the active set is UNCHANGED (no mass flip committed).
        active_after = cur.execute(
            "SELECT COUNT(*) FROM listings WHERE is_active = 1"
        ).fetchone()[0]
        total = cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    finally:
        conn.close()

    assert active_after == 20, "the >50% over-flip must be ROLLED BACK (active set intact)"
    assert total == 20, "abort must never delete a row"


@pytest.mark.unit
def test_small_flip_under_threshold_commits_normally():
    """Non-vacuous control: a LEGITIMATE small flip (<50% of active) is NOT aborted — the
    cycle-relative mark commits and only the genuinely-stale rows go inactive. Proves the
    >50% guard doesn't fire spuriously on normal operation."""
    rows = (
        [(i, "2026-06-20T12:00:00", 1) for i in range(1, 19)]    # 18 fresh -> MAX, stay active
        + [(i, "2026-05-21T12:00:00", 1) for i in range(19, 21)]  # 2 old (30d) -> flip (2/20=10%)
    )
    conn, cur = _make_db(rows)
    try:
        marked = _mark_inactive_with_abort(cur, conn)  # must NOT raise
        active_after = cur.execute(
            "SELECT COUNT(*) FROM listings WHERE is_active = 1"
        ).fetchone()[0]
        total = cur.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    finally:
        conn.close()

    assert marked == 2, "only the 2 genuinely-stale rows flip"
    assert active_after == 18, "the 18 fresh rows stay active (committed normally)"
    assert total == 20, "no rows deleted"


@pytest.mark.unit
def test_wallclock_form_WOULD_wipe_frozen_snapshot():
    """DEMONSTRATES THE BUG the fix removes: the OLD wall-clock predicate flips a
    frozen snapshot to 100% inactive. This guards against anyone reverting to NOW().

    (This runs the buggy form on purpose to prove the contrast — it is NOT the shipped
    SQL. If this ever marks 0, the wall-clock form changed and this contrast test
    should be revisited.)"""
    frozen = "2026-03-01T12:00:00"  # > 2 days before today's wall clock
    rows = [(i, frozen, 1) for i in range(1, 21)]  # 20 active rows
    conn, cur = _make_db(rows)
    try:
        cur.execute(_WALLCLOCK_SQLITE)
        marked = cur.rowcount
        active = cur.execute("SELECT COUNT(*) FROM listings WHERE is_active = 1").fetchone()[0]
    finally:
        conn.close()

    assert marked == 20, "the buggy wall-clock form wipes the WHOLE frozen snapshot"
    assert active == 0, "this is exactly the nightly prod wipe the cycle-relative fix prevents"


@pytest.mark.unit
def test_workflow_sql_is_cycle_relative_not_wallclock():
    """Pin the SHIPPED Postgres SQL in daily-scrape.yml so it can't regress to wall
    clock. The mark-inactive UPDATE MUST:
      * anchor the cutoff to (SELECT MAX(last_seen...) FROM listings) — cycle-relative,
      * NOT use `NOW() - INTERVAL` as the cutoff (the wall-clock footgun),
      * still ONLY flip is_active (no DELETE/TRUNCATE/DROP in the step).
    This keeps the in-memory logic test above honest about what actually runs in prod."""
    text = WORKFLOW.read_text()

    # Isolate the "Mark stale listings inactive" step body so we assert against THAT
    # step, not some other NOW() elsewhere in the file.
    m = re.search(
        r"- name: Mark stale listings inactive(.*?)(?:\n      - name:|\Z)",
        text,
        re.DOTALL,
    )
    assert m, "could not locate the 'Mark stale listings inactive' step in the workflow"
    step = m.group(1)

    # Strip Python comment lines (`# ...`) so an explanatory comment that QUOTES the old
    # buggy `NOW() - INTERVAL` form (to document what NOT to do) can't trip the
    # wall-clock assertion below. We test the executable CODE, not the prose.
    code = "\n".join(
        ln for ln in step.splitlines() if not ln.lstrip().startswith("#")
    )

    # The mark-inactive UPDATE must exist and target is_active.
    assert "UPDATE listings" in code and "is_active = 0" in code, (
        "mark-inactive UPDATE missing or not targeting is_active"
    )

    # Cycle-relative anchor: cutoff derived from MAX(last_seen) of the table.
    assert re.search(r"MAX\s*\(\s*last_seen", code), (
        "mark-inactive cutoff is NOT cycle-relative — it must derive from "
        "(SELECT MAX(last_seen...) FROM listings), not wall-clock NOW(). "
        "This is the exact bug that wipes the frozen prod snapshot nightly."
    )

    # The cutoff must NOT be wall-clock `NOW() - INTERVAL` in executable code. (A
    # comment quoting the old form for documentation is fine; it's stripped above.)
    assert not re.search(r"NOW\(\)\s*-\s*INTERVAL", code, re.IGNORECASE), (
        "mark-inactive still uses a WALL-CLOCK `NOW() - INTERVAL` cutoff — this zeroes "
        "is_active on the frozen prod snapshot every night. Use MAX(last_seen) - INTERVAL."
    )

    # SAFETY: the step must NEVER delete/truncate/drop — is_active is a reversible flag.
    # Check executable code (comments stripped) so prose can't false-positive.
    upper = code.upper()
    for danger in ("DELETE FROM", "TRUNCATE", "DROP TABLE", "DROP "):
        assert danger not in upper, (
            f"mark-inactive step contains a destructive op ({danger!r}); it must ONLY "
            f"flip the reversible is_active boolean, never remove rows."
        )

    # >50% ABORT GUARD (defense-in-depth, ported from cli/main.py mark_inactive 765-781):
    # the step must compute the marked fraction against the ACTIVE set and REFUSE to commit
    # a mass flip. Pin the three load-bearing tokens so the guard can't silently regress out:
    #   * the >50% threshold comparison (`> 0.5 * active...`),
    #   * a rollback (never commit a mass flip),
    #   * a non-zero exit (SystemExit(1)) so the step FAILS loudly.
    assert re.search(r">\s*0\.5\s*\*\s*active", code), (
        "mark-inactive step is missing the >50% abort threshold — it must refuse to commit "
        "when `marked > 0.5 * active_before` (a mis-dated MAX sweeping the cutoff past most "
        "active rows). Mirrors cli/main.py mark_inactive."
    )
    assert re.search(r"\brollback\b", code, re.IGNORECASE), (
        "mark-inactive >50% abort must ROLL BACK the UPDATE (never commit a mass is_active "
        "flip) before failing."
    )
    assert re.search(r"SystemExit\(\s*1\s*\)", code), (
        "mark-inactive >50% abort must `raise SystemExit(1)` so the step FAILS loudly "
        "instead of silently proceeding with a mass flip."
    )
