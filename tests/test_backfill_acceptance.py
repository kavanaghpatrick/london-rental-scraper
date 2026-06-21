"""R4 — BACKFILL ACCEPTANCE + NON-DESTRUCTIVE GUARD (TDD, test-FIRST).

This file encodes "the floorplan -> OCR -> sqft bulk backfill SUCCEEDED" as
executable acceptance criteria, and — equally important — that the backfill was
NON-DESTRUCTIVE (it only ADDED size_sqft / floorplan_url / floor columns to rows
that lacked them; it never deleted rows or clobbered an existing sqft).

WHAT THE BACKFILL DOES (the running job):
    Stage 1:  python3 -m cli.main enrich-floorplans --source rightmove
              -> captures listings.floorplan_url for active rightmove rows.
    Stage 2:  python3 scripts/ocr_enrich.py --source rightmove --workers 4
              -> downloads each floorplan image, OCRs it (real Tesseract /
                 FloorplanExtractor), writes size_sqft (+ floor_count /
                 property_levels / has_* floor flags) ONLY where size_sqft was
                 NULL/0 and floorplan_url is present.
    Both stages WRITE output/rentals.db.

THE GOAL: recover size_sqft for the ~1,932 active no-sqft Rightmove rows (Rightmove
is 86% of the 28.7% no-sqft hole), so those rows become TRAINABLE
(retrain_canonical filters `WHERE size_sqft > 0`).

BASELINE: /tmp/rentals.db.prebackfill is a COPY of rentals.db taken BEFORE the
backfill started (mtime 20:47, backfill start 20:48). The non-destructive tests
diff the live DB against this immutable baseline.

CONCURRENCY: output/rentals.db is being WRITTEN live. Every read here opens it in
SQLite immutable mode (`file:...?immutable=1`) — NEVER a plain connection, NEVER a
write. No row is ever modified by this test.

GATING:
  - Tests marked `data` auto-skip when no populated rentals.db (CI without a DB).
  - The ACCEPTANCE-THRESHOLD tests (yield) additionally skip while the backfill is
    still IN PROGRESS (OCR stage not finished), because a partial run legitimately
    hasn't met the recovery target yet. Set BACKFILL_DONE=1 to force the thresholds
    to be asserted (the human runs the suite this way once the job exits, BEFORE
    deciding to retrain). The NON-DESTRUCTIVE invariants always run when a DB +
    baseline exist — those must hold at every instant, mid-run included.

Run mid-backfill (invariants only, thresholds skip):
    python3 -m pytest tests/test_backfill_acceptance.py -v
Run as the retrain gate (assert thresholds too):
    BACKFILL_DONE=1 python3 -m pytest tests/test_backfill_acceptance.py -v
"""
import os
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The live (being-written) canonical DB and the pre-backfill baseline copy.
LIVE_DB = Path(os.environ.get("SCRAPER_DB_PATH", ROOT / "output" / "rentals.db"))
BASELINE_DB = Path(os.environ.get("BACKFILL_BASELINE_DB", "/tmp/rentals.db.prebackfill"))

# Did the human flag the backfill as finished? Thresholds are only HARD-asserted then.
BACKFILL_DONE = os.environ.get("BACKFILL_DONE", "").strip().lower() in {"1", "true", "yes"}

# --- ACCEPTANCE THRESHOLDS (the "backfill succeeded" definition) -------------
# Floorplan_url coverage: stage 1 must capture a floorplan_url for a meaningful
# share of the active no-sqft rightmove rows it targets. Liveness sampling found
# ~11/12 listings carry floorplans, so a healthy capture run should reach the
# active rightmove pages and tag well over half of them. We set a deliberately
# conservative floor (dead links / delisted / blocked pages are expected) and the
# real bar is the sqft-recovered count below.
MIN_FLOORPLAN_URL_COVERAGE_FRAC = 0.50   # >=50% of active no-sqft RM rows get a floorplan_url
# Sqft recovered: the headline deliverable. At least this many previously-no-sqft
# active rightmove rows must come back with a usable (trainable) size_sqft.
MIN_SQFT_RECOVERED_ROWS = 400            # absolute floor of newly-trainable RM rows
# Of the rows that HAD a floorplan_url going into OCR, OCR must succeed on a
# reasonable share (the rest are unpar. images / dead links — accounted for, not
# a failure, but the engine working means most parse).
MIN_OCR_YIELD_ON_FLOORPLANS_FRAC = 0.40  # >=40% of floorplan'd rows yield sqft

# --- PER-ROW SANITY BOUNDS ---------------------------------------------------
# A recovered sqft must be physically sane AND trainable. retrain_canonical's
# quality filter keeps 150 <= size_sqft <= 10000 and sqft_per_bed >= 70; the OCR
# engine itself bounds 100 < sqft < 100000. Recovered values outside the trainable
# band are silently dropped by the retrain (harmless) but a recovered value outside
# the SANE band (<100 or >50000) means the OCR mis-parsed a phone number / a scan
# artifact and should never have been written.
SANE_SQFT_MIN, SANE_SQFT_MAX = 100, 50000        # OCR engine plausibility
TRAINABLE_SQFT_MIN, TRAINABLE_SQFT_MAX = 150, 10000  # retrain quality filter
MIN_SQFT_PER_BED = 70                             # retrain sqft_per_bed floor
# At most this fraction of NEWLY-recovered rows may be insane (mis-parses). A clean
# OCR run should be ~0; we allow a tiny slack for genuine OCR noise.
MAX_INSANE_RECOVERED_FRAC = 0.02


# ---------------------------------------------------------------------------
# Helpers — every connection is IMMUTABLE/read-only (DB is being written live).
# ---------------------------------------------------------------------------
def _ro_conn(db_path):
    """Open a strictly read-only, immutable connection (safe vs the live writer)."""
    conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _q1(conn, sql, params=()):
    return conn.execute(sql, params).fetchone()[0]


def _db_ok(p):
    return p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def live():
    if not _db_ok(LIVE_DB):
        pytest.skip(f"no populated live DB at {LIVE_DB}")
    conn = _ro_conn(LIVE_DB)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def baseline():
    if not _db_ok(BASELINE_DB):
        pytest.skip(
            f"no pre-backfill baseline at {BASELINE_DB} — non-destructive diff needs it "
            "(it should be a COPY of rentals.db taken BEFORE the backfill started)"
        )
    conn = _ro_conn(BASELINE_DB)
    yield conn
    conn.close()


def _ocr_finished():
    """True if the OCR stage of the backfill has finished writing.

    Signal: the backfill orchestrator appends an 'after:' line to
    /tmp/backfill_progress.log on completion. If that log can't be read, fall back
    to BACKFILL_DONE.

    The real-scale yield thresholds (>=400 recovered rows, >=50% floorplan coverage)
    only mean something against the real post-backfill rentals.db. In PR CI / fixture
    mode the live DB is ABSENT, so there is nothing real to assert — return False (skip)
    regardless of any (possibly stale, dev-box) /tmp/backfill_progress.log.
    """
    if not LIVE_DB.exists():
        return False
    if BACKFILL_DONE:
        return True
    log = Path("/tmp/backfill_progress.log")
    try:
        text = log.read_text(errors="ignore")
    except OSError:
        return False
    return "BULK BACKFILL DONE" in text or "after: sqft=" in text


_skip_until_done = pytest.mark.skipif(
    not _ocr_finished(),
    reason="backfill OCR stage not finished yet — yield thresholds are only the "
    "retrain GATE; run with BACKFILL_DONE=1 once the job exits to assert them",
)


# ===========================================================================
# 1. NON-DESTRUCTIVE INVARIANTS — must hold at EVERY instant (mid-run too).
#    The backfill may only ADD sqft/floorplan_url/floor data; never delete a row,
#    never change identity, never clobber an existing sqft.
# ===========================================================================
@pytest.mark.data
class TestNonDestructive:
    def test_row_count_unchanged(self, live, baseline):
        """The backfill is UPDATE-only — total + active row counts cannot move."""
        for label, where in [("total", ""), ("active", " WHERE is_active=1")]:
            b = _q1(baseline, f"SELECT COUNT(*) FROM listings{where}")
            l = _q1(live, f"SELECT COUNT(*) FROM listings{where}")
            assert l == b, (
                f"{label} row count changed {b} -> {l}: the backfill must be "
                "UPDATE-only (no INSERT/DELETE)"
            )

    def test_no_existing_sqft_was_overwritten(self, live, baseline):
        """Every row that HAD size_sqft>0 in the baseline must keep that exact value.

        ocr_enrich only SELECTs rows where size_sqft IS NULL/0, so it can never touch
        an already-populated sqft. This pins that guarantee against the real data.
        """
        base_sqft = {
            r["id"]: r["size_sqft"]
            for r in baseline.execute(
                "SELECT id, size_sqft FROM listings WHERE size_sqft IS NOT NULL AND size_sqft > 0"
            )
        }
        assert base_sqft, "baseline unexpectedly had no sqft rows"
        live_sqft = {
            r["id"]: r["size_sqft"]
            for r in live.execute(
                "SELECT id, size_sqft FROM listings WHERE id IN (%s)"
                % ",".join(str(i) for i in base_sqft)
            )
        }
        changed = {
            i: (base_sqft[i], live_sqft.get(i))
            for i in base_sqft
            if live_sqft.get(i) != base_sqft[i]
        }
        assert not changed, (
            f"{len(changed)} pre-existing size_sqft values were overwritten by the "
            f"backfill (sample: {dict(list(changed.items())[:5])}) — OCR must only "
            "fill rows that were NULL/0"
        )

    def test_sqft_count_is_monotonic_nondecreasing(self, live, baseline):
        """Backfill can only ADD sqft, so #rows with sqft>0 never drops."""
        b = _q1(baseline, "SELECT COUNT(*) FROM listings WHERE size_sqft > 0")
        l = _q1(live, "SELECT COUNT(*) FROM listings WHERE size_sqft > 0")
        assert l >= b, f"#rows with sqft>0 DECREASED {b} -> {l}: backfill destroyed sqft"

    def test_floorplan_url_count_is_monotonic_nondecreasing(self, live, baseline):
        b = _q1(baseline, "SELECT COUNT(*) FROM listings WHERE floorplan_url IS NOT NULL AND floorplan_url != ''")
        l = _q1(live, "SELECT COUNT(*) FROM listings WHERE floorplan_url IS NOT NULL AND floorplan_url != ''")
        assert l >= b, f"#rows with floorplan_url DECREASED {b} -> {l}"

    def test_identity_columns_untouched_on_recovered_rows(self, live, baseline):
        """On rows the backfill enriched, identity/price columns must be byte-identical.

        The backfill must not perturb price_pcm / bedrooms / postcode / source /
        property_id — only the sqft/floorplan/floor columns. We check this on the
        set of rows whose sqft went 0/NULL -> >0 (the recovered rows).
        """
        recovered_ids = [
            r["id"]
            for r in live.execute(
                "SELECT id FROM listings WHERE source='rightmove' AND size_sqft > 0"
            )
        ]
        base_no_sqft = {
            r["id"]
            for r in baseline.execute(
                "SELECT id FROM listings WHERE source='rightmove' "
                "AND (size_sqft IS NULL OR size_sqft = 0)"
            )
        }
        recovered = [i for i in recovered_ids if i in base_no_sqft]
        if not recovered:
            pytest.skip("no rows recovered yet — nothing to check identity on")
        cols = "id, source, property_id, price_pcm, bedrooms, postcode"
        b = {r["id"]: tuple(r) for r in baseline.execute(
            f"SELECT {cols} FROM listings WHERE id IN (%s)"
            % ",".join(str(i) for i in recovered))}
        l = {r["id"]: tuple(r) for r in live.execute(
            f"SELECT {cols} FROM listings WHERE id IN (%s)"
            % ",".join(str(i) for i in recovered))}
        mismatched = {i: (b[i], l[i]) for i in b if b[i] != l[i]}
        assert not mismatched, (
            f"{len(mismatched)} recovered rows had identity/price columns mutated by "
            f"the backfill (sample: {dict(list(mismatched.items())[:3])})"
        )


# ===========================================================================
# 2. PER-ROW SANITY — every NEWLY-recovered sqft must be physically plausible.
#    (Runs whenever any rows have been recovered; doesn't need the job to finish.)
# ===========================================================================
@pytest.mark.data
class TestRecoveredSqftSanity:
    @pytest.fixture(scope="class")
    def recovered_rows(self, live, baseline):
        base_no_sqft = {
            r["id"]
            for r in baseline.execute(
                "SELECT id FROM listings WHERE source='rightmove' "
                "AND (size_sqft IS NULL OR size_sqft = 0)"
            )
        }
        rows = [
            dict(r)
            for r in live.execute(
                "SELECT id, size_sqft, bedrooms FROM listings "
                "WHERE source='rightmove' AND size_sqft > 0"
            )
            if r["id"] in base_no_sqft
        ]
        if not rows:
            pytest.skip("no recovered rows yet (OCR stage not producing sqft)")
        return rows

    def test_recovered_sqft_in_sane_bounds(self, recovered_rows):
        """No recovered sqft may be a mis-parsed phone number / scan artifact.

        At most MAX_INSANE_RECOVERED_FRAC of recovered rows may fall outside the
        engine's plausibility band [100, 50000].
        """
        insane = [r for r in recovered_rows if not (SANE_SQFT_MIN <= r["size_sqft"] <= SANE_SQFT_MAX)]
        frac = len(insane) / len(recovered_rows)
        assert frac <= MAX_INSANE_RECOVERED_FRAC, (
            f"{len(insane)}/{len(recovered_rows)} ({frac:.1%}) recovered sqft are "
            f"outside the sane band [{SANE_SQFT_MIN},{SANE_SQFT_MAX}] "
            f"(sample: {[(r['id'], r['size_sqft']) for r in insane[:5]]}) — OCR mis-parse"
        )

    def test_recovered_sqft_per_bed_plausible(self, recovered_rows):
        """sqft/bed must clear the retrain floor for rows that will be trained on.

        Rows below MIN_SQFT_PER_BED are dropped by retrain (harmless), so we only
        assert that the share of implausibly-tiny-per-bed recovered rows is small —
        a flood of them signals systematic mis-parsing (e.g. reading a room dimension
        as the whole-flat total).
        """
        beds = lambda r: r["bedrooms"] if r["bedrooms"] and r["bedrooms"] > 0 else 0.5
        tiny = [r for r in recovered_rows if r["size_sqft"] / beds(r) < MIN_SQFT_PER_BED]
        frac = len(tiny) / len(recovered_rows)
        assert frac <= 0.10, (
            f"{len(tiny)}/{len(recovered_rows)} ({frac:.1%}) recovered rows have "
            f"sqft/bed < {MIN_SQFT_PER_BED} — likely a room-size mis-parse, not the "
            "whole-flat total"
        )

    def test_trainable_share_of_recovered_is_high(self, recovered_rows):
        """Most recovered sqft should land in the retrain-trainable band [150,10000].

        This is the real payoff measure: recovered-but-untrainable rows don't help
        the model. We don't hard-fail on a few, but the bulk must be usable.
        """
        trainable = [
            r for r in recovered_rows
            if TRAINABLE_SQFT_MIN <= r["size_sqft"] <= TRAINABLE_SQFT_MAX
        ]
        frac = len(trainable) / len(recovered_rows)
        assert frac >= 0.80, (
            f"only {len(trainable)}/{len(recovered_rows)} ({frac:.1%}) recovered sqft "
            f"are in the trainable band [{TRAINABLE_SQFT_MIN},{TRAINABLE_SQFT_MAX}]"
        )


# ===========================================================================
# 3. YIELD THRESHOLDS — the "backfill SUCCEEDED" gate. Only HARD-asserted once
#    the OCR stage has finished (or BACKFILL_DONE=1). These are the numbers the
#    human checks BEFORE deciding to retrain.
# ===========================================================================
@pytest.mark.data
@_skip_until_done
class TestBackfillYieldThresholds:
    @pytest.fixture(scope="class")
    def targets(self, baseline):
        """The denominator: active no-sqft rightmove rows the backfill set out to fix."""
        n = _q1(baseline,
                "SELECT COUNT(*) FROM listings WHERE source='rightmove' AND is_active=1 "
                "AND (size_sqft IS NULL OR size_sqft = 0)")
        assert n > 0, "baseline had no active no-sqft rightmove rows to recover"
        return n

    def test_floorplan_url_coverage(self, live, baseline, targets):
        """Stage 1 captured floorplan_url for >= MIN_FLOORPLAN_URL_COVERAGE_FRAC of targets."""
        base_cov = _q1(baseline,
            "SELECT COUNT(*) FROM listings WHERE source='rightmove' AND is_active=1 "
            "AND floorplan_url IS NOT NULL AND floorplan_url != ''")
        live_cov = _q1(live,
            "SELECT COUNT(*) FROM listings WHERE source='rightmove' AND is_active=1 "
            "AND floorplan_url IS NOT NULL AND floorplan_url != ''")
        gained = live_cov - base_cov
        frac = live_cov / (targets + base_cov)  # coverage over the active RM no-sqft+already universe
        assert frac >= MIN_FLOORPLAN_URL_COVERAGE_FRAC, (
            f"floorplan_url coverage {frac:.1%} (gained {gained}) below "
            f"{MIN_FLOORPLAN_URL_COVERAGE_FRAC:.0%}: stage-1 enrich under-captured"
        )

    def test_sqft_recovered_count(self, live, baseline):
        """>= MIN_SQFT_RECOVERED_ROWS active rightmove rows came back with usable sqft."""
        base_ids = {
            r[0] for r in baseline.execute(
                "SELECT id FROM listings WHERE source='rightmove' AND is_active=1 "
                "AND (size_sqft IS NULL OR size_sqft = 0)")
        }
        recovered = sum(
            1 for r in live.execute(
                "SELECT id FROM listings WHERE source='rightmove' AND is_active=1 "
                f"AND size_sqft >= {TRAINABLE_SQFT_MIN} AND size_sqft <= {TRAINABLE_SQFT_MAX}")
            if r[0] in base_ids
        )
        assert recovered >= MIN_SQFT_RECOVERED_ROWS, (
            f"only {recovered} active rightmove rows recovered a trainable sqft "
            f"(need >= {MIN_SQFT_RECOVERED_ROWS}): backfill yield too low to justify a retrain"
        )

    def test_ocr_yield_on_floorplanned_rows(self, live, baseline):
        """Of rows that ended up WITH a floorplan_url, a healthy share got sqft.

        Low yield here (vs the floorplan coverage being fine) points at OCR failing
        on the images — the dead-link / unparseable-image rate is accounted for by
        the 1 - MIN_OCR_YIELD_ON_FLOORPLANS_FRAC slack, not silently ignored.
        """
        base_no_sqft = {
            r[0] for r in baseline.execute(
                "SELECT id FROM listings WHERE source='rightmove' AND is_active=1 "
                "AND (size_sqft IS NULL OR size_sqft = 0)")
        }
        # Among the originally-no-sqft rows that now HAVE a floorplan_url, how many got sqft?
        denom = sum(
            1 for r in live.execute(
                "SELECT id FROM listings WHERE source='rightmove' AND is_active=1 "
                "AND floorplan_url IS NOT NULL AND floorplan_url != ''")
            if r[0] in base_no_sqft
        )
        if denom == 0:
            pytest.skip("no originally-no-sqft rows have a floorplan_url to OCR")
        numer = sum(
            1 for r in live.execute(
                "SELECT id FROM listings WHERE source='rightmove' AND is_active=1 "
                "AND floorplan_url IS NOT NULL AND floorplan_url != '' AND size_sqft > 0")
            if r[0] in base_no_sqft
        )
        frac = numer / denom
        assert frac >= MIN_OCR_YIELD_ON_FLOORPLANS_FRAC, (
            f"OCR yield {numer}/{denom} ({frac:.1%}) below "
            f"{MIN_OCR_YIELD_ON_FLOORPLANS_FRAC:.0%}: floorplans captured but OCR not "
            "extracting sqft from them"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
