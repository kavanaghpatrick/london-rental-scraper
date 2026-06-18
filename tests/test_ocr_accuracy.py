"""
test_ocr_accuracy.py — R2: OCR-ACCURACY VALIDATION for the floorplan -> sqft backfill.

WHY THIS FILE EXISTS
--------------------
The bulk backfill recovers `size_sqft` for ~1,932 active no-sqft Rightmove rows by
OCR-ing their floorplan images (the new `/property-floorplan/` URL format the revived
enricher captures). Before we let that recovered sqft flow into a RETRAIN, we must prove
the OCR'd numbers are CORRECT, not garbage. "More rows" is worthless if the rows are wrong
— a bad sqft poisons `ppsf`, the size interactions, and the whole price model.

This file is the acceptance gate for "the OCR'd sqft is trustworthy". It is REAL-DATA,
NO-MOCKS: it runs the actual `FloorplanExtractor` (Tesseract) on real floorplan images
and checks the extracted area against an INDEPENDENT ground truth.

THREE GROUND-TRUTH STRATEGIES (no human labelling required)
-----------------------------------------------------------
(GT-A) RAW-TEXT GROUND TRUTH (self-checking, network, the workhorse):
    For each OCR'd row, re-download its floorplan image, re-run OCR, and read the
    'Gross Internal Area … NNN sq ft' / 'NNN sq ft' number STRAIGHT OUT OF the OCR
    raw_text with an independent regex. That regex-from-text value is the ground truth;
    we check the stored/extracted `total_sqft` agrees with it. This catches the failure
    mode that matters — the hierarchical parser picking the WRONG number off the page
    (a room dimension, a sqm value, the largest-on-page heuristic firing wrongly).

(GT-B) STORED-VALUE CONSISTENCY (network):
    The backfill wrote `size_sqft` from `FloorplanExtractor.total_sqft`. Re-OCR the same
    image and confirm we reproduce the stored value (determinism / no write-time bug).

(GT-C) LABELLED FIXTURE (offline, always-on):
    The committed `savills_floorplan_782sqft.png` has a known total of 782 sq ft. The
    methodology's accuracy maths + bad-extraction detector are exercised against it with
    ZERO network, so the gate's *logic* is always tested even in PR CI.

ACCURACY THRESHOLDS (the gate)
------------------------------
On the sample of rows where OCR returned a sqft AND the raw text contains an explicit
'sq ft' number (GT-A applies):
    - >= 85% of rows within +/-10% of the raw-text ground truth, AND
    - median absolute error <= 5%.
Rows where OCR returned None (no 'sq ft' text on the plan) are a COVERAGE miss, not an
ACCURACY miss — they are reported separately and never counted as "wrong" (the extractor
correctly abstained rather than inventing a number).

BAD-EXTRACTION DETECTOR (`is_implausible_sqft`)
-----------------------------------------------
Independent of ground truth, a recovered sqft is flagged garbage if:
    - outside [150, 10000]  (matches retrain_canonical SQFT_MIN/MAX), or
    - sqft/beds < 70        (matches retrain_canonical SQFT_PER_BED_MIN), or
    - sqft/beds > 4000      (one bedroom in a 4000+ sqft "room" => mis-parse).
This is the cheap, no-network guard the retrain's own quality filter ALSO applies, so a
bad OCR value can never silently become a training row.

MARKERS / CI BEHAVIOUR
----------------------
  - GT-C tests + all the pure logic tests are UNIT (`pytest.mark.unit`): always run.
  - GT-A / GT-B tests are LIVE (`pytest.mark.live`) AND DATA (`pytest.mark.data`):
    they need the network (download images) and a populated rentals.db. They are
    OPT-IN (`pytest -m live`) and auto-skip with no DB, so PR CI stays green while the
    real accuracy read runs on demand / in the backfill gate.

CONCURRENCY: the live backfill is WRITING output/rentals.db. These tests open it in
SQLite IMMUTABLE mode (read-only, no locks) — they never write and never block the job.

Run:
  pytest tests/test_ocr_accuracy.py -v                 # offline logic + fixture (GT-C)
  pytest tests/test_ocr_accuracy.py -v -m live         # real accuracy read (GT-A/B)
"""
from __future__ import annotations

import os
import re
import sqlite3
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = Path(
    os.environ.get("SCRAPER_DB_PATH", ROOT / "output" / "rentals.db")
)
FIXTURE_IMG = ROOT / "tests" / "fixtures" / "floorplans" / "savills_floorplan_782sqft.png"
FIXTURE_GROUND_TRUTH = 782

# ── Accuracy gate thresholds ────────────────────────────────────────────────
WITHIN_TOL_PCT = 10.0       # a row is "accurate" if |ocr - gt| / gt <= 10%
MIN_FRAC_WITHIN_TOL = 0.85  # >= 85% of GT-A rows must be within tolerance
MAX_MEDIAN_ABS_ERR_PCT = 5.0
MIN_GTA_SAMPLE = 5          # need at least this many raw-text-anchored rows to gate

# ── Bad-extraction detector bounds (mirror retrain_canonical quality filter) ─
SQFT_MIN, SQFT_MAX = 150, 10000
SQFT_PER_BED_MIN = 70
SQFT_PER_BED_MAX = 4000


# ===========================================================================
# Methodology primitives (pure — these are the gate's reusable logic)
# ===========================================================================
def is_implausible_sqft(sqft, bedrooms) -> bool:
    """Bad-extraction detector. True => the recovered sqft is garbage and must NOT
    reach training. Independent of any ground truth (a cheap structural sanity gate)."""
    if sqft is None:
        return False  # None = abstained, not garbage (a coverage miss)
    try:
        sqft = float(sqft)
    except (TypeError, ValueError):
        return True
    if not (SQFT_MIN <= sqft <= SQFT_MAX):
        return True
    if bedrooms and bedrooms > 0:
        spb = sqft / bedrooms
        if spb < SQFT_PER_BED_MIN or spb > SQFT_PER_BED_MAX:
            return True
    return False


# Independent ground-truth regex: pull an explicit 'NNN sq ft' off the OCR raw text.
# Deliberately DIFFERENT (simpler/stricter) from the extractor's hierarchical parser so
# it is a genuine cross-check, not the same code grading its own homework. We take the
# MAX 'sq ft' mention (the GIA total is the largest sq-ft figure on a floorplan; room
# dimensions are smaller), bounded to a believable property total.
_SQFT_IN_TEXT = re.compile(r"(\d[\d,]*)\s*(?:sq\.?\s*ft|sqft|square\s*feet)", re.I)


def raw_text_ground_truth_sqft(raw_text: str):
    """Independent ground truth: the largest believable 'sq ft' number in the OCR text,
    or None if the plan states no area in sq ft. This is GT-A's anchor."""
    if not raw_text:
        return None
    vals = []
    for m in _SQFT_IN_TEXT.findall(raw_text):
        try:
            v = int(m.replace(",", ""))
        except ValueError:
            continue
        if SQFT_MIN <= v <= SQFT_MAX:
            vals.append(v)
    return max(vals) if vals else None


def abs_err_pct(a, b) -> float:
    return abs(a - b) / b * 100.0


# ===========================================================================
# (GT-C) Offline, always-on: methodology logic on the labelled fixture image.
#        Proves the gate's accuracy maths + detector work with NO network.
# ===========================================================================
def _extractor_or_skip():
    try:
        from property_scraper.utils.floorplan_extractor import (
            FloorplanExtractor,
            OCR_AVAILABLE,
        )
    except Exception as e:  # pragma: no cover - import guard
        pytest.skip(f"floorplan_extractor import failed: {e}")
    if not OCR_AVAILABLE:
        pytest.skip("OCR deps (pytesseract/Pillow/tesseract) not available")
    try:
        return FloorplanExtractor()
    except RuntimeError as e:  # pragma: no cover
        pytest.skip(str(e))


@pytest.mark.unit
class TestBadExtractionDetector:
    """The garbage-sqft guard, pinned by example. Pure logic — no OCR, no DB."""

    def test_plausible_london_flat_passes(self):
        assert is_implausible_sqft(900, 2) is False
        assert is_implausible_sqft(539, 2) is False   # real backfill row id=163's stored
        assert is_implausible_sqft(4798, 6) is False  # real backfill row id=724 (large house)

    def test_none_is_not_garbage(self):
        # OCR abstaining (no sq-ft text) is a coverage miss, NOT a bad extraction.
        assert is_implausible_sqft(None, 2) is False

    def test_too_small_total_flagged(self):
        assert is_implausible_sqft(40, 1) is True     # < SQFT_MIN (a room dim mis-read as total)

    def test_too_large_total_flagged(self):
        assert is_implausible_sqft(50000, 5) is True  # > SQFT_MAX (scan-artifact number)

    def test_sqft_per_bed_too_low_flagged(self):
        # 100 sqft across 2 beds = 50 sqft/bed < 70 => mis-parse / sub-room reading.
        assert is_implausible_sqft(100, 2) is True

    def test_sqft_per_bed_too_high_flagged(self):
        # 9000 sqft for a 1-bed => almost certainly grabbed a wrong/huge number.
        assert is_implausible_sqft(9000, 1) is True

    def test_studio_zero_beds_only_range_checked(self):
        # 0/None beds: skip the per-bed check, only the absolute range gates.
        assert is_implausible_sqft(450, 0) is False
        assert is_implausible_sqft(450, None) is False
        assert is_implausible_sqft(60, 0) is True


@pytest.mark.unit
class TestRawTextGroundTruthRegex:
    """The independent GT-A anchor: extract the area from OCR text with a DIFFERENT
    code path than the extractor, so it's a real cross-check."""

    def test_picks_gross_internal_area_number(self):
        txt = "Approximate Gross Internal Area 72.6 sq m / 782 sq ft\nKitchen 12'3 x 9'8"
        assert raw_text_ground_truth_sqft(txt) == 782

    def test_takes_largest_believable_sqft_not_room_dim(self):
        # Room measurements in sq ft are smaller than the GIA total; take the max.
        txt = "Bedroom 120 sq ft\nReception 210 sq ft\nTotal 950 sq ft"
        assert raw_text_ground_truth_sqft(txt) == 950

    def test_ignores_out_of_range_numbers(self):
        txt = "ref 99999 sq ft photo\nTotal 540 sq ft"  # 99999 > SQFT_MAX => ignored
        assert raw_text_ground_truth_sqft(txt) == 540

    def test_no_sqft_text_returns_none(self):
        assert raw_text_ground_truth_sqft("FOURTH FLOOR\nDimensions only, no area") is None
        assert raw_text_ground_truth_sqft("") is None


@pytest.mark.unit
@pytest.mark.skipif(not FIXTURE_IMG.exists(), reason=f"fixture missing: {FIXTURE_IMG}")
class TestMethodologyOnLabelledFixture:
    """End-to-end methodology on the REAL labelled image (782 sq ft), offline.

    This is the always-on proof that: real OCR runs, GT-A regex agrees with the
    extractor, error maths is correct, and the row passes the gate — all with NO
    network, so the gate's logic can never silently rot in PR CI."""

    def test_real_ocr_matches_raw_text_ground_truth(self):
        ext = _extractor_or_skip()
        data = ext.extract_from_file(str(FIXTURE_IMG))
        assert data.total_sqft is not None, "real OCR returned no sqft on the labelled fixture"

        gt = raw_text_ground_truth_sqft(data.raw_text)
        assert gt is not None, (
            "GT-A anchor found no 'sq ft' in raw text — methodology can't self-check this "
            f"image; raw head: {(data.raw_text or '')[:160]!r}"
        )
        # The extractor's hierarchical pick must agree with the independent text anchor.
        assert abs_err_pct(data.total_sqft, gt) <= WITHIN_TOL_PCT, (
            f"extractor total_sqft={data.total_sqft} disagrees with raw-text GT={gt}"
        )
        # And both must agree with the human-labelled ground truth.
        assert abs_err_pct(data.total_sqft, FIXTURE_GROUND_TRUTH) <= WITHIN_TOL_PCT

    def test_fixture_row_is_plausible(self):
        # A real 782 sqft / 2-bed-ish flat must NOT be flagged garbage.
        assert is_implausible_sqft(FIXTURE_GROUND_TRUTH, 2) is False


# ===========================================================================
# (GT-A / GT-B) LIVE accuracy read on the REAL backfilled DB.
#   Marked live+data: opt-in (`-m live`), auto-skipped with no populated DB.
#   Reads rentals.db in IMMUTABLE mode (never blocks the running backfill).
# ===========================================================================
def _db_available() -> bool:
    return DB_PATH.exists() and DB_PATH.stat().st_size > 0


def _immutable_conn():
    """Read-only, lock-free connection — safe while the backfill writes the same file."""
    conn = sqlite3.connect(f"file:{DB_PATH}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _ocr_sample_rows(limit=40):
    """Rows the backfill OCR'd: active Rightmove with stored sqft AND a NEW-format
    `/property-floorplan/` URL (the format the revived enricher captures + OCRs).
    Legacy `_FLP_` URLs are excluded — they are stale and mostly 404 (their sqft did
    not come from OCR), so they'd add download noise without adding ground truth."""
    conn = _immutable_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, property_id, size_sqft, bedrooms, floorplan_url
            FROM listings
            WHERE source = 'rightmove' AND is_active = 1
              AND size_sqft > 0
              AND floorplan_url LIKE '%property-floorplan%'
            ORDER BY id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _download(url, timeout=20):
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "image/*,*/*;q=0.8",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200 and r.content and len(r.content) > 1000:
            return r.content
    except Exception:
        pass
    return None


@pytest.mark.live
@pytest.mark.data
@pytest.mark.skipif(not _db_available(), reason="no populated rentals.db")
class TestLiveOcrAccuracyOnBackfill:
    """The real accuracy read: re-OCR backfilled rows and grade them against the
    independent raw-text ground truth (GT-A) + reproduce the stored value (GT-B)."""

    @pytest.fixture(scope="class")
    def graded(self):
        ext = _extractor_or_skip()
        rows = _ocr_sample_rows()
        if not rows:
            pytest.skip(
                "no OCR'd /property-floorplan/ rows yet — backfill STAGE 2 (OCR) "
                "has not produced rows; re-run once it has"
            )
        results = []
        for r in rows:
            img = _download(r["floorplan_url"])
            if img is None:
                results.append({**r, "ocr": None, "gt": None, "status": "dl_fail"})
                continue
            try:
                data = ext.extract_from_bytes(img)
            except Exception as e:  # pragma: no cover - real OCR rarely throws
                results.append({**r, "ocr": None, "gt": None, "status": f"ocr_err:{e}"[:40]})
                continue
            ocr = data.total_sqft
            gt = raw_text_ground_truth_sqft(data.raw_text)
            results.append(
                {**r, "ocr": ocr, "gt": gt, "status": "ok" if ocr is not None else "ocr_none"}
            )
        return results

    def test_no_recovered_sqft_is_implausible(self, graded):
        """GATE 1 (bad-extraction): every backfilled sqft already in the DB must pass
        the garbage detector. A failure here means a wrong number is already a candidate
        training row — block the retrain."""
        bad = [
            g for g in graded
            if is_implausible_sqft(g["size_sqft"], g["bedrooms"])
        ]
        assert not bad, (
            "implausible stored size_sqft on backfilled rows (would poison training): "
            + ", ".join(f"id={b['id']} sqft={b['size_sqft']} beds={b['bedrooms']}" for b in bad[:10])
        )

    def test_ocr_accuracy_meets_threshold(self, graded):
        """GATE 2 (accuracy): on rows where OCR returned a value AND the plan states an
        explicit sq-ft area (GT-A applies), >=85% within +/-10% and median abs err <=5%."""
        gta = [g for g in graded if g["ocr"] is not None and g["gt"] is not None]
        if len(gta) < MIN_GTA_SAMPLE:
            pytest.skip(
                f"only {len(gta)} raw-text-anchored rows (< {MIN_GTA_SAMPLE}); "
                "insufficient GT-A sample to gate accuracy — re-run with more backfilled rows"
            )
        errs = [abs_err_pct(g["ocr"], g["gt"]) for g in gta]
        within = sum(1 for e in errs if e <= WITHIN_TOL_PCT)
        frac = within / len(gta)
        median_err = statistics.median(errs)
        assert frac >= MIN_FRAC_WITHIN_TOL, (
            f"only {within}/{len(gta)} ({frac:.0%}) OCR rows within +/-{WITHIN_TOL_PCT:.0f}% "
            f"of the raw-text ground truth (need >= {MIN_FRAC_WITHIN_TOL:.0%})"
        )
        assert median_err <= MAX_MEDIAN_ABS_ERR_PCT, (
            f"median abs OCR error {median_err:.1f}% > {MAX_MEDIAN_ABS_ERR_PCT:.0f}% gate"
        )

    def test_stored_sqft_reproduces_under_reocr(self, graded):
        """GATE 3 (determinism / write-time integrity): where re-OCR returns a value,
        it must reproduce the value the backfill stored (within tolerance). Catches a
        write-time bug where the DB sqft != what the extractor actually produced."""
        reproduced = [
            g for g in graded
            if g["ocr"] is not None
            and abs_err_pct(g["ocr"], g["size_sqft"]) <= WITHIN_TOL_PCT
        ]
        candidates = [g for g in graded if g["ocr"] is not None]
        if not candidates:
            pytest.skip("no rows produced an OCR value on re-run (all abstained / dl_fail)")
        frac = len(reproduced) / len(candidates)
        assert frac >= MIN_FRAC_WITHIN_TOL, (
            f"only {len(reproduced)}/{len(candidates)} ({frac:.0%}) re-OCR'd rows reproduce "
            f"the stored size_sqft — possible write-time mismatch between extractor and DB"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
