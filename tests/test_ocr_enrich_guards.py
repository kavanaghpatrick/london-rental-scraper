"""Pure-function unit tests for scripts/ocr_enrich.py's data-protection guards.

These lock down the two pieces of logic that protect the platform's data integrity in
the floorplan-OCR backfill, WITHOUT any OCR engine or database:

  1. sqft_passes_sanity_gate — a mis-read OCR number (a room dimension, a phone number,
     a scan artifact) must NOT be written as a flat's size_sqft and poison the model.
  2. select_field_updates — the NO-OVERWRITE rule: OCR may only FILL a missing size_sqft,
     never clobber a value that was already scraped from the listing.

CI-SAFETY: ocr_enrich.py imports the OCR engine LAZILY (inside main()), so this test
imports the module and exercises the pure helpers with NO pytesseract/Pillow/tesseract
installed — it runs for real in PR CI (it does not skip). That is the whole point: the
guard that stops bad data reaching the DB is now executed on every PR, not just where
tesseract happens to be installed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_ocr_enrich():
    """Import scripts/ocr_enrich.py by path (it's a script, not a package module).

    Must succeed with NO OCR deps — proves the lazy-import refactor keeps the pure
    decision logic testable in CI.
    """
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "ocr_enrich", str(ROOT / "scripts" / "ocr_enrich.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ocr = _load_ocr_enrich()


# ---------------------------------------------------------------------------------
# sqft sanity gate
# ---------------------------------------------------------------------------------
class TestSqftSanityGate:
    def test_plausible_flat_accepted(self):
        # 900 sqft, 2 bed, £2700 pcm -> ppsf=3.0 (>=3), spb=450 (in 80..4000) -> accept.
        assert ocr.sqft_passes_sanity_gate(900, 2, 2700) == 900

    def test_none_or_zero_sqft_rejected(self):
        assert ocr.sqft_passes_sanity_gate(None, 2, 2700) is None
        assert ocr.sqft_passes_sanity_gate(0, 2, 2700) is None

    def test_below_absolute_min_rejected(self):
        # 40 sqft is a room dimension mis-read as the total — must be rejected.
        assert ocr.sqft_passes_sanity_gate(40, 1, 1000) is None
        assert ocr.sqft_passes_sanity_gate(ocr.SQFT_SANITY_MIN - 1, 1, None) is None

    def test_above_absolute_max_rejected(self):
        # A scan-artifact number well above any London home.
        assert ocr.sqft_passes_sanity_gate(50000, 5, 50000) is None
        assert ocr.sqft_passes_sanity_gate(ocr.SQFT_SANITY_MAX + 1, 4, None) is None

    def test_ppsf_too_low_rejected(self):
        # 8000 sqft at £2000 pcm -> ppsf=0.25 (< 3): economically impossible, reject.
        assert ocr.sqft_passes_sanity_gate(8000, 4, 2000) is None

    def test_ppsf_too_high_rejected(self):
        # 200 sqft at £8000 pcm -> ppsf=40 (> 30): reject. (200 is also < min but use 500.)
        assert ocr.sqft_passes_sanity_gate(500, 2, 30000) is None

    def test_sqft_per_bed_too_high_rejected(self):
        # 9000 sqft / 1 bed = 9000 spb (> 4000) AND in absolute range -> reject on spb.
        assert ocr.sqft_passes_sanity_gate(9000, 1, 27000) is None

    def test_sqft_per_bed_too_low_rejected(self):
        # 200 sqft / 3 beds = 66 spb (< 80). Use 300/4=75 to stay in absolute range.
        assert ocr.sqft_passes_sanity_gate(300, 4, 1200) is None

    def test_missing_price_skips_ppsf_check(self):
        # No price -> ppsf check skipped; spb still checked; absolute range still checked.
        assert ocr.sqft_passes_sanity_gate(900, 2, None) == 900

    def test_missing_beds_skips_per_bed_check(self):
        # Studio / 0 beds -> spb check skipped; ppsf + absolute range still apply.
        assert ocr.sqft_passes_sanity_gate(500, 0, 3000) == 500
        assert ocr.sqft_passes_sanity_gate(500, None, 3000) == 500


# ---------------------------------------------------------------------------------
# no-overwrite field selection
# ---------------------------------------------------------------------------------
class TestNoOverwriteFieldSelection:
    def test_fills_missing_sqft(self):
        updates = ocr.select_field_updates({"success": True, "sqft": 800, "orig_sqft": None})
        assert ("size_sqft", 800) in updates

    def test_never_overwrites_existing_scraped_sqft(self):
        # orig_sqft present (scraped) -> OCR sqft must NOT be written. THE core guard.
        updates = ocr.select_field_updates({"success": True, "sqft": 800, "orig_sqft": 650})
        cols = [c for c, _ in updates]
        assert "size_sqft" not in cols

    def test_existing_sqft_still_lets_floor_fields_through(self):
        # No-overwrite is sqft-specific; floor data can still be filled alongside.
        updates = ocr.select_field_updates(
            {"success": True, "sqft": 800, "orig_sqft": 650, "floor_count": 2}
        )
        cols = dict(updates)
        assert "size_sqft" not in cols
        assert cols.get("floor_count") == 2

    def test_floor_flags_selected_when_true_only(self):
        updates = ocr.select_field_updates(
            {
                "success": True,
                "sqft": None,
                "orig_sqft": None,
                "floor_data": {"has_basement": True, "has_ground": False, "has_first_floor": True},
            }
        )
        cols = dict(updates)
        assert cols.get("has_basement") is True
        assert cols.get("has_first_floor") is True
        assert "has_ground" not in cols  # False flags are not written

    def test_no_data_no_updates(self):
        assert ocr.select_field_updates({"success": True, "sqft": None, "orig_sqft": None}) == []


def test_module_imports_without_ocr_deps():
    """Self-documenting: the module loaded above with no OCR engine present.

    If a future change re-adds a module-scope `import pytesseract` + sys.exit, this
    file fails to import and this test (and the whole file) errors loudly — flagging
    that the pure guards have become un-testable in CI again.
    """
    assert hasattr(ocr, "sqft_passes_sanity_gate")
    assert hasattr(ocr, "select_field_updates")
