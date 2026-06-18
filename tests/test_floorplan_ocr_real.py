#!/usr/bin/env python3
"""WS3 — REAL OCR-engine guard for the floorplan -> sqft pipeline.

The floorplan -> OCR -> sqft pipeline went dead since 2026-01-14, but the OCR ENGINE
itself was never the problem — what broke is floorplan_url CAPTURE (the enricher is
wired into no workflow; ocr_enrich starves because it only SELECTs rows that already
have a floorplan_url). This test pins down the half that DOES work so the revival
can't accidentally regress it: the real FloorplanExtractor (Tesseract) reading a REAL
captured Savills floorplan image must recover the ground-truth area.

NO MOCKS: this runs Tesseract on tests/fixtures/floorplans/savills_floorplan_782sqft.png
(a real floorplan with a known total of 782 sq ft). If Tesseract isn't installed the
test SKIPS (so CI without the OCR system dep stays green) rather than failing.

Run: python3 -m pytest tests/test_floorplan_ocr_real.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "floorplans" / "savills_floorplan_782sqft.png"
GROUND_TRUTH_SQFT = 782


def _extractor_or_skip():
    """Return a FloorplanExtractor, or skip if OCR deps / tesseract are unavailable."""
    try:
        from property_scraper.utils.floorplan_extractor import FloorplanExtractor, OCR_AVAILABLE
    except Exception as e:  # import error (missing pytesseract/Pillow)
        pytest.skip(f"floorplan_extractor import failed: {e}")
    if not OCR_AVAILABLE:
        pytest.skip("OCR deps (pytesseract/Pillow) not available")
    try:
        return FloorplanExtractor()
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def extractor():
    return _extractor_or_skip()


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"floorplan fixture missing: {FIXTURE}")
class TestRealFloorplanOCR:
    """The OCR engine recovers the real ground-truth sqft from a real floorplan."""

    def test_total_sqft_is_782(self, extractor):
        data = extractor.extract_from_file(str(FIXTURE))
        assert data.total_sqft == GROUND_TRUTH_SQFT, (
            f"OCR recovered total_sqft={data.total_sqft}, expected {GROUND_TRUTH_SQFT}; "
            f"raw_text head: {(data.raw_text or '')[:200]!r}"
        )

    def test_extract_from_bytes_matches_file(self, extractor):
        # The enricher/ocr_enrich path reads bytes off the network; ensure the bytes
        # entrypoint yields the SAME area as the file entrypoint (no path-only quirk).
        data = extractor.extract_from_bytes(FIXTURE.read_bytes())
        assert data.total_sqft == GROUND_TRUTH_SQFT

    def test_sqft_in_plausible_range(self, extractor):
        # Sanity gate the downstream COALESCE/UPDATE relies on: a real floorplan must
        # land a believable London-flat area, not 0 / a phone number / a giant scan artifact.
        data = extractor.extract_from_file(str(FIXTURE))
        assert data.total_sqft is not None
        assert 100 <= data.total_sqft <= 50000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
