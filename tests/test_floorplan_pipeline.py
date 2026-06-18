"""
WS3 — FLOORPLAN → OCR → SQFT pipeline revival (TDD).

The OCR ENGINE works (Tesseract 5.5.2; FloorplanExtractor mature + tested). What is
DEAD since 2026-01-14 is floorplan_url CAPTURE — coverage by first_seen went
Dec2025 68% → Jan2026 38% → June2026 0/6990. Three root causes, two of which this
file's tests pin (the third, the Postgres/SQLite split, is covered by the wiring +
the enricher being routed before ocr_enrich on the same Postgres DB):

  (1) The floorplan_enricher Rightmove URL-extractor (`_extract_rightmove`) matches
      ZERO on live pages: Rightmove changed floorplan URLs from `_FLP_` to
      `/property-floorplan/` AND dropped <script id="__NEXT_DATA__">. So even once the
      enricher is wired, it captures no floorplan_url for Rightmove (= 86% of the
      no-sqft hole).  ← TestRightmoveFloorplanUrlExtraction  (RED now)

  (2) The enricher is wired into NO GitHub workflow. daily-scrape.yml runs the scrape
      then scripts/ocr_enrich.py, but ocr_enrich only SELECTs rows WHERE floorplan_url
      IS NOT NULL — so with no upstream capture step it STARVES. The fix inserts an
      `enrich-floorplans --source rightmove` step BEFORE the OCR step.
      ← TestWorkflowWiring  (RED now)

  (3) The OCR engine itself is proven on a REAL floorplan image (no mocks).
      ← TestOcrEngineOnRealFloorplan  (GREEN — the engine was never the problem)

Run:
  python3 -m pytest tests/test_floorplan_pipeline.py -v

These are unit/self-contained (real HTML fixture, real image fixture, real YAML) —
no live DB, so they run in default PR CI (not gated by conftest's no-DB skip).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures" / "extension"
FLOORPLAN_IMG = ROOT / "tests" / "fixtures" / "floorplans" / "savills_floorplan_782sqft.png"
WORKFLOW = ROOT / ".github" / "workflows" / "daily-scrape.yml"


# ---------------------------------------------------------------------------
# A scrapy-Response stand-in: the enricher's _extract_rightmove reads
# response.text (full HTML) and response.css('script#__NEXT_DATA__::text').get().
# Live Rightmove has NO __NEXT_DATA__, so .get() returns None — we model that.
# ---------------------------------------------------------------------------
class _FakeSelector:
    def get(self):
        return None


class FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def css(self, _selector: str):
        # Live Rightmove pages no longer ship <script id="__NEXT_DATA__">.
        return _FakeSelector()


def _load_fixture(name: str) -> str:
    p = FIXTURES / name
    if not p.exists():
        pytest.skip(f"fixture missing: {p}")
    return p.read_text(encoding="utf-8", errors="ignore")


@pytest.fixture(scope="module")
def enricher():
    """A floorplan_enricher spider instance (constructor needs no live DB)."""
    from property_scraper.spiders.floorplan_enricher import FloorplanEnricherSpider
    # db_path is only touched on DB writes, not by _extract_rightmove.
    return FloorplanEnricherSpider(source="rightmove", db_path=":memory:")


# ===========================================================================
# (1) Rightmove floorplan-URL extraction — the _FLP_ → /property-floorplan/ bug
# ===========================================================================
class TestRightmoveFloorplanUrlExtraction:
    """`_extract_rightmove` must capture the NEW /property-floorplan/ URL from the
    live page HTML (the old `_FLP_`-only regex + dead __NEXT_DATA__ path finds
    nothing). RED until the extractor learns the /property-floorplan/ pattern."""

    def test_captures_property_floorplan_url_with_sqft_listing(self, enricher):
        html = _load_fixture("rightmove_169944029.html")
        # Sanity: this REAL page truly has NO _FLP_ and DOES carry /property-floorplan/.
        assert "_FLP_" not in html, "fixture invalid: expected no legacy _FLP_ token"
        assert "property-floorplan" in html

        floorplan_url, _desc = enricher._extract_rightmove(FakeResponse(html))

        assert floorplan_url, (
            "enricher captured NO floorplan_url from a live Rightmove page — the "
            "_FLP_-only regex no longer matches; must detect '/property-floorplan/'"
        )
        assert "/property-floorplan/" in floorplan_url
        # Must be the full-size image, not the resized thumbnail (…_max_296x197).
        assert "_max_" not in floorplan_url, (
            f"captured the thumbnail, not the full-size floorplan: {floorplan_url}"
        )
        # Matches the decoded ground truth for this listing.
        assert floorplan_url.endswith(
            "/property-floorplan/ce4a84bec/169944029/"
            "ce4a84bec148f470b62a457f79068a53.jpeg"
        ), floorplan_url

    def test_captures_property_floorplan_url_no_sqft_listing(self, enricher):
        html = _load_fixture("rightmove_163282418.html")
        assert "_FLP_" not in html
        floorplan_url, _desc = enricher._extract_rightmove(FakeResponse(html))
        assert floorplan_url and "/property-floorplan/" in floorplan_url, (
            f"no floorplan_url captured for the no-sqft listing: {floorplan_url!r}"
        )
        assert "163282418" in floorplan_url

    def test_no_false_positive_on_pageless_html(self, enricher):
        """A page with no floorplan media yields None (so we don't write garbage)."""
        floorplan_url, _desc = enricher._extract_rightmove(
            FakeResponse("<html><body>no media here</body></html>")
        )
        assert floorplan_url is None


# ===========================================================================
# (3) OCR engine on a REAL floorplan image — proves the engine was never the bug.
#     (NO mocks: real Tesseract on a real PNG.)
# ===========================================================================
class TestOcrEngineOnRealFloorplan:
    def test_extract_sqft_from_real_floorplan_file(self):
        from property_scraper.utils.floorplan_extractor import (
            FloorplanExtractor,
            OCR_AVAILABLE,
        )

        if not OCR_AVAILABLE:
            pytest.skip("pytesseract/tesseract not available")
        if not FLOORPLAN_IMG.exists():
            pytest.skip(f"floorplan fixture missing: {FLOORPLAN_IMG}")

        extractor = FloorplanExtractor()
        result = extractor.extract_from_file(str(FLOORPLAN_IMG))

        assert result is not None and result.total_sqft is not None, (
            "OCR returned no sqft from the real floorplan image"
        )
        # Real OCR ground truth for this fixture is 782 sqft; allow ±1% OCR jitter.
        assert 774 <= result.total_sqft <= 790, (
            f"expected ~782 sqft from real OCR, got {result.total_sqft}"
        )

    def test_extract_sqft_from_real_floorplan_bytes(self):
        """The enricher/ocr_enrich path downloads bytes then OCRs — exercise that API."""
        from property_scraper.utils.floorplan_extractor import (
            FloorplanExtractor,
            OCR_AVAILABLE,
        )

        if not OCR_AVAILABLE:
            pytest.skip("pytesseract/tesseract not available")
        if not FLOORPLAN_IMG.exists():
            pytest.skip(f"floorplan fixture missing: {FLOORPLAN_IMG}")

        extractor = FloorplanExtractor()
        result = extractor.extract_from_bytes(FLOORPLAN_IMG.read_bytes())
        assert result and result.total_sqft and 774 <= result.total_sqft <= 790, (
            f"bytes path: expected ~782 sqft, got {result and result.total_sqft}"
        )


# ===========================================================================
# (2) Workflow wiring — enrich-floorplans MUST run before scripts/ocr_enrich.py
#     in daily-scrape.yml (else OCR starves on NULL floorplan_url). RED now.
# ===========================================================================
class TestWorkflowWiring:
    @pytest.fixture(scope="class")
    def workflow_text(self):
        if not WORKFLOW.exists():
            pytest.skip(f"workflow missing: {WORKFLOW}")
        return WORKFLOW.read_text(encoding="utf-8")

    # NB: assertions parse the workflow as TEXT (no PyYAML) so they run in CI without
    # an extra dependency. Step ordering == position in the file (steps are sequential),
    # so character offsets of the actual run commands encode run order.

    def test_enrich_floorplans_step_exists(self, workflow_text):
        assert "enrich-floorplans" in workflow_text, (
            "daily-scrape.yml has NO `enrich-floorplans` step — the floorplan_enricher "
            "is wired into no workflow, so floorplan_url is never captured for scheduled "
            "runs and ocr_enrich.py starves."
        )

    def test_enrich_floorplans_targets_rightmove(self, workflow_text):
        import re

        assert re.search(r"enrich-floorplans[^\n]*rightmove", workflow_text), (
            "enrich-floorplans step must cover rightmove (86% of the no-sqft hole)"
        )

    def test_enrich_runs_before_ocr_enrich(self, workflow_text):
        enrich = workflow_text.find("cli.main enrich-floorplans")
        ocr = workflow_text.find("ocr_enrich.py")
        assert ocr >= 0, "expected the existing scripts/ocr_enrich.py step"
        assert enrich >= 0, "expected an enrich-floorplans run step (capture floorplan_url first)"
        assert enrich < ocr, (
            "enrich-floorplans must run BEFORE ocr_enrich.py so floorplan_url is "
            "populated before OCR reads it"
        )

    def test_enrich_runs_after_scrape(self, workflow_text):
        scrape = workflow_text.find("cli.main scrape")
        enrich = workflow_text.find("cli.main enrich-floorplans")
        assert scrape >= 0 and enrich > scrape, (
            "enrich-floorplans must run AFTER the scrape (enrich existing rows), "
            f"got scrape@{scrape} enrich@{enrich}"
        )

    def test_enrich_writes_to_postgres_db_ocr_reads(self, workflow_text):
        """The scheduled run writes PROD Postgres; the enrich step must target the
        SAME DB ocr_enrich reads (via --postgres / POSTGRES_URL), else the captured
        floorplan_url lands in SQLite and OCR (on Postgres) still starves."""
        import re

        assert re.search(r"enrich-floorplans[^\n]*(?:--postgres|POSTGRES_URL)", workflow_text), (
            "enrich-floorplans in the Postgres-mode daily run must write Postgres "
            "(pass --postgres or set POSTGRES_URL) so OCR reads the rows it captured"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
