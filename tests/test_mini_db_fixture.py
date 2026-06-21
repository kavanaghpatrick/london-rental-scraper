"""WORKSTREAM B — mini rentals.db fixture is wired so the structural data-validation
suites EXECUTE in PR CI (instead of all-skipping when the live DB is absent).

This file is itself a CI-safe behavioral guard:

  * It proves the committed fixture (tests/fixtures/mini_rentals.db) exists, has the
    real pipeline schema, and contains representative rows across sources.
  * It proves conftest WIRES that fixture in when no live output/rentals.db is present
    (the CI condition) — i.e. _db_available() is True via the fixture, so the
    structural test_scrape_validation / test_backfill_acceptance asserts run for real
    rather than skipping.
  * It asserts REAL facts about the fixture (counts, integrity), so it can't pass
    vacuously.

It has NO DB/network/node dependency of its own (it builds the fixture in a tmp dir
and reads the committed copy), so it always runs in PR CI — and is on the
anti-silent-skip CRITICAL allowlist.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COMMITTED_FIXTURE = ROOT / "tests" / "fixtures" / "mini_rentals.db"


# --------------------------------------------------------------------------- #
# The committed fixture exists and is the real schema.
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_committed_fixture_exists_and_nonempty():
    assert COMMITTED_FIXTURE.exists(), (
        f"committed mini fixture missing at {COMMITTED_FIXTURE} — build it with "
        "`python3 tests/fixtures/build_mini_rentals.py` and commit it"
    )
    assert COMMITTED_FIXTURE.stat().st_size > 0
    # Keep it SMALL — a fixture that grows into the real DB defeats the purpose.
    assert COMMITTED_FIXTURE.stat().st_size < 256 * 1024, (
        "mini fixture should stay tiny (<256KB); it ballooned — did the live DB leak in?"
    )


@pytest.mark.unit
def test_fixture_has_pipeline_schema():
    """listings/price_history/scrape_runs exist with the columns the validation
    suite + dashboard queries touch."""
    conn = sqlite3.connect(f"file:{COMMITTED_FIXTURE}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"listings", "price_history", "scrape_runs"} <= tables

        cols = {r[1] for r in conn.execute("PRAGMA table_info(listings)")}
        # the historical-tracking columns the pre-scrape schema test requires
        required = {
            "source", "property_id", "first_seen", "last_seen", "is_active",
            "address_fingerprint", "price_change_count", "size_sqft",
            "floorplan_url", "postcode", "price_pcm", "bedrooms", "price_period",
        }
        missing = required - cols
        assert not missing, f"fixture listings missing columns: {missing}"
    finally:
        conn.close()


@pytest.mark.unit
def test_fixture_builder_is_deterministic(tmp_path):
    """Re-building the fixture yields identical row-level content (no clock/rng)."""
    from tests.fixtures.build_mini_rentals import build_mini_rentals

    a = build_mini_rentals(tmp_path / "a.db")
    b = build_mini_rentals(tmp_path / "b.db")

    def dump(p):
        c = sqlite3.connect(str(p))
        try:
            rows = c.execute(
                "SELECT source, property_id, price_pcm, postcode, size_sqft, "
                "floorplan_url, address_fingerprint, first_seen, last_seen, "
                "is_active FROM listings ORDER BY id"
            ).fetchall()
            ph = c.execute(
                "SELECT listing_id, price_pcm, recorded_at FROM price_history "
                "ORDER BY id"
            ).fetchall()
        finally:
            c.close()
        return rows, ph

    assert dump(a) == dump(b), "fixture builder is non-deterministic"


@pytest.mark.unit
def test_fixture_is_representative():
    """Real content: multiple sources, an inactive row, no-sqft backfill targets,
    a floorplan'd row, and a price-change history — so the structural suites have
    something meaningful to assert."""
    conn = sqlite3.connect(f"file:{COMMITTED_FIXTURE}?mode=ro", uri=True)
    try:
        sources = {r[0] for r in conn.execute("SELECT DISTINCT source FROM listings")}
        assert {"savills", "knightfrank", "foxtons", "rightmove", "chestertons"} <= sources, (
            f"fixture should span the real sources, got {sources}"
        )
        total = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM listings WHERE is_active=1").fetchone()[0]
        assert total >= 6 and active >= 1 and active < total, (
            "need an inactive row to exercise the active filter"
        )
        # backfill targets: active rightmove rows with no sqft
        rm_no_sqft = conn.execute(
            "SELECT COUNT(*) FROM listings WHERE source='rightmove' AND is_active=1 "
            "AND (size_sqft IS NULL OR size_sqft=0)"
        ).fetchone()[0]
        assert rm_no_sqft >= 1, "fixture needs an active no-sqft rightmove row (backfill target)"
        # at least one row WITH a floorplan_url and one of those is a no-sqft target
        fp_target = conn.execute(
            "SELECT COUNT(*) FROM listings WHERE source='rightmove' "
            "AND (size_sqft IS NULL OR size_sqft=0) "
            "AND floorplan_url IS NOT NULL AND floorplan_url != ''"
        ).fetchone()[0]
        assert fp_target >= 1, "fixture needs a no-sqft rightmove row that has a floorplan_url"
        # a price-change history row exists
        ph = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
        assert ph >= total, "every listing should have at least an initial price log"
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# The WIRING guard: conftest exposes whether we're running against the fixture,
# and when it is, the data-validation suites must NOT be all-skipped.
# --------------------------------------------------------------------------- #
def _load_conftest():
    """Import tests/conftest.py as a module by path (robust to cwd / targeted runs)."""
    import importlib.util

    cpath = ROOT / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("_tests_conftest_under_test", cpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_conftest_exposes_active_db_path():
    """conftest resolves an ACTIVE DB path (live OR the committed fixture) and exposes
    it, so the structural suites always have a DB to assert against in CI."""
    conftest = _load_conftest()

    assert hasattr(conftest, "_active_db_path"), (
        "conftest must expose _active_db_path() — the resolved DB (live or fixture)"
    )
    p = conftest._active_db_path()
    assert p is not None and Path(p).exists(), (
        f"conftest._active_db_path() must point at an existing DB, got {p!r}"
    )
    # The two fixture-backed structural suites must ALWAYS have a DB to run against —
    # the live DB OR the committed fixture. This is what un-skips them in PR CI.
    assert hasattr(conftest, "_structural_suites_have_db"), (
        "conftest must expose _structural_suites_have_db()"
    )
    assert conftest._structural_suites_have_db(), (
        "with the fixture present, the structural suites must have a DB so they EXECUTE "
        "instead of all-skipping"
    )


def test_scrape_validation_structural_tests_are_not_skipped(request):
    """In THIS session, the structural scrape-validation tests must be COLLECTED and
    NOT skip-marked. If conftest stopped wiring the fixture, they'd silently go dark —
    which is exactly the regression this guards.

    Enforced on a full-suite run only (CI); a targeted dev run that didn't collect
    test_scrape_validation is skipped to avoid a false alarm.
    """
    items = request.session.items
    sv = [it for it in items if "test_scrape_validation" in str(it.fspath)]
    if not sv:
        pytest.skip("test_scrape_validation not collected in this targeted run")

    def is_skipped(it):
        for m in it.iter_markers():
            if m.name == "skip":
                return True
            if m.name == "skipif" and any(bool(c) for c in m.args):
                return True
        return False

    # STRUCTURAL = the pre-scrape readiness + data-integrity asserts. These MUST run
    # against the fixture in CI. The POST-scrape class (TestPostScrapeValidation) is a
    # snapshot/freshness COMPARISON against the real scraped DB — correctly skipped on
    # the fixture (see conftest) — so it's excluded from "must-run structural".
    def is_post_scrape(it):
        return bool(it.cls) and it.cls.__name__ == "TestPostScrapeValidation"

    structural = [it for it in sv if not is_post_scrape(it)]
    skipped_structural = [it.nodeid for it in structural if is_skipped(it)]
    assert not skipped_structural, (
        "structural scrape-validation tests are SKIPPED in this run — the mini fixture "
        "wiring in conftest is not active. These must EXECUTE in PR CI.\n  "
        + "\n  ".join(skipped_structural)
    )
    # And there must be a healthy number of them actually running (not collapsed to ~0).
    assert len(structural) - len(skipped_structural) >= 10, (
        f"only {len(structural) - len(skipped_structural)} structural scrape-validation "
        "tests are executing — expected the full integrity/pre-scrape set"
    )
