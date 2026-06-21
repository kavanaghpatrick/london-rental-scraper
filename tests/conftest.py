"""
Shared pytest fixtures + collection rules for the CI suite (#49).

Goal: a default `pytest` run (what PR/push CI runs) executes UNIT + self-contained
tests and is GREEN with NO populated database — because CI has no rentals.db (it's
gitignored). Tests that read the live canonical DB are DATA-VALIDATION tests; they
are auto-skipped when the DB is absent (so CI is green) and run for real in the
daily-scrape post-scrape job where a populated DB exists.

Marker convention (see pytest.ini):
  - (default)  unit / self-contained — always runs, gates PR.
  - data       reads the live rentals.db — skipped if the DB is missing.
  - live       hits the network — opt-in only (`-m live`).
"""
import os
from pathlib import Path

import pandas as pd
import pytest

# Make local pytest match CI's pandas behaviour. Newer pandas (CI) does NOT silently
# downcast object-dtype columns on .fillna(); pandas 2.3.x locally DOES (emitting a
# FutureWarning), which masked the unseen/coordless-row np.log1p(object) crash
# (rental_price_models_v20 distance block). Turning the future behaviour on here means
# a coordless single-row inference reproduces the CI TypeError locally instead of
# passing silently. See tests/test_model_inference.py::test_coordless_row_* and
# the np-log1p-distance-dtype-bug finding (task #6).
pd.set_option('future.no_silent_downcasting', True)

# The canonical (live) DB path the data-validation tests read when it exists. This is
# the REAL 20MB+ rentals.db produced by daily-scrape; it's gitignored, so in PR CI it
# is ABSENT.
_LIVE_DB_PATH = Path(
    os.environ.get(
        "SCRAPER_DB_PATH",
        Path(__file__).resolve().parent.parent / "output" / "rentals.db",
    )
)

# Committed mini fixture (tests/fixtures/mini_rentals.db) — a SMALL, deterministic,
# schema-faithful SQLite DB. When the live DB is absent (PR CI), we build/serve THIS
# so the STRUCTURAL data-validation asserts EXECUTE pre-merge instead of all-skipping.
# Freshness/recency asserts stay gated separately (_scrape_ran_today below): the
# fixture's dates are in the past, so they correctly still skip — only structural/logic
# checks run against it.
_FIXTURE_DB_PATH = Path(__file__).resolve().parent / "fixtures" / "mini_rentals.db"


def _path_ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


# A POST-backfill variant of the fixture, built on demand (NOT committed — it's
# derived). Pairing baseline (committed, = pre) + backfilled (= live/post) lets the
# backfill RECOVERED-ROW sanity tests run for real in CI: one row goes no-sqft -> sqft
# between the two, exactly like a real OCR backfill. The "_"-prefixed name keeps it out
# of the way; it's gitignored as a derived artifact.
_FIXTURE_BACKFILLED_DB_PATH = Path(__file__).resolve().parent / "fixtures" / "_mini_rentals_backfilled.db"


def _force_no_live_db() -> bool:
    """CI-simulation switch: pretend the LIVE DB is absent even on a dev box that has
    output/rentals.db, so the fixture/no-live path is exercised exactly as PR CI."""
    return os.environ.get("CONFTEST_FORCE_NO_LIVE_DB", "").strip().lower() in {
        "1", "true", "yes"
    }


def _ensure_fixture_db() -> Path:
    """Build the committed mini fixture if it isn't already on disk, return its path.

    The fixture is committed, so this normally just returns the existing file; the
    on-demand build is a belt-and-braces fallback (e.g. a fresh checkout that somehow
    lacks the artifact, or a dev who deleted it).
    """
    if not _path_ok(_FIXTURE_DB_PATH):
        # Import lazily so a missing builder never breaks collection when the live DB
        # is present and the fixture isn't needed. The COMMITTED fixture is the
        # canonical "baseline" (pre-backfill) state.
        from tests.fixtures.build_mini_rentals import build_mini_rentals
        build_mini_rentals(_FIXTURE_DB_PATH, variant="baseline")
    return _FIXTURE_DB_PATH


def _ensure_backfilled_fixture_db() -> Path:
    """Build (derived/uncommitted) the POST-backfill variant: identical to the baseline
    except one rightmove row recovered a sqft. Serves as the backfill LIVE (post) DB so
    the recovered-row sanity tests run for real."""
    from tests.fixtures.build_mini_rentals import build_mini_rentals
    build_mini_rentals(_FIXTURE_BACKFILLED_DB_PATH, variant="backfilled")
    return _FIXTURE_BACKFILLED_DB_PATH


def _active_db_path() -> Path:
    """The DB the data-validation suites should run against THIS session.

    Live DB if present (daily-scrape post-scrape) — otherwise the committed mini
    fixture (PR CI), so the structural suites are never all-skipped just because the
    20MB live DB is gitignored.

    CI-SIMULATION: set CONFTEST_FORCE_NO_LIVE_DB=1 to force the live-DB-absent branch
    even on a dev box that HAS output/rentals.db, so the fixture path can be exercised
    exactly as PR CI would (without deleting the real DB or fiddling SCRAPER_DB_PATH,
    which the test modules also read).
    """
    if not _force_no_live_db() and _path_ok(_LIVE_DB_PATH):
        return _LIVE_DB_PATH
    return _ensure_fixture_db()


# Resolve the active DB ONCE, at conftest import time. This is EITHER the live DB
# (daily-scrape) OR the committed mini fixture (PR CI fallback).
_ACTIVE_DB_PATH = _active_db_path()

# IMPORTANT — SCOPED fixture wiring (do NOT overload SCRAPER_DB_PATH globally):
# SCRAPER_DB_PATH is read by SEVERAL `data`-marked modules (e.g. test_dedupe_postgres),
# many of which need the FULL real DB's scale and are meaningless against a tiny
# fixture. So when we fall back to the fixture we must NOT point SCRAPER_DB_PATH at it
# globally — that would un-skip those unrelated data tests and break them. Instead the
# fixture is wired ONLY into the two STRUCTURAL suites it's built for, by patching THEIR
# module-level path constants after import (see _wire_fixture_into_modules in
# pytest_collection_modifyitems). Every OTHER `data`-marked test keeps skipping when the
# live DB is absent, exactly as before.
#
# The MODULES whose data-validation asserts the mini fixture is designed to satisfy:
_FIXTURE_BACKED_MODULES = ("test_scrape_validation", "test_backfill_acceptance")


def _on_fixture() -> bool:
    """True when the active DB is the committed mini fixture (PR CI), not the live DB."""
    return _ACTIVE_DB_PATH == _FIXTURE_DB_PATH


# Back-compat alias: some call sites referenced DB_PATH directly. In fixture mode this
# is the committed baseline fixture; in live mode it's the real DB.
DB_PATH = _ACTIVE_DB_PATH


def _db_available() -> bool:
    """Whether the LIVE canonical DB is present (daily-scrape). This is what the GENERIC
    `data`-marker no-DB skip keys on, so unrelated data tests still skip in PR CI.

    NOTE: this is intentionally LIVE-only (honouring the CI-sim force switch). The mini
    fixture un-skips ONLY the two fixture-backed structural suites (handled separately),
    not the whole `data` set.
    """
    if _force_no_live_db():
        return False
    return _path_ok(_LIVE_DB_PATH)


def _structural_suites_have_db() -> bool:
    """Whether the two fixture-backed structural suites have a DB to run against —
    the live DB OR the committed fixture. Always True once the fixture exists."""
    return _path_ok(_ACTIVE_DB_PATH)


def _wire_fixture_into_modules():
    """Point the two fixture-backed modules' DB-path constants at the fixture variants,
    so their STRUCTURAL asserts run against the fixture in PR CI — WITHOUT touching the
    shared SCRAPER_DB_PATH env (which other data tests read). Live mode leaves the real
    DB in place (no-op). Idempotent."""
    if not _on_fixture():
        return
    import sys

    backfilled = _ensure_backfilled_fixture_db()

    def _patch(basename, attrs):
        # pytest collects these as 'tests.test_scrape_validation' (package-qualified,
        # because tests/__init__.py exists), NOT the bare name — so look up the ALREADY
        # IMPORTED module in sys.modules by exact OR suffix match and patch THAT object.
        # Fail LOUDLY in fixture mode if it isn't found (a silent except here previously
        # let the fixture path never reach the tests, so they ran against the real DB).
        mod = next(
            (m for n, m in list(sys.modules.items())
             if m is not None and (n == basename or n.endswith("." + basename))),
            None,
        )
        if mod is None:
            raise RuntimeError(
                f"fixture wiring failed: collected test module for '{basename}' not in "
                f"sys.modules — cannot repoint its DB path. (Check the package-qualified "
                f"collection name / tests/__init__.py.)"
            )
        for k, v in attrs.items():
            setattr(mod, k, v)

    # test_scrape_validation reads its own module-level DB_PATH; repoint it.
    _patch("test_scrape_validation", {"DB_PATH": backfilled})
    # test_backfill_acceptance: LIVE = backfilled (post), BASELINE = committed (pre).
    _patch("test_backfill_acceptance", {"LIVE_DB": backfilled, "BASELINE_DB": _FIXTURE_DB_PATH})


# POST-scrape tests that are only valid INSIDE a fresh snapshot -> scrape -> validate
# cycle: they assert a scrape ran "today" (last_seen) and that sqft coverage didn't
# regress vs the snapshot. Run bare against a static DB they fail for environmental
# reasons (no scrape ran today) — that's noise, not a regression. We gate them on
# "did a scrape actually run today" instead of xfail-ing: xfail(strict=False) would
# mark them expected-to-fail even in daily-scrape where they MUST run and pass,
# masking a genuine post-scrape regression.
_POST_SCRAPE_FRESHNESS_TESTS = {
    "test_post_scrape_last_seen_updated",
    "test_post_scrape_sqft_coverage_maintained",
}


def _scrape_ran_today() -> bool:
    """True only when the DB shows listings updated TODAY (a scrape ran today).

    That is the real precondition for the post-scrape freshness asserts — the
    signature of an actual snapshot -> scrape -> validate cycle (daily-scrape).
    Keying on the DB's last_seen (not the snapshot file's timestamp) is robust:
    refreshing scrape_snapshot.json locally without scraping does NOT spuriously
    un-skip these tests, because no scrape means updated_today == 0.
    """
    import datetime
    import sqlite3

    if not _db_available():
        return False
    today = datetime.date.today().isoformat()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            (n,) = conn.execute(
                "SELECT COUNT(*) FROM listings WHERE last_seen LIKE ?", (today + "%",)
            ).fetchone()
        finally:
            conn.close()
        return n > 0
    except sqlite3.Error:
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip DATA-VALIDATION tests when there's no populated DB.

    A test is treated as data-validation if it's marked `data` OR lives in
    test_scrape_validation.py (which reads the live DB). This keeps the default CI
    run green without a DB while still running these for real where a DB exists.

    Additionally, the POST-scrape FRESHNESS asserts (see _POST_SCRAPE_FRESHNESS_TESTS)
    are skipped unless a scrape actually ran today (DB updated_today > 0) — i.e. we are
    not inside a fresh snapshot -> scrape -> validate cycle — so a local/bare run against
    a static DB is clean while daily-scrape (scrape ran today) still runs them for real.
    """
    # --- API integration tests (#48) need ephemeral Postgres + `next dev`.
    # serving's test file now skipif-guards on _dashboard_ready() (node + next +
    # postgres/pg_ctl), so it skips cleanly on its own. This is a BELT-AND-BRACES
    # second guard on the exact binary that errored: testing.postgresql.Postgresql()
    # invokes `initdb`, and serving's check looks for `postgres`/`pg_ctl` not `initdb`
    # specifically — so on the narrow edge where postgres exists but initdb doesn't,
    # this still prevents a setup ERROR. Harmless overlap; keeps PR CI green either way.
    import shutil
    if shutil.which("initdb") is None:
        skip_no_pg = pytest.mark.skip(reason="no `initdb` (Postgres) — API integration test; runs where Postgres is installed")
        for item in items:
            if "test_api_dashboard" in str(item.fspath) and "test_dashboard_typechecks" not in item.name:
                # Keep the pure tsc typecheck unit test; skip the server/Postgres ones.
                item.add_marker(skip_no_pg)

    def _in_fixture_module(item) -> bool:
        return any(m in str(item.fspath) for m in _FIXTURE_BACKED_MODULES)

    # FIXTURE MODE (PR CI): no LIVE DB, so the two structural suites run against the
    # committed mini fixture. Wire it into THOSE modules only (never the shared env), so
    # other `data` tests are unaffected.
    if _on_fixture():
        _wire_fixture_into_modules()

        # The POST-scrape tests are snapshot/freshness COMPARISONS — they diff against
        # tests/scrape_snapshot.json (captured from the real 20MB DB) and assert a scrape
        # ran today. Meaningless against the tiny fixture, so skip the whole
        # TestPostScrapeValidation class here; the structural/integrity asserts still run.
        skip_fixture_postscrape = pytest.mark.skip(
            reason="running against the mini fixture (PR CI) — post-scrape "
            "snapshot/freshness comparisons only run against a real scraped DB "
            "(daily-scrape); structural/integrity asserts run here"
        )
        for item in items:
            if (
                "test_scrape_validation" in str(item.fspath)
                and "TestPostScrapeValidation" in (item.cls.__name__ if item.cls else "")
            ):
                item.add_marker(skip_fixture_postscrape)

    # Freshness gate for the post-scrape asserts: when a REAL DB is PRESENT but no scrape
    # ran today (a static/frozen DB), skip them — they only mean something inside a
    # fresh snapshot -> scrape -> validate cycle (daily-scrape). (On the fixture the whole
    # post-scrape class is already skipped above, so this only bites a real DB.)
    if _db_available() and not _scrape_ran_today():
        skip_stale = pytest.mark.skip(
            reason="no scrape ran today (DB updated_today == 0) — post-scrape freshness "
            "asserts only run inside a fresh snapshot->scrape->validate cycle (daily-scrape)"
        )
        for item in items:
            if (
                "test_scrape_validation" in str(item.fspath)
                and item.name in _POST_SCRAPE_FRESHNESS_TESTS
            ):
                item.add_marker(skip_stale)

    if _db_available():
        return  # LIVE DB present (daily-scrape) — run EVERYTHING for real.

    # No LIVE DB (PR CI). Skip the GENERIC `data`-marked tests that need the full real DB
    # — EXCEPT the two fixture-backed structural suites, which the mini fixture lets run.
    skip_no_db = pytest.mark.skip(reason="no populated rentals.db (data-validation test; runs in daily-scrape)")
    for item in items:
        if _in_fixture_module(item):
            continue  # the fixture covers these — let them execute against it
        is_data = "data" in item.keywords or "test_scrape_validation" in str(item.fspath)
        if is_data:
            item.add_marker(skip_no_db)


# ---------------------------------------------------------------------------------
# ANTI-SILENT-SKIP instrumentation (#49 systemic fix).
#
# The team's distrust: a critical regression test that silently STOPS RUNNING in CI —
# renamed, de-collected, or over-broadly skipped — looks identical to "all green".
# tests/test_ci_critical_tests_run.py inspects the live session's collected items to
# assert each test on the CRITICAL allowlist was collected AND not skip-marked in this
# env. We expose the post-skip collected nodeids on the session config so the meta-test
# (and any future tooling) has a stable handle on "what actually ran this run".
# ---------------------------------------------------------------------------------
def pytest_collection_finish(session):
    # Record every nodeid that survived collection (after the skips above are applied).
    session.config._collected_nodeids = {item.nodeid for item in session.items}
