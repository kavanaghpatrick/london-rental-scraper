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

# The canonical DB path the data-validation tests read (matches
# tests/test_scrape_validation.py's DB_PATH resolution).
DB_PATH = Path(
    os.environ.get(
        "SCRAPER_DB_PATH",
        Path(__file__).resolve().parent.parent / "output" / "rentals.db",
    )
)


def _db_available() -> bool:
    return DB_PATH.exists() and DB_PATH.stat().st_size > 0


def pytest_collection_modifyitems(config, items):
    """Auto-skip DATA-VALIDATION tests when there's no populated DB.

    A test is treated as data-validation if it's marked `data` OR lives in
    test_scrape_validation.py (which reads the live DB). This keeps the default CI
    run green without a DB while still running these for real where a DB exists.
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

    if _db_available():
        return  # DB present (e.g. daily-scrape post-scrape) — run everything.

    skip_no_db = pytest.mark.skip(reason="no populated rentals.db (data-validation test; runs in daily-scrape)")
    for item in items:
        is_data = "data" in item.keywords or "test_scrape_validation" in str(item.fspath)
        if is_data:
            item.add_marker(skip_no_db)
