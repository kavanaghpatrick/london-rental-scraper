#!/usr/bin/env python3
"""OPT-IN live smoke tests — hit real sites, 1-2 pages, ~20 items.

These are EXCLUDED from the default `pytest` run (pytest.ini: -m "not live").
Run explicitly with:   pytest -m live tests/test_live_smoke.py -v

Purpose: catch site-level breakage (markup drift, blocks, 403/429) that unit
tests on saved fixtures cannot. They run a real bounded crawl into a SCRATCH
output dir (never the canonical output/rentals.db) and assert items land with
populated fields.

Each spider crawls into a tmp OUTPUT_DIR, so the SQLitePipeline writes
<scratch>/rentals.db, never the canonical DB. CLOSESPIDER_ITEMCOUNT bounds it.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent

pytestmark = pytest.mark.live  # whole module is opt-in


def _run_crawl(spider, settings_module, extra_args, scratch_dir, itemcount=20,
               timeout_s=420):
    """Run a bounded crawl into a scratch OUTPUT_DIR; return path to scratch rentals.db."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "scrapy", "crawl", spider,
        *extra_args,
        "-s", f"OUTPUT_DIR={scratch_dir}",
        "-s", f"CLOSESPIDER_ITEMCOUNT={itemcount}",
        "-s", "LOG_LEVEL=WARNING",
    ]
    env = {
        "PYTHONPATH": str(PROJECT_ROOT),
        "SCRAPY_SETTINGS_MODULE": settings_module,
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), env=env,
        capture_output=True, text=True, timeout=timeout_s,
    )
    return scratch_dir / "rentals.db", proc


def _assert_landed(db_path, source, min_items=5):
    assert db_path.exists(), f"{source}: scratch DB not created"
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT price_pcm, bedrooms, postcode, size_sqft, url FROM listings "
        "WHERE source=?", (source,),
    ).fetchall()
    conn.close()
    assert len(rows) >= min_items, f"{source}: only {len(rows)} items landed (<{min_items})"
    # every row has a price and a url
    assert all(r[0] and r[0] > 0 for r in rows), f"{source}: rows missing price_pcm"
    assert all(r[4] and r[4].startswith("http") for r in rows), f"{source}: rows missing url"
    return rows


@pytest.mark.smoke
def test_rightmove_live_smoke(tmp_path):
    db, proc = _run_crawl(
        "rightmove", "property_scraper.settings_standard",
        ["-a", "areas=Chelsea", "-a", "max_pages=1", "-a", "fetch_details=false"],
        tmp_path / "rm_smoke",
    )
    rows = _assert_landed(db, "rightmove", min_items=10)
    # at least some postcodes present (search results usually carry them)
    assert any(r[2] for r in rows), "rightmove: no postcodes extracted"


@pytest.mark.smoke
def test_savills_live_smoke(tmp_path):
    db, proc = _run_crawl(
        "savills", "property_scraper.settings",
        ["-a", "max_pages=4", "-a", "fetch_floorplans=false"],
        tmp_path / "sv_smoke",
    )
    rows = _assert_landed(db, "savills", min_items=5)
    # savills guarantees sqft on every yielded row
    assert all(r[3] and r[3] > 0 for r in rows), "savills: rows missing sqft"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "live"])
