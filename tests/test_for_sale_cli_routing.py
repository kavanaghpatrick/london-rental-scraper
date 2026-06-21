"""test_for_sale_cli_routing.py — TDD RED contract for the Inc2 CLI + PIPELINE ROUTING
that keeps the FOR-SALE vertical physically isolated from the rental stack.

WHAT THIS PINS (Inc2 §1 / §5 FILE B + AMENDMENT 4)
--------------------------------------------------
A spider in sale mode yields a plain dict with `listing_type == "sale"` into the SAME
ITEM_PIPELINES chain (CleanData→Dup→Json→SQLite). `SQLitePipeline` UNCONDITIONALLY writes
the rental `listings` table in output/rentals.db today, so without interception a sale row
would land in the rental table. Inc2 closes this with:

  * a `--listing-type {rent,sale}` Typer option on `scrape` (default rent; unknowns coerced
    to rent), threaded into run_spider, which appends `-a listing_type=sale` ONLY for sale
    (rental scrapy command byte-identical to today);
  * a NEW `SaleListingPipeline` (priority 500) that writes output/sales.db / sale_listings
    and IGNORES rental items;
  * a per-item `listing_type == "sale"` discriminator guard at the TOP of every rental
    pipeline's process_item, so sale items NEVER touch the rental table/jsonl;
  * a `get_sale_db_path()` read seam separate from the unchanged `get_db_path()`;
  * a guard rejecting `--postgres` with `--listing-type sale` (SQLite-only in Inc2).

Written FIRST and MUST FAIL (RED) until Group A wires the option/seams/pipeline: the only
legitimate failure cause is the missing production symbols (the `--listing-type` option,
`get_sale_db_path`, `SaleListingPipeline`, the discriminator guards), not a test defect.

ZERO RENTAL REGRESSION: the default-invocation + rental-write tests assert the rental path
is unchanged. CI-SAFE: marker `for_sale`; no network/crawl; in-memory + tmp SQLite only.
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import main as cli_main
from for_sale import sale_data

pytestmark = pytest.mark.for_sale

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "for_sale"
runner = CliRunner()


def _plain(s):
    """Strip ANSI escapes + collapse all whitespace so substring checks survive
    Rich's terminal-width/version-dependent wrapping (a CI-vs-local rendering
    flake, not a behavioural contract)."""
    return re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", s or ""))


# ── Test doubles ──────────────────────────────────────────────────────────────────

class _FakeSettings:
    """Mimics spider.settings.get('OUTPUT_DIR', default)."""

    def __init__(self, output_dir):
        self._d = {"OUTPUT_DIR": str(output_dir)}

    def get(self, key, default=None):
        return self._d.get(key, default)


class _FakeSpider:
    def __init__(self, output_dir):
        self.settings = _FakeSettings(output_dir)


class _FakePopen:
    """Captures the cmd a run_spider build produced without crawling."""

    last_cmd = None

    def __init__(self, cmd, *args, **kwargs):
        type(self).last_cmd = list(cmd)

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


def _sale_item(**overrides):
    base = {
        "source": "rightmove", "property_id": "SALE1", "listing_type": "sale",
        "url": "https://www.rightmove.co.uk/properties/SALE1",
        "asking_price": 1_950_000, "postcode": "SW3", "bedrooms": 2,
        "address": "1 Sale Street, Chelsea, London, SW3", "area": "Chelsea",
    }
    base.update(overrides)
    return base


def _rental_item(**overrides):
    base = {
        "source": "rightmove", "property_id": "RENT1",
        "url": "https://www.rightmove.co.uk/properties/RENT1",
        "price_pcm": 3500, "postcode": "SW3", "bedrooms": 2,
        "address": "2 Rent Street, Chelsea, London, SW3", "area": "Chelsea",
    }
    base.update(overrides)
    return base


def _count(db_path, table):
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        if cur.fetchone() is None:
            return None  # table does not exist
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


# ── 1. The --listing-type option ──────────────────────────────────────────────────

def test_scrape_has_listing_type_option_default_rent():
    """`scrape` exposes a --listing-type option defaulting to 'rent'.

    Introspects the registered Typer OptionInfo directly (stdlib `inspect` only)
    instead of grepping the Rich-rendered --help text: that help table wraps/
    ANSI-formats differently across Rich versions and terminal widths (an 80-col
    CI runner split the option name out of the captured string), which is a
    rendering flake, not a behavioural contract. Deliberately avoids `import click`
    too — the CI python-tests env does not expose click as a top-level module even
    though Typer is installed. The OptionInfo's param_decls + default ARE the contract.
    """
    import inspect

    scrape_cb = next(
        (
            info.callback
            for info in cli_main.app.registered_commands
            if getattr(info, "callback", None) is not None
            and "listing_type" in inspect.signature(info.callback).parameters
        ),
        None,
    )
    assert scrape_cb is not None, "no command exposes a listing_type parameter"
    opt = inspect.signature(scrape_cb).parameters["listing_type"].default
    assert "--listing-type" in (getattr(opt, "param_decls", None) or ()), (
        f"listing_type must be a --listing-type Option, got {opt!r}"
    )
    assert getattr(opt, "default", None) == "rent", (
        f"--listing-type default must be 'rent', got {getattr(opt, 'default', None)!r}"
    )


def test_listing_type_coerces_unknown_to_rent(monkeypatch):
    """Only an exact (case-insensitive) 'sale' selects sale; anything else -> rent.
    Observed through the value threaded into run_spider."""
    captured = {}

    def fake_run_spider(spider_name, **kwargs):
        captured["listing_type"] = kwargs.get("listing_type")
        return True, "ok", False, {"item_scraped_count": 1}

    monkeypatch.setattr(cli_main, "run_spider", fake_run_spider)

    def threaded(value):
        captured.clear()
        runner.invoke(
            cli_main.app,
            ["scrape", "--source", "rightmove", "--listing-type", value, "--dry-run"],
        )
        return captured.get("listing_type")

    assert threaded("sale") == "sale"
    assert threaded("SALE") == "sale"
    assert threaded("Sale ") == "rent"   # trailing space is not an exact match
    assert threaded("SOLD") == "rent"
    assert threaded("x") == "rent"


def test_sale_flag_passes_dash_a_listing_type_to_run_spider(monkeypatch):
    captured = {}

    def fake_run_spider(spider_name, **kwargs):
        captured.update(kwargs)
        return True, "ok", False, {"item_scraped_count": 1}

    monkeypatch.setattr(cli_main, "run_spider", fake_run_spider)
    result = runner.invoke(
        cli_main.app,
        ["scrape", "--source", "rightmove", "--listing-type", "sale", "--dry-run"],
    )
    assert result.exit_code == 0, result.stdout
    assert captured.get("listing_type") == "sale"


def test_default_invocation_keeps_rent_and_no_dash_a(monkeypatch):
    """ZERO-REGRESSION: no --listing-type -> run_spider gets listing_type == 'rent'."""
    captured = {}

    def fake_run_spider(spider_name, **kwargs):
        captured.update(kwargs)
        return True, "ok", False, {"item_scraped_count": 1}

    monkeypatch.setattr(cli_main, "run_spider", fake_run_spider)
    result = runner.invoke(cli_main.app, ["scrape", "--source", "rightmove", "--dry-run"])
    assert result.exit_code == 0, result.stdout
    assert captured.get("listing_type") == "rent"


def test_run_spider_appends_listing_type_sale_to_cmd(monkeypatch):
    """run_spider appends `-a listing_type=sale` ONLY in sale mode; the rental cmd is
    byte-identical to today (no such -a)."""
    monkeypatch.setattr(cli_main.subprocess, "Popen", _FakePopen)

    cli_main.run_spider("rightmove", listing_type="sale", dry_run=True)
    cmd = _FakePopen.last_cmd
    assert "-a" in cmd and "listing_type=sale" in cmd
    idx = cmd.index("listing_type=sale")
    assert cmd[idx - 1] == "-a", "listing_type=sale must follow a -a flag"

    _FakePopen.last_cmd = None
    cli_main.run_spider("rightmove", listing_type="rent", dry_run=True)
    assert "listing_type=sale" not in _FakePopen.last_cmd


# ── 2. The sales.db read/write seams ──────────────────────────────────────────────

def test_get_sale_db_path_is_separate_file():
    sale_path = cli_main.get_sale_db_path()
    rent_path = cli_main.get_db_path()
    assert sale_path == cli_main.PROJECT_ROOT / "output" / "sales.db"
    assert rent_path == cli_main.PROJECT_ROOT / "output" / "rentals.db"
    assert sale_path != rent_path


def test_sale_pipeline_writes_sales_db_not_rentals_db(tmp_path):
    from property_scraper.pipelines import SaleListingPipeline

    spider = _FakeSpider(tmp_path)
    pipe = SaleListingPipeline()
    pipe.open_spider(spider)
    pipe.process_item(_sale_item(), spider)
    pipe.close_spider(spider)

    sales_db = tmp_path / "sales.db"
    rentals_db = tmp_path / "rentals.db"
    assert sales_db.exists(), "SaleListingPipeline must write output/sales.db"
    assert not rentals_db.exists(), "SaleListingPipeline must NOT create rentals.db"
    assert _count(sales_db, "sale_listings") == 1
    assert _count(sales_db, "listings") is None, "no rental listings table in sales.db"


def test_sale_pipeline_ignores_rental_items(tmp_path):
    from property_scraper.pipelines import SaleListingPipeline

    spider = _FakeSpider(tmp_path)
    pipe = SaleListingPipeline()
    pipe.open_spider(spider)
    returned = pipe.process_item(_rental_item(), spider)  # listing_type absent (rental)
    pipe.close_spider(spider)

    assert dict(returned).get("property_id") == "RENT1", "rental item passes through"
    sales_db = tmp_path / "sales.db"
    assert _count(sales_db, "sale_listings") == 0, "rental item must not land in sale_listings"


# ── 3. The rental-pipeline discriminator guard (CORE NO-LEAK) ─────────────────────

def test_rental_sqlite_pipeline_skips_sale_items(tmp_path):
    """CORE NO-LEAK GUARD: a sale item (listing_type=='sale') passed to the rental
    SQLitePipeline is returned untouched and NEVER written to the rental listings table."""
    from property_scraper.pipelines import SQLitePipeline

    spider = _FakeSpider(tmp_path)
    pipe = SQLitePipeline()
    pipe.open_spider(spider)
    returned = pipe.process_item(_sale_item(), spider)
    pipe.close_spider(spider)

    assert dict(returned).get("property_id") == "SALE1", "sale item passes through unchanged"
    rentals_db = tmp_path / "rentals.db"
    assert _count(rentals_db, "listings") == 0, "sale item leaked into rental listings table"


def test_rental_sqlite_pipeline_still_writes_rental_items(tmp_path):
    """ZERO-REGRESSION: a normal rental item is STILL written by the rental pipeline."""
    from property_scraper.pipelines import SQLitePipeline

    spider = _FakeSpider(tmp_path)
    pipe = SQLitePipeline()
    pipe.open_spider(spider)
    pipe.process_item(_rental_item(), spider)
    pipe.close_spider(spider)

    rentals_db = tmp_path / "rentals.db"
    assert _count(rentals_db, "listings") == 1, "rental item must still be persisted"


# ── 4. Guards + invariants ────────────────────────────────────────────────────────

def test_postgres_plus_sale_is_rejected():
    result = runner.invoke(
        cli_main.app,
        ["scrape", "--source", "rightmove", "--listing-type", "sale", "--postgres"],
    )
    assert result.exit_code != 0
    # exit_code != 0 is the binding contract; message check is ANSI/wrap-normalized
    # ("SQLite" is one token so it can't be split across a Rich line wrap).
    assert "SQLite" in _plain(result.output)


def test_registry_unchanged_by_inc2():
    from cli import registry
    assert set(registry.SPIDERS.keys()) == {
        "savills", "knightfrank", "chestertons", "foxtons", "rightmove"
    }
    for cfg in registry.SPIDERS.values():
        assert not hasattr(cfg, "listing_type"), "registry must carry no listing_type"


# ── 5. AMENDMENT 4 — end-to-end Rightmove sale dict routes to sales.db only ────────

def test_rightmove_sale_dict_routes_to_sales_db_end_to_end(tmp_path):
    """AMENDMENT 4: a Rightmove sale dict flowing through the FULL committed chain
    (CleanData -> SQLite[rental] -> SaleListingPipeline) lands ONLY in sales.db /
    sale_listings; the rental listings table count stays 0. Proves the 4 discriminator
    guards + SaleListingPipeline close the already-shipped Rightmove sale-mode regression."""
    from for_sale.listing_parse import parse_rightmove_for_sale
    from property_scraper.pipelines import CleanDataPipeline, SQLitePipeline, SaleListingPipeline

    props = json.loads(
        (FIXTURES / "rightmove_for_sale_properties.json").read_text()
    )
    sale_item = dict(parse_rightmove_for_sale(props[0], "Chelsea"))  # spider-boundary dict
    assert sale_item["listing_type"] == "sale"

    spider = _FakeSpider(tmp_path)
    clean = CleanDataPipeline()
    rental = SQLitePipeline()
    sale = SaleListingPipeline()
    for p in (clean, rental, sale):
        p.open_spider(spider)
    try:
        item = sale_item
        for p in (clean, rental, sale):
            item = p.process_item(item, spider)
    finally:
        for p in (clean, rental, sale):
            p.close_spider(spider)

    sales_db = tmp_path / "sales.db"
    rentals_db = tmp_path / "rentals.db"
    assert _count(sales_db, "sale_listings") == 1, "sale row must land in sale_listings"
    # rentals.db may be opened by the rental pipeline, but the sale row must NOT be there.
    assert _count(rentals_db, "listings") == 0, "sale dict leaked into rental listings table"
