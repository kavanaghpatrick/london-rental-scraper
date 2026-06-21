"""test_for_sale_playwright_spider_mode.py — TDD RED contract for the SALE-MODE WIRING
of the three Playwright spiders (savills / knightfrank / chestertons) in Inc2.

WHAT THIS PINS (Inc2 §3.4 / §5 FILE C, AMENDMENT 3)
---------------------------------------------------
Each Playwright spider must gain, WITHOUT changing its default rental behaviour:
  * a `listing_type` kwarg (default "rent"; any typo/unknown coerced to "rent");
  * a start-URL SECTION seam (`start_url_for(area)`) that swaps the rental section token
    for the for-sale one — Savills `property-to-rent`->`property-for-sale`, KnightFrank
    `to-let`->`for-sale`, Chestertons `/properties/lettings`->`/properties/sales`;
  * a delegating method `parse_card_data_for_sale(card_data, area)` that, in sale mode,
    calls the PURE `parse_<site>_for_sale` seam from for_sale.listing_parse and applies the
    spider-boundary 0->None asking_price normalisation (mirroring rightmove_spider's
    parse_for_sale_property), returning a plain dict with listing_type=="sale".

Per AMENDMENT 3 every test is suffixed with the site name so `-k` filters and grep stay
unambiguous against the existing Inc1 rental data-layer tests.

Written FIRST and MUST FAIL (RED) until Group B wires the spiders: the only legitimate
failure cause is the missing production seams (`start_url_for` on the Playwright spiders,
the `listing_type` coercion, `parse_card_data_for_sale`), not a test defect. The fixtures
are network-free committed card-dicts.

ZERO RENTAL REGRESSION: the default-mode tests assert the rental section is unchanged.
CI-SAFE: marker `for_sale`; no crawl/network; committed fixtures only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.for_sale

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "for_sale"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


# ── SAVILLS sale-mode wiring ──────────────────────────────────────────────────────

def _savills():
    from property_scraper.spiders.savills_spider import SavillsSpider
    return SavillsSpider


def test_default_savills_spider_is_rental_unchanged():
    spider = _savills()()
    assert getattr(spider, "listing_type", "rent") == "rent"
    assert "property-to-rent" in spider.start_url_for("Chelsea")
    assert "property-for-sale" not in spider.start_url_for("Chelsea")


def test_sale_mode_targets_for_sale_section_savills():
    spider = _savills()(listing_type="sale")
    assert spider.listing_type == "sale"
    url = spider.start_url_for("Chelsea")
    assert "property-for-sale" in url
    assert "property-to-rent" not in url


def test_typo_listing_type_falls_back_to_rent_savills():
    for bad in ("Sale!", "buy"):
        spider = _savills()(listing_type=bad)
        assert spider.listing_type == "rent"
        assert "property-to-rent" in spider.start_url_for("Chelsea")


def test_sale_mode_delegates_to_pure_seam_savills():
    spider = _savills()(listing_type="sale")
    card = _load("savills_card_data_for_sale.json")[0]
    it = spider.parse_card_data_for_sale(card, "Chelsea")
    assert isinstance(it, dict)
    assert it["listing_type"] == "sale"
    assert it["asking_price"] and it["asking_price"] >= 100_000
    assert "price_pcm" not in it and "price_pw" not in it
    # A POA / amount-0 card normalises to asking_price None at the spider boundary.
    poa = {**card, "price": 0, "text": "Some Street, Chelsea, London SW3\nFlat"}
    it_poa = spider.parse_card_data_for_sale(poa, "Chelsea")
    assert it_poa["asking_price"] is None


# ── KNIGHT FRANK sale-mode wiring ─────────────────────────────────────────────────

def _kf():
    from property_scraper.spiders.knightfrank_spider import KnightFrankSpider
    return KnightFrankSpider


def test_default_knightfrank_spider_is_rental_unchanged():
    spider = _kf()()
    assert getattr(spider, "listing_type", "rent") == "rent"
    assert "to-let" in spider.start_url_for("Chelsea")
    assert "for-sale" not in spider.start_url_for("Chelsea")


def test_sale_mode_targets_for_sale_section_knightfrank():
    spider = _kf()(listing_type="sale")
    assert spider.listing_type == "sale"
    url = spider.start_url_for("Chelsea")
    assert "for-sale" in url
    assert "to-let" not in url


def test_typo_listing_type_falls_back_to_rent_knightfrank():
    for bad in ("Sale!", "buy"):
        spider = _kf()(listing_type=bad)
        assert spider.listing_type == "rent"
        assert "to-let" in spider.start_url_for("Chelsea")


def test_sale_mode_delegates_to_pure_seam_knightfrank():
    spider = _kf()(listing_type="sale")
    card = _load("knightfrank_card_data_for_sale.json")[0]
    it = spider.parse_card_data_for_sale(card, "Chelsea")
    assert isinstance(it, dict)
    assert it["listing_type"] == "sale"
    assert it["asking_price"] and it["asking_price"] >= 100_000
    assert "price_pcm" not in it and "price_pw" not in it
    # A card whose text carries no £ (POA) normalises asking_price to None.
    poa = {**card, "text": "For Sale\nSome Mansions, Chelsea, London SW3\nFlat\n2\n1"}
    it_poa = spider.parse_card_data_for_sale(poa, "Chelsea")
    assert it_poa["asking_price"] is None


# ── CHESTERTONS sale-mode wiring ──────────────────────────────────────────────────

def _chest():
    from property_scraper.spiders.chestertons_spider import ChestertonsSpider
    return ChestertonsSpider


def test_default_chestertons_spider_is_rental_unchanged():
    spider = _chest()()
    assert getattr(spider, "listing_type", "rent") == "rent"
    assert "/properties/lettings" in spider.start_url_for("Chelsea")
    assert "/properties/sales" not in spider.start_url_for("Chelsea")


def test_sale_mode_targets_for_sale_section_chestertons():
    spider = _chest()(listing_type="sale")
    assert spider.listing_type == "sale"
    url = spider.start_url_for("Chelsea")
    assert "/properties/sales" in url
    assert "/properties/lettings" not in url


def test_typo_listing_type_falls_back_to_rent_chestertons():
    for bad in ("Sale!", "buy"):
        spider = _chest()(listing_type=bad)
        assert spider.listing_type == "rent"
        assert "/properties/lettings" in spider.start_url_for("Chelsea")


def test_sale_mode_delegates_to_pure_seam_chestertons():
    spider = _chest()(listing_type="sale")
    card = _load("chestertons_card_data_for_sale.json")[0]
    it = spider.parse_card_data_for_sale(card, "Chelsea")
    assert isinstance(it, dict)
    assert it["listing_type"] == "sale"
    assert it["asking_price"] and it["asking_price"] >= 100_000
    assert "price_pcm" not in it and "price_pw" not in it
    # A card whose textContent carries no £ (POA) normalises asking_price to None.
    poa = {**card, "textContent": "For Sale\nOakfield Street, Chelsea, SW10\n2\n2"}
    it_poa = spider.parse_card_data_for_sale(poa, "Chelsea")
    assert it_poa["asking_price"] is None
