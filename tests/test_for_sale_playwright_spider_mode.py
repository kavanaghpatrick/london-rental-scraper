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

import asyncio
import json
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.for_sale

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "for_sale"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def _drain(async_gen):
    """Collect every item yielded by an async generator into a list (no event
    loop assumptions for the caller)."""
    async def _run():
        out = []
        async for x in async_gen:
            out.append(x)
        return out
    return asyncio.run(_run())


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


# ════════════════════════════════════════════════════════════════════════════════
# R1 — the page-parse LOOP must ROUTE to parse_card_data_for_sale in sale mode.
#
# The seams above (parse_card_data_for_sale) already exist and pass; the REAL bug
# (savills_spider:361, knightfrank_spider:401, chestertons_spider:369) is that the
# harvest loop calls the RENTAL parse_card_data UNCONDITIONALLY. These integration
# tests DRIVE the actual async page-parse loop in sale mode with a stubbed page
# whose evaluate() returns the committed sale cards, and assert every YIELDED item
# is a sale item (listing_type=='sale', positive asking_price, NO price_pcm).
#
# RED before the fix: the loop routes sale cards through parse_card_data → yields a
# rental PropertyItem with price_pcm (==0 for a sale card) and no asking_price.
# ════════════════════════════════════════════════════════════════════════════════


class _ScriptedPage:
    """Minimal async stand-in for a Playwright page. `evaluate(script)` dispatches
    on substrings in the script text to return scripted payloads; all wait/close
    methods are async no-ops. Network-free + deterministic."""

    def __init__(self, routes):
        # routes: list of (substring, payload) checked in order for each evaluate().
        self._routes = routes

    async def evaluate(self, script, *args, **kwargs):
        for needle, payload in self._routes:
            if needle in script:
                return payload() if callable(payload) else payload
        return None

    async def query_selector_all(self, *a, **k):
        return []

    async def wait_for_timeout(self, *a, **k):
        return None

    async def wait_for_function(self, *a, **k):
        return None

    async def wait_for_selector(self, *a, **k):
        return None

    async def wait_for_load_state(self, *a, **k):
        return None

    async def reload(self, *a, **k):
        return None

    async def close(self, *a, **k):
        return None


def _fake_response(playwright_page, meta_extra=None):
    """A duck-typed Scrapy response carrying the stubbed playwright_page in meta."""
    meta = {"playwright_page": playwright_page, "request_start": 0.0, "page": 1}
    if meta_extra:
        meta.update(meta_extra)
    body = b"<html></html>"
    return types.SimpleNamespace(
        meta=meta,
        status=200,
        body=body,
        url="https://example.test/for-sale",
        request=types.SimpleNamespace(replace=lambda **k: None),
        text="<html></html>",
        css=lambda *a, **k: types.SimpleNamespace(get=lambda: None),
    )


def _assert_all_sale(items):
    """Every yielded item must be a SALE item: sale-named asking price, no rental
    pcm/pw magnitude, listing_type sale. Accepts dict (sale data-layer path)."""
    assert items, "loop yielded nothing in sale mode"
    for it in items:
        # dict-like access works for both plain dict and scrapy Item.
        get = it.get if hasattr(it, "get") else (lambda k, d=None: it[k] if k in it else d)
        assert get("listing_type") == "sale", f"not a sale item: {dict(it) if hasattr(it,'keys') else it}"
        ask = get("asking_price")
        assert isinstance(ask, int) and ask >= 100_000, f"bad asking_price {ask!r}"
        # The rental pcm field must NOT carry a (zero/garbage) value on a sale item.
        assert not get("price_pcm"), f"sale item leaked price_pcm={get('price_pcm')!r}"
        assert not get("price_pw"), f"sale item leaked price_pw={get('price_pw')!r}"


def test_savills_sale_mode_loop_yields_sale_items():
    spider = _savills()(listing_type="sale")
    cards = _load("savills_card_data_for_sale.json")
    page = _ScriptedPage([
        ("sv-results-listing__item", cards),                 # harvest -> committed sale cards
        ("sv-pagination", {"totalPages": 1, "currentPage": 1, "hasNext": False}),
    ])
    items = _drain(spider.parse_all_pages(_fake_response(page)))
    _assert_all_sale(items)


def test_knightfrank_sale_mode_loop_yields_sale_items():
    spider = _kf()(listing_type="sale")
    cards = _load("knightfrank_card_data_for_sale.json")
    page = _ScriptedPage([
        ("property-features", cards),   # the card-harvest evaluate returns sale cards
        ("noResultsIndicators", False),
        ("no properties match", False),
    ])
    items = _drain(spider.parse_search(_fake_response(page, {"page": 1})))
    _assert_all_sale(items)


def test_chestertons_sale_mode_loop_yields_sale_items():
    spider = _chest()(listing_type="sale")
    cards = _load("chestertons_card_data_for_sale.json")
    page = _ScriptedPage([
        ("Load More", False),                 # no pagination clicks
        ("pegasus-property-card", cards),     # the card-harvest evaluate returns sale cards
    ])
    items = _drain(spider.parse_search(_fake_response(page)))
    _assert_all_sale(items)


# ════════════════════════════════════════════════════════════════════════════════
# R2 — Savills IN-BROWSER harvest drops sale cards because it hard-requires a
# Monthly/Weekly rental token. A "Guide Price £875,000" sale card has neither and
# is dropped → sale harvest emits 0 cards. The fix accepts a sale-price token when
# listing_type=='sale'; the rental gate stays unchanged for rent mode.
#
# We lift the gate logic into a tested pure helper on the spider and feed committed
# innerText. RED before the fix: either the helper doesn't exist, or a Guide-Price
# card is rejected in sale mode.
# ════════════════════════════════════════════════════════════════════════════════

_GUIDE_PRICE_INNERTEXT = (
    "Lower Sloane Street, Chelsea, London SW1W 8AA\n"
    "1 Bedroom 1 Bathroom 466 sq ft\nApartment\nGuide Price £875,000"
)
_RENTAL_INNERTEXT = (
    "Some Street, Chelsea, London SW3 1AA\n"
    "2 Bedrooms 1 Bathroom 721 sq ft\nApartment\n£4,723 Monthly"
)


def test_savills_harvest_accepts_guide_price_card_in_sale_mode():
    """In sale mode a Guide-Price (no Monthly/Weekly token) card MUST survive the
    harvest gate. RED before R2: the rental-only gate drops it (0 cards)."""
    spider = _savills()(listing_type="sale")
    assert hasattr(spider, "harvest_card_passes_gate"), (
        "R2: expected a tested harvest-gate helper "
        "SavillsSpider.harvest_card_passes_gate(inner_text) lifted from the "
        "in-browser gate so it can be unit-tested."
    )
    assert spider.harvest_card_passes_gate(_GUIDE_PRICE_INNERTEXT) is True
    # A rental card (Monthly token) is also fine in sale mode (it still has a £).
    assert spider.harvest_card_passes_gate(_RENTAL_INNERTEXT) is True


def test_savills_harvest_rent_mode_unchanged_drops_guide_price():
    """Rent mode behaviour is UNCHANGED: a sale Guide-Price card (no Monthly/Weekly)
    is still dropped; a rental Monthly card still passes."""
    spider = _savills()()  # default rent
    assert spider.harvest_card_passes_gate(_RENTAL_INNERTEXT) is True
    assert spider.harvest_card_passes_gate(_GUIDE_PRICE_INNERTEXT) is False


def test_savills_harvest_js_source_has_sale_branch():
    """The IN-BROWSER harvest JS (page.evaluate string) must itself branch on the
    vertical so the live crawl — not just the helper — keeps sale cards. RED before
    the fix: the evaluate string hard-gates on monthlyMatch/weeklyMatch only."""
    import inspect
    src = inspect.getsource(_savills().parse_all_pages)
    # The harvest must reference the listing vertical (so sale mode changes the gate)
    # rather than unconditionally `continue`-ing when no Monthly/Weekly token.
    assert "listing_type" in src or "saleMatch" in src or "salePrice" in src, (
        "R2: the Savills harvest JS still has no sale-price branch — a Guide-Price "
        "card is dropped in sale mode."
    )


# ════════════════════════════════════════════════════════════════════════════════
# R4 — Foxtons sale entirely unwired. foxtons_spider.py has ZERO listing_type/
# for_sale references; `-a listing_type=sale` is ignored → it scrapes the rental
# section. The fix adds the listing_type kwarg (default rent, typo->rent), a
# for-sale section (/properties-for-sale/) start URL, and a sale branch routing the
# __NEXT_DATA__ to parse_foxtons_for_sale. Mirror rightmove_spider's sale pattern.
# ════════════════════════════════════════════════════════════════════════════════


def _foxtons():
    from property_scraper.spiders.foxtons_spider import FoxtonsSpider
    return FoxtonsSpider


def test_default_foxtons_spider_is_rental_unchanged():
    spider = _foxtons()()
    assert getattr(spider, "listing_type", "rent") == "rent"
    url = spider.start_url_for("Chelsea")
    assert "properties-to-rent" in url
    assert "properties-for-sale" not in url


def test_sale_mode_targets_for_sale_section_foxtons():
    spider = _foxtons()(listing_type="sale")
    assert spider.listing_type == "sale"
    url = spider.start_url_for("Chelsea")
    assert "properties-for-sale" in url
    assert "properties-to-rent" not in url


def test_typo_listing_type_falls_back_to_rent_foxtons():
    for bad in ("Sale!", "buy"):
        spider = _foxtons()(listing_type=bad)
        assert spider.listing_type == "rent"
        assert "properties-to-rent" in spider.start_url_for("Chelsea")


def test_sale_mode_routes_to_parse_foxtons_for_sale():
    """In sale mode the spider must route the __NEXT_DATA__ property to the pure
    parse_foxtons_for_sale seam (asking_price from priceFrom, never price_pcm)."""
    spider = _foxtons()(listing_type="sale")
    prop = _load("foxtons_for_sale_properties.json")[0]
    it = spider.parse_property_for_sale(prop, "Chelsea")
    assert isinstance(it, dict)
    assert it["listing_type"] == "sale"
    assert it["asking_price"] and it["asking_price"] >= 100_000
    assert "price_pcm" not in it and "price_pw" not in it


def test_foxtons_sale_mode_page_loop_yields_sale_items():
    """Drive the actual parse_search loop in sale mode over a stubbed __NEXT_DATA__
    response carrying the committed for-sale properties; every yielded item must be
    a sale item. RED before R4: parse_search calls the rental parse_property and
    points at the rental section."""
    spider = _foxtons()(listing_type="sale")
    # start_requests seeds per-area bookkeeping; seed it here so the loop reaches the
    # routing decision rather than tripping on a KeyError before it.
    spider.stats["by_area"]["Chelsea"] = {"count": 0, "pages": 0}
    props = _load("foxtons_for_sale_properties.json")
    next_data = {"props": {"pageProps": {"pageData": {"data": {"data": props}}}}}
    script = json.dumps(next_data)

    def _css(sel):
        # parse_search reads response.css('script#__NEXT_DATA__::text').get()
        if "__NEXT_DATA__" in sel:
            return types.SimpleNamespace(get=lambda: script)
        return types.SimpleNamespace(get=lambda: None)

    resp = types.SimpleNamespace(
        meta={"area": "Chelsea", "page": 1, "request_start": 0.0},
        status=200,
        body=b"x",
        url="https://www.foxtons.co.uk/properties-for-sale/chelsea/",
        css=_css,
    )
    out = list(spider.parse_search(resp))
    # parse_search may also yield a pagination Request; keep only the sale items.
    items = [x for x in out if hasattr(x, "get") and x.get("listing_type") == "sale"]
    _assert_all_sale(items)
