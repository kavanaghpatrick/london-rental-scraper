"""
test_for_sale_scrape_layer.py — TDD contract for the FOR-SALE vertical's
SCRAPED-LISTINGS data layer (increment #1).

ARCHITECTURE (per the project directive): the for-sale tool REUSES the existing
scraping infrastructure pointed at the FOR-SALE sections of the realtor sites
(rightmove /property-for-sale/, foxtons /properties-for-sale/, …). The data is
SCRAPED FOR-SALE *ASKING* listings — NOT Land Registry sold-price data. (Land
Registry is a possible FUTURE enrichment/validation, out of scope here.)

This increment proves the FOUNDATION the directive asked for:
  1. a for-sale DATA SCHEMA isolated from rentals — a sale ASKING price field of a
     different magnitude than rental price_pcm, in its own item, so a £875,000 sale
     can never be mistaken for / written into the rental price_pcm column.
  2. for-sale PARSE SEAMS that reuse the rental extraction patterns (id/url/address/
     postcode/beds/baths/type/sqft are IDENTICAL; only the URL and the price field
     differ) against COMMITTED real-site for-sale __NEXT_DATA__ fixtures.

CI-SAFE: reads COMMITTED for-sale fixtures captured from the live sites
(tests/fixtures/for_sale/{rightmove,foxtons}_for_sale_properties.json — real
__NEXT_DATA__ search-result slices, PII-free public listing data). No live network,
no DB, no binary deps → runs on every PR. Mirrors tests/test_spider_parsing.py.

Marker: `for_sale` (registered in pytest.ini under strict_markers). Plain unit
tests (no DB/network) so they ALWAYS run and gate the PR, and are on the
anti-silent-skip allowlist (tests/test_ci_critical_tests_run.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.for_sale

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "for_sale"
RM_FIXTURE = FIXTURES / "rightmove_for_sale_properties.json"
FX_FIXTURE = FIXTURES / "foxtons_for_sale_properties.json"


def load(name: Path):
    with open(name) as f:
        return json.load(f)


# ── Committed real for-sale fixtures must exist and look like for-sale data ──

def test_for_sale_fixtures_exist_and_nonempty():
    for fx in (RM_FIXTURE, FX_FIXTURE):
        assert fx.exists(), f"missing committed for-sale fixture: {fx.name}"
        data = load(fx)
        assert isinstance(data, list) and len(data) >= 3, f"{fx.name} too small/empty"


def test_rightmove_fixture_is_for_sale_not_rental():
    """The captured Rightmove for-sale search property uses price.frequency
    'not specified' (the sale marker) — NOT 'monthly'/'weekly' (rental markers).
    This is the structural difference the for-sale parser keys on."""
    props = load(RM_FIXTURE)
    freqs = {p["price"]["frequency"] for p in props}
    assert freqs == {"not specified"}, f"expected sale frequency, got {freqs}"
    # Sale magnitude, not pcm: every asking price is well into the hundreds of thousands+
    assert all(p["price"]["amount"] >= 100_000 for p in props)


def test_foxtons_fixture_is_for_sale_not_rental():
    """The captured Foxtons for-sale property carries instructionType 'sale' and the
    asking price in priceFrom — NOT the rental pricePcm field (which is junk for sales)."""
    props = load(FX_FIXTURE)
    assert {p["instructionType"] for p in props} == {"sale"}
    assert all(int(p["priceFrom"]) >= 100_000 for p in props)


# ── The for-sale item schema (for_sale.items) ────────────────────────────────

def test_sale_item_has_asking_price_distinct_from_rental_pcm():
    """The for-sale item must expose a SALE-magnitude asking-price field that is
    NOT named price_pcm and NOT routed through the rental validator's pcm ceiling."""
    from for_sale.items import SaleListingItem, SALE_PRICE_FIELD
    item = SaleListingItem()
    # The canonical sale price field exists and is sale-named, not rent-named.
    assert SALE_PRICE_FIELD in item.fields
    assert SALE_PRICE_FIELD != "price_pcm"
    assert "pcm" not in SALE_PRICE_FIELD and "pw" not in SALE_PRICE_FIELD
    # And the rental-only pcm/pw fields are absent from the sale schema (isolation).
    assert "price_pcm" not in item.fields
    assert "price_pw" not in item.fields


def test_validate_sale_item_accepts_london_sale_magnitudes():
    """A £875k–£40M London asking price must VALIDATE (the rental validator rejects
    >£500k as 'suspiciously high price_pcm' — the sale validator must not)."""
    from for_sale.items import SaleListingItem, validate_sale_item, SALE_PRICE_FIELD
    item = SaleListingItem()
    item["source"] = "rightmove"
    item["property_id"] = "89029677"
    item["url"] = "https://www.rightmove.co.uk/properties/89029677"
    item["address"] = "Lower Sloane Street, Chelsea, London SW1W"
    item[SALE_PRICE_FIELD] = 875_000
    ok, issues = validate_sale_item(item)
    assert ok, f"valid London sale rejected: {issues}"
    # A 40M mansion is still valid for sale.
    item[SALE_PRICE_FIELD] = 40_000_000
    ok, _ = validate_sale_item(item)
    assert ok


def test_validate_sale_item_rejects_rental_magnitude_as_sale():
    """A £3,000 value is a monthly RENT, not a sale price — the sale validator must
    reject it (floor guards against a rental leaking into the sale table)."""
    from for_sale.items import SaleListingItem, validate_sale_item, SALE_PRICE_FIELD
    item = SaleListingItem()
    item["source"] = "rightmove"
    item["property_id"] = "x"
    item["url"] = "https://www.rightmove.co.uk/properties/x"
    item["address"] = "Somewhere, SW3"
    item[SALE_PRICE_FIELD] = 3_000  # a rent, not a sale price
    ok, issues = validate_sale_item(item)
    assert not ok and any("price" in i.lower() for i in issues)


# ── Rightmove for-sale parse seam (for_sale.listing_parse) ───────────────────

@pytest.fixture(scope="module")
def parse():
    from for_sale import listing_parse
    return listing_parse


class TestRightmoveForSaleParse:
    @pytest.fixture(scope="module")
    def props(self):
        return load(RM_FIXTURE)

    def test_all_props_parse(self, parse, props):
        for p in props:
            item = parse.parse_rightmove_for_sale(p, "Chelsea")
            assert item is not None, f"parse returned None for {p.get('id')}"

    def test_identity_and_url(self, parse, props):
        for p in props:
            item = parse.parse_rightmove_for_sale(p, "Chelsea")
            assert item["source"] == "rightmove"
            assert item["property_id"]
            assert item["url"].startswith("https://www.rightmove.co.uk/properties/")
            assert item["listing_type"] == "sale"
            assert item["area"] == "Chelsea"

    def test_asking_price_is_sale_amount_not_pcm(self, parse, props):
        """The asking price is the raw sale amount (hundreds of thousands+), stored in
        the sale field — it must NEVER be the *12/52 pcm-converted value."""
        from for_sale.items import SALE_PRICE_FIELD
        for p in props:
            item = parse.parse_rightmove_for_sale(p, "Chelsea")
            assert item[SALE_PRICE_FIELD] == p["price"]["amount"]
            assert item[SALE_PRICE_FIELD] >= 100_000
            assert "price_pcm" not in item.fields or item.get("price_pcm") in (None, 0)

    def test_beds_baths_type_sqft_reused(self, parse, props):
        """beds/baths/type/sqft extraction is the SAME as the rental seam."""
        for p in props:
            item = parse.parse_rightmove_for_sale(p, "Chelsea")
            beds = item.get("bedrooms")
            assert beds is None or isinstance(beds, int)
            assert item.get("property_type")  # propertySubType present in fixture
        # At least one fixture row carries sqft (the £875k 1-bed flat shows 466 sq ft).
        sqfts = [parse.parse_rightmove_for_sale(p, "Chelsea").get("size_sqft") for p in props]
        assert any(s and s > 0 for s in sqfts), "no sqft parsed — displaySize seam broke"

    def test_postcode_extracted_when_in_address(self, parse):
        prop = {
            "id": "pc1",
            "propertyUrl": "/properties/pc1#/?channel=RES_BUY",
            "price": {"amount": 1_250_000, "frequency": "not specified"},
            "displayAddress": "10 Cadogan Square, Knightsbridge, London SW1X 0JU",
            "bedrooms": 2, "bathrooms": 2, "propertySubType": "Flat",
            "location": {"latitude": 51.5, "longitude": -0.16},
        }
        item = parse.parse_rightmove_for_sale(prop, "Chelsea")
        assert item["postcode"] == "SW1X0JU"

    def test_missing_id_returns_none(self, parse):
        assert parse.parse_rightmove_for_sale({"price": {"amount": 900000}}, "Chelsea") is None


class TestFoxtonsForSaleParse:
    @pytest.fixture(scope="module")
    def props(self):
        return load(FX_FIXTURE)

    def test_all_props_parse(self, parse, props):
        for p in props:
            item = parse.parse_foxtons_for_sale(p, "Chelsea")
            assert item is not None

    def test_asking_price_from_pricefrom_not_pricepcm(self, parse, props):
        """Foxtons sale price is priceFrom; pricePcm is junk (annualized) for sales and
        must be ignored."""
        from for_sale.items import SALE_PRICE_FIELD
        for p in props:
            item = parse.parse_foxtons_for_sale(p, "Chelsea")
            assert item[SALE_PRICE_FIELD] == int(p["priceFrom"])
            # never the junk pcm value
            assert item[SALE_PRICE_FIELD] != p.get("pricePcm")

    def test_identity_postcode_beds(self, parse, props):
        for p in props:
            item = parse.parse_foxtons_for_sale(p, "Chelsea")
            assert item["source"] == "foxtons"
            assert item["listing_type"] == "sale"
            assert item["property_id"]
            assert item["url"].startswith("https://www.foxtons.co.uk/")
            beds = item.get("bedrooms")
            assert beds is None or isinstance(beds, int)
            # postcodeShort populates a postcode/district
            if p.get("postcodeShort"):
                assert item.get("postcode")


# ── Selector-regression — frozen exact extraction (drift fails loudly) ───────

@pytest.mark.selector
class TestForSaleSelectorRegression:
    def test_rightmove_first_row_exact(self, parse):
        # NB: Rightmove for-sale search displayAddress frequently OMITS the postcode
        # ("Lower Sloane Street, Chelsea, London") — a real-data limitation shared with
        # the rental seam — so postcode is None here; detail-page/coords enrichment
        # (a later increment) backfills it. The other fields parse exactly.
        from for_sale.items import SALE_PRICE_FIELD
        props = load(RM_FIXTURE)
        item = parse.parse_rightmove_for_sale(props[0], "Chelsea")
        assert (item["property_id"], item[SALE_PRICE_FIELD], item["bedrooms"],
                item["postcode"], item["size_sqft"]) == (
            "89029677", 875_000, 1, None, 466)

    def test_foxtons_first_row_exact(self, parse):
        from for_sale.items import SALE_PRICE_FIELD
        props = load(FX_FIXTURE)
        item = parse.parse_foxtons_for_sale(props[0], "Chelsea")
        assert (item["property_id"], item[SALE_PRICE_FIELD], item["bedrooms"],
                item["postcode"]) == ("nwca5232210", 13_858_000, 5, "SW10")


# ── ISOLATION GUARD — the for-sale vertical must not couple to the rental stack ─

def test_for_sale_modules_do_not_import_rental_model():
    """Hard isolation: for_sale must not import the parity-gated rental model chain.
    It REUSES PATTERNS (re-implements the postcode regex / a thin item) but never
    couples to rental_price_models_v20 / canonical_predict, so a rental retrain can
    never perturb for-sale and vice-versa."""
    import for_sale.items as a
    import for_sale.listing_parse as b
    for mod in (a, b):
        src = Path(mod.__file__).read_text()
        for banned in ("import rental_price_models_v20", "from rental_price_models_v20",
                       "import canonical_predict", "from canonical_predict"):
            assert banned not in src, f"{Path(mod.__file__).name} couples to rental: {banned}"
