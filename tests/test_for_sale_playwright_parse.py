"""test_for_sale_playwright_parse.py — TDD RED contract for the THREE Playwright
for-sale PARSE SEAMS (Inc2): parse_savills_for_sale / parse_knightfrank_for_sale /
parse_chestertons_for_sale in for_sale.listing_parse.

WHAT THIS PINS (Inc2 §3 / §4 / §5 FILE A)
-----------------------------------------
The Playwright spiders (savills/knightfrank/chestertons) emit a plain `card_data` dict
from their in-page `page.evaluate()` (NOT a __NEXT_DATA__ blob). In FOR-SALE mode each
spider must DELEGATE to a PURE seam `parse_<site>_for_sale(card_data, area)` that REUSES
the rental extraction patterns (postcode / sqft / beds / baths / id / url) and differs
ONLY in routing the headline £ into `asking_price` (sale magnitude) — never price_pcm /
price_pw. These seams are exercised against COMMITTED card-dict fixtures (no network),
exactly like tests/test_spider_parsing.py exercises the rental seams.

This file is written FIRST and MUST FAIL (RED) until Group B implements the three
parse_*_for_sale functions: the only legitimate reason for failure is the undefined
production symbols, not a test/fixture defect (the fixtures are hand-derived from the
§3 extraction maps and the frozen-tuple selector tests pin row-0 exactly).

ZERO RENTAL REGRESSION: nothing here imports or mutates rental_price_models_v20,
canonical_predict, the parity-gated xgboost.js, or the rental `listings` table.
CI-SAFE: marker `for_sale`; committed fixtures only; no DB/network/OCR/node deps.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.for_sale

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "for_sale"


# ── Network-free fixture loaders (committed card-dict samples, §4) ────────────────

@pytest.fixture(scope="module")
def savills_cards():
    return json.loads((FIXTURES / "savills_card_data_for_sale.json").read_text())


@pytest.fixture(scope="module")
def kf_cards():
    return json.loads((FIXTURES / "knightfrank_card_data_for_sale.json").read_text())


@pytest.fixture(scope="module")
def chest_cards():
    return json.loads((FIXTURES / "chestertons_card_data_for_sale.json").read_text())


def test_playwright_fixtures_exist_and_nonempty():
    for name in (
        "savills_card_data_for_sale.json",
        "knightfrank_card_data_for_sale.json",
        "chestertons_card_data_for_sale.json",
    ):
        p = FIXTURES / name
        assert p.exists(), f"committed for-sale card fixture missing: {name}"
        cards = json.loads(p.read_text())
        assert isinstance(cards, list) and len(cards) >= 3, f"{name} must hold >=3 cards"


# ── SAVILLS for-sale parse seam ───────────────────────────────────────────────────

class TestSavillsForSaleParse:
    def test_all_cards_parse(self, savills_cards):
        from for_sale.listing_parse import parse_savills_for_sale
        parsed = [parse_savills_for_sale(c, "Chelsea") for c in savills_cards]
        assert all(it is not None for it in parsed)
        assert len(parsed) == len(savills_cards)

    def test_asking_price_is_sale_not_pcm(self, savills_cards):
        from for_sale.listing_parse import parse_savills_for_sale
        for c in savills_cards:
            it = parse_savills_for_sale(c, "Chelsea")
            assert it["listing_type"] == "sale"
            assert it["asking_price"] >= 100_000
            # The sale item carries NO rental magnitude field.
            assert "price_pcm" not in it
            assert "price_pw" not in it

    def test_source_and_identity(self, savills_cards):
        from for_sale.listing_parse import parse_savills_for_sale
        for c in savills_cards:
            it = parse_savills_for_sale(c, "Chelsea")
            assert it["source"] == "savills"
            # property_id is the /property-detail/<slug> id (not a content hash) for
            # every fixture row (they all carry a property-detail href).
            m = re.search(r"/property-detail/([^/?#]+)", c["href"])
            assert it["property_id"] == m.group(1)
            assert it["url"].startswith("https://")

    def test_beds_baths_sqft_postcode_reused(self, savills_cards):
        from for_sale.listing_parse import parse_savills_for_sale
        for c in savills_cards:
            it = parse_savills_for_sale(c, "Chelsea")
            for k in ("bedrooms", "bathrooms", "size_sqft"):
                assert it[k] is None or isinstance(it[k], int)
            # postcode is a string or None; the size-absent row must not crash.
            assert it["postcode"] is None or isinstance(it["postcode"], str)

    def test_price_qualifier_captured(self, savills_cards):
        from for_sale.listing_parse import parse_savills_for_sale
        quals = {parse_savills_for_sale(c, "Chelsea")["price_qualifier"] for c in savills_cards}
        assert "Offers in Excess of" in quals

    def test_missing_href_and_text_returns_none(self):
        from for_sale.listing_parse import parse_savills_for_sale
        assert parse_savills_for_sale({}, "Chelsea") is None


# ── KNIGHT FRANK for-sale parse seam ──────────────────────────────────────────────

class TestKnightFrankForSaleParse:
    def test_all_cards_parse(self, kf_cards):
        from for_sale.listing_parse import parse_knightfrank_for_sale
        parsed = [parse_knightfrank_for_sale(c, "Chelsea") for c in kf_cards]
        assert all(it is not None for it in parsed)
        assert len(parsed) == len(kf_cards)

    def test_asking_price_is_sale_not_pcm(self, kf_cards):
        from for_sale.listing_parse import parse_knightfrank_for_sale
        for c in kf_cards:
            it = parse_knightfrank_for_sale(c, "Chelsea")
            assert it["listing_type"] == "sale"
            assert it["asking_price"] >= 100_000
            assert "price_pcm" not in it
            assert "price_pw" not in it

    def test_source_and_identity(self, kf_cards):
        from for_sale.listing_parse import parse_knightfrank_for_sale
        for c in kf_cards:
            it = parse_knightfrank_for_sale(c, "Chelsea")
            assert it["source"] == "knightfrank"
            assert re.fullmatch(r"[a-z]{3}\d+", it["property_id"]), it["property_id"]
            assert it["url"].startswith("https://")

    def test_beds_baths_positional_heuristic(self, kf_cards):
        """The KF beds/baths come from the positional single-digit sequence AFTER the
        address line (nums[-3]/nums[-2]). Row 0 = Alexandra Mansions 2-bed/1-bath."""
        from for_sale.listing_parse import parse_knightfrank_for_sale
        it = parse_knightfrank_for_sale(kf_cards[0], "Chelsea")
        assert it["bedrooms"] == 2
        assert it["bathrooms"] == 1

    def test_size_none_when_no_sqft_token(self, kf_cards):
        from for_sale.listing_parse import parse_knightfrank_for_sale
        # The committed sample includes >=1 row with no 'sqft' token (row 2).
        sizes = [parse_knightfrank_for_sale(c, "Chelsea")["size_sqft"] for c in kf_cards]
        assert None in sizes

    def test_missing_text_and_href_returns_none(self):
        from for_sale.listing_parse import parse_knightfrank_for_sale
        assert parse_knightfrank_for_sale({}, "Chelsea") is None


# ── CHESTERTONS for-sale parse seam ───────────────────────────────────────────────

class TestChestertonsForSaleParse:
    def test_all_cards_parse(self, chest_cards):
        from for_sale.listing_parse import parse_chestertons_for_sale
        parsed = [parse_chestertons_for_sale(c, "Chelsea") for c in chest_cards]
        assert all(it is not None for it in parsed)
        assert len(parsed) == len(chest_cards)

    def test_asking_price_is_sale_not_pcm(self, chest_cards):
        from for_sale.listing_parse import parse_chestertons_for_sale
        for c in chest_cards:
            it = parse_chestertons_for_sale(c, "Chelsea")
            assert it["listing_type"] == "sale"
            assert it["asking_price"] >= 100_000
            assert "price_pcm" not in it
            assert "price_pw" not in it

    def test_source_and_identity(self, chest_cards):
        from for_sale.listing_parse import parse_chestertons_for_sale
        for c in chest_cards:
            it = parse_chestertons_for_sale(c, "Chelsea")
            assert it["source"] == "chestertons"
            assert it["url"].startswith("https://www.chestertons.co.uk")

    def test_sales_url_yields_stable_id_not_hash(self):
        """MANDATORY /sales/ id-fix regression: the rental regex only matched /lettings/;
        the sale seam MUST accept /sales/ and derive the STABLE '<num>_<REF>' id (not a
        content hash). `/properties/12345/sales/ABC123` -> '12345_ABC123'."""
        from for_sale.listing_parse import parse_chestertons_for_sale
        card = {
            "href": "/properties/12345/sales/ABC123",
            "address": "Oakfield Street, Chelsea, SW10",
            "letType": "For Sale",
            "textContent": "For Sale\nOakfield Street, Chelsea, SW10\n2\n2\n£3,000,000",
        }
        it = parse_chestertons_for_sale(card, "Chelsea")
        assert it["property_id"] == "12345_ABC123"
        assert not it["property_id"].startswith("chestertons_")

    def test_under_offer_and_new_build_flags(self, chest_cards):
        from for_sale.listing_parse import parse_chestertons_for_sale
        flags = [
            (it["is_under_offer"], it["is_new_build"])
            for it in (parse_chestertons_for_sale(c, "Chelsea") for c in chest_cards)
        ]
        # The committed sample carries >=1 under-offer row and >=1 new-build row.
        assert any(uo == 1 for uo, _ in flags)
        assert any(nb == 1 for _, nb in flags)

    def test_size_none_when_no_ft_token(self, chest_cards):
        from for_sale.listing_parse import parse_chestertons_for_sale
        sizes = [parse_chestertons_for_sale(c, "Chelsea")["size_sqft"] for c in chest_cards]
        assert None in sizes

    def test_missing_returns_none(self):
        from for_sale.listing_parse import parse_chestertons_for_sale
        assert parse_chestertons_for_sale({}, "Chelsea") is None


# ── SELECTOR REGRESSION — frozen row-0 tuples (markup/JSON-shape drift) ────────────

@pytest.mark.selector
class TestPlaywrightForSaleSelectorRegression:
    """Frozen (property_id, asking_price, bedrooms, postcode, size_sqft) for fixture
    row 0 of each site. Hand-derived from the §3 extraction maps; if a parse seam drifts
    these break loudly."""

    def test_savills_first_row_exact(self, savills_cards):
        from for_sale.listing_parse import parse_savills_for_sale
        it = parse_savills_for_sale(savills_cards[0], "Chelsea")
        assert (
            it["property_id"], it["asking_price"], it["bedrooms"],
            it["postcode"], it["size_sqft"],
        ) == ("gbcrres240123", 875000, 1, "SW1W 8AA", 466)

    def test_knightfrank_first_row_exact(self, kf_cards):
        from for_sale.listing_parse import parse_knightfrank_for_sale
        it = parse_knightfrank_for_sale(kf_cards[0], "Chelsea")
        assert (
            it["property_id"], it["asking_price"], it["bedrooms"],
            it["postcode"], it["size_sqft"],
        ) == ("lon012345", 1950000, 2, "SW3", 625)

    def test_chestertons_first_row_exact(self, chest_cards):
        from for_sale.listing_parse import parse_chestertons_for_sale
        it = parse_chestertons_for_sale(chest_cards[0], "Chelsea")
        assert (
            it["property_id"], it["asking_price"], it["bedrooms"],
            it["postcode"], it["size_sqft"],
        ) == ("12345_ABC123", 4950000, 5, "SW10", 2540)


# ── ISOLATION GUARD — the for-sale parse module must not import the rental model ──

def test_listing_parse_module_does_not_import_rental_model():
    import for_sale.listing_parse as m
    src = Path(m.__file__).read_text()
    for banned in (
        "import rental_price_models_v20",
        "from rental_price_models_v20",
        "import canonical_predict",
        "from canonical_predict",
    ):
        assert banned not in src, f"for_sale.listing_parse illegally couples to rental: {banned}"
