#!/usr/bin/env python3
"""Unit tests for spider parsing — fixture-based, NO live network calls.

Covers the pure-Python parse seams of each spider:
  - rightmove.parse_property(prop_dict, area)      -> __NEXT_DATA__ search property
  - savills.parse_card_data(card_dict)             -> Playwright-extracted card

These exercise field extraction (price/beds/postcode/sqft/url/source) and the
price-period conversion logic against SAVED fixtures captured from the live sites.

Selector-regression intent: if a site changes the JSON/card shape, the parser
output drifts and these assertions fail loudly in CI (see TestSelectorRegression).

Fixtures live in tests/fixtures/ and are captured once (see tools/capture_fixtures
or the docstring of each fixture). Re-capture only when a site genuinely changes.

Other sources (knightfrank, chestertons, foxtons, openrent) follow the SAME
pattern via their parse_card_data / parse_property seams — owned by the
respective scraper teammates; add their fixtures + classes here.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from property_scraper.spiders.rightmove_spider import RightmoveSpider
from property_scraper.spiders.savills_spider import SavillsSpider
from property_scraper.spiders.foxtons_spider import FoxtonsSpider
from property_scraper.spiders.openrent_spider import OpenRentSpider

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name):
    with open(FIXTURES / name) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Rightmove — parse_property(prop, area) from __NEXT_DATA__ search results
# ---------------------------------------------------------------------------
class TestRightmoveParsing:
    @pytest.fixture(scope="class")
    def spider(self):
        # Constructor runs standalone (no crawl) — only sets config/stats.
        return RightmoveSpider(areas="Chelsea", max_pages="1", fetch_details="false")

    @pytest.fixture(scope="class")
    def props(self):
        return load_fixture("rightmove_search_properties.json")

    def test_fixture_present_and_nonempty(self, props):
        assert isinstance(props, list) and len(props) >= 1

    def test_all_props_parse_to_items(self, spider, props):
        for prop in props:
            item = spider.parse_property(prop, "Chelsea")
            assert item is not None, f"parse_property returned None for {prop.get('id')}"

    def test_required_fields_populated(self, spider, props):
        for prop in props:
            item = spider.parse_property(prop, "Chelsea")
            # Identity
            assert item["source"] == "rightmove"
            assert item["property_id"]
            assert item["url"].startswith("https://www.rightmove.co.uk/properties/")
            assert item["area"] == "Chelsea"
            # Price present and positive
            assert item["price_pcm"] and item["price_pcm"] > 0
            assert item["price_period"] in ("pcm", "pw")

    def test_price_period_conversion_consistent(self, spider, props):
        """price_pcm and price_pw must be internally consistent for the period."""
        for prop in props:
            item = spider.parse_property(prop, "Chelsea")
            if item["price_period"] == "pw":
                # pcm derived from weekly: price*52/12
                assert item["price_pcm"] == int(item["price"] * 52 / 12)
            else:
                assert item["price_pcm"] == item["price"]

    def test_bedrooms_int_or_none(self, spider, props):
        for prop in props:
            item = spider.parse_property(prop, "Chelsea")
            beds = item.get("bedrooms")
            assert beds is None or isinstance(beds, int)

    def test_postcode_extracted_when_present_in_address(self, spider):
        """Postcode regex must pull a valid UK outcode/postcode from displayAddress."""
        prop = {
            "id": "test1",
            "propertyUrl": "/properties/test1",
            "price": {"amount": 3000, "frequency": "monthly"},
            "displayAddress": "10 Cadogan Square, Knightsbridge, London SW1X 0JU",
            "bedrooms": 2,
            "location": {"latitude": 51.5, "longitude": -0.16},
        }
        item = spider.parse_property(prop, "Chelsea")
        assert item["postcode"] == "SW1X0JU"

    def test_missing_id_returns_none(self, spider):
        item = spider.parse_property({"price": {"amount": 1000}}, "Chelsea")
        assert item is None

    def test_sqft_from_displaysize(self, spider):
        prop = {
            "id": "sz1",
            "propertyUrl": "/properties/sz1",
            "price": {"amount": 4000, "frequency": "monthly"},
            "displayAddress": "Flat, SW3 1AA",
            "bedrooms": 1,
            "displaySize": "750 sq ft",
            "location": {},
        }
        item = spider.parse_property(prop, "Chelsea")
        assert item["size_sqft"] == 750


# ---------------------------------------------------------------------------
# R3 — sqft regex must tolerate the REAL Rightmove search format "675 sq. ft."
#       (periods + spaces). The old regex ([\d,]+)\s*sq\s*ft cannot match a
#       period after "sq" or before "ft", so size_sqft silently drops to None.
#       T2: iterate the REAL committed fixture and assert every displaySize that
#       names a sqft figure yields a positive int. RED before the fix.
# ---------------------------------------------------------------------------
import re as _re


class TestRightmoveSqftRegexRealFormat:
    @pytest.fixture(scope="class")
    def spider(self):
        return RightmoveSpider(areas="Chelsea", max_pages="1", fetch_details="false")

    @pytest.fixture(scope="class")
    def props(self):
        return load_fixture("rightmove_search_properties.json")

    def test_fixture_has_period_format_sqft(self, props):
        """Guard the test itself: the real fixture DOES contain at least one
        'NNN sq. ft.' (period) displaySize — otherwise this regression test is
        vacuous. (Today: '675 sq. ft.', '274 sq. ft.')"""
        sq = _re.compile(r"\d+\s*sq", _re.I)
        named = [p.get("displaySize", "") for p in props
                 if p.get("displaySize") and sq.search(p["displaySize"])]
        assert named, "fixture carries no '<n> sq...' displaySize — cannot test R3"
        assert any("sq." in ds or "sq ." in ds for ds in named), (
            "fixture has no period-bearing 'sq. ft.' displaySize — R3 would be vacuous"
        )

    def test_every_sqft_displaysize_parses_to_positive_int(self, spider, props):
        """Every prop whose displaySize names a sqft figure (/\\d+\\s*sq/i) must
        parse to a positive int size_sqft. FAILS today: '675 sq. ft.' and
        '274 sq. ft.' yield None under the period-blind regex."""
        sq = _re.compile(r"\d+\s*sq", _re.I)
        checked = 0
        for prop in props:
            ds = prop.get("displaySize") or ""
            if not sq.search(ds):
                continue
            checked += 1
            item = spider.parse_property(prop, "Chelsea")
            val = item.get("size_sqft")
            assert isinstance(val, int) and val > 0, (
                f"displaySize {ds!r} (id={prop.get('id')}) parsed to {val!r}; "
                f"the sqft regex must tolerate periods/spaces (R3)."
            )
        assert checked >= 1


# ---------------------------------------------------------------------------
# R3 (chestertons) — the card sqft regex (\d{3,5})\s*ft drops comma-thousands
#       and period formats. The fix must KEEP matching the bare-"ft" card format
#       Chestertons currently emits (zero rental regression) while ALSO matching
#       "1,234 sq ft" / "675 sq. ft.". RED today on the comma-thousands case.
# ---------------------------------------------------------------------------
class TestChestertonsSqftRegex:
    @pytest.fixture(scope="class")
    def spider(self):
        from property_scraper.spiders.chestertons_spider import ChestertonsSpider
        return ChestertonsSpider()

    def _card(self, body):
        # Minimal chestertons card dict (the shape _extract_and_yield_cards emits).
        return {
            "href": "/properties/12345/lettings/ABC1",
            "address": "Oakfield Street, Chelsea, SW10",
            "letType": "Long Let",
            "textContent": f"Long Let\nOakfield Street, Chelsea, SW10\n2\n2\n{body}\n£3,500\n(pcm)",
        }

    def test_bare_ft_still_parses_zero_regression(self, spider):
        """The format Chestertons cards emit today ('2540 ft') MUST keep parsing —
        this guards against a regression when the regex is broadened for R3."""
        item = spider.parse_card_data(self._card("2540 ft"))
        assert item["size_sqft"] == 2540

    def test_comma_thousands_sqft_parses(self, spider):
        """'1,234 sq ft' must parse to 1234. FAILS today: (\\d{3,5})\\s*ft cannot
        cross the comma/space/'sq' and yields None."""
        item = spider.parse_card_data(self._card("1,234 sq ft"))
        assert item["size_sqft"] == 1234

    def test_period_sqft_parses(self, spider):
        """'675 sq. ft.' must parse to 675 (period-tolerant, R3)."""
        item = spider.parse_card_data(self._card("675 sq. ft."))
        assert item["size_sqft"] == 675


# ---------------------------------------------------------------------------
# R3 (rightmove_enricher) — the detail-text sqft fallback regex must tolerate
#       periods. The enricher's parse_detail text-fallback list at lines 178-181
#       drives DB back-fill; "675 sq. ft." in page text must extract 675.
# ---------------------------------------------------------------------------
class TestRightmoveEnricherSqftRegex:
    def _enricher_source(self):
        """Read the enricher module SOURCE so the test pins the ACTUAL production
        regex text, not a copy that could drift."""
        from property_scraper.spiders import rightmove_enricher
        return Path(rightmove_enricher.__file__).read_text()

    def test_enricher_sqft_regex_tolerates_periods(self):
        """The enricher's page-text sqft fallback must tolerate the real
        '675 sq. ft.' search format (periods between 'sq' and 'ft'). The old
        r'(\\d{3,5})\\s*sq\\.?\\s*ft' patterns are period-blind there. RED today:
        the enricher source carries none of the period-tolerant sq[\\s.]*ft form."""
        import re as _r
        # Anchor on REAL fixture data so this tracks real-site format, not a sample.
        props = load_fixture("rightmove_search_properties.json")
        period_samples = [p["displaySize"] for p in props
                          if p.get("displaySize") and "sq." in p["displaySize"]]
        assert period_samples, "fixture lost its 'sq. ft.' samples — cannot test enricher R3"
        fixed = _r.compile(r"([\d,]+)\s*sq[\s.]*ft", _r.I)
        for s in period_samples:
            assert fixed.search(s), f"period format {s!r} unmatched by spec regex"
        # The enricher SOURCE must carry the period-tolerant token (R3 fix).
        src = self._enricher_source()
        assert r"sq[\s.]*ft" in src, (
            "rightmove_enricher still lacks a period-tolerant sq[\\s.]*ft sqft "
            "pattern in its source (R3)."
        )


# ---------------------------------------------------------------------------
# Savills — parse_card_data(card) from Playwright-extracted card dict
# ---------------------------------------------------------------------------
class TestSavillsParsing:
    @pytest.fixture(scope="class")
    def spider(self):
        return SavillsSpider()

    @pytest.fixture(scope="class")
    def cards(self):
        return load_fixture("savills_card_data.json")

    def test_fixture_present(self, cards):
        assert isinstance(cards, list) and len(cards) >= 1

    def test_required_fields_populated(self, spider, cards):
        for card in cards:
            item = spider.parse_card_data(card)
            assert item is not None
            assert item["source"] == "savills"
            assert item["property_id"]
            assert item["url"].startswith("https://search.savills.com/property-detail/")
            assert item["price_pcm"] and item["price_pcm"] > 0
            assert item["size_sqft"] and item["size_sqft"] > 0  # savills always has sqft

    def test_property_id_from_url(self, spider):
        card = {
            "href": "https://search.savills.com/property-detail/gbtest123abc",
            "text": "Some Street, SW1 1AA\n1 Bedroom 500 sq ft\n£2,000 Monthly",
            "address": "Some Street, SW1 1AA",
            "sqft": "500", "price": 2000, "beds": "1", "baths": "1",
            "postcode": "SW1 1AA", "furnished": None,
        }
        item = spider.parse_card_data(card)
        assert item["property_id"] == "gbtest123abc"

    def test_weekly_price_already_converted_by_js(self, spider, cards):
        """Savills JS converts weekly->monthly before parse_card_data; price is pcm."""
        for card in cards:
            item = spider.parse_card_data(card)
            # price_pw derived from pcm
            assert item["price_pw"] == int(item["price_pcm"] * 12 / 52)

    def test_postcode_preserved(self, spider, cards):
        for card in cards:
            item = spider.parse_card_data(card)
            assert item["postcode"]  # full postcode preserved from card

    def test_beds_int_or_none(self, spider, cards):
        for card in cards:
            item = spider.parse_card_data(card)
            beds = item.get("bedrooms")
            assert beds is None or isinstance(beds, int)


# ---------------------------------------------------------------------------
# Foxtons — parse_property(prop, area, response) from __NEXT_DATA__ search results
# (owned by the foxtons scraper; fixture + structural tests seeded here)
# ---------------------------------------------------------------------------
class TestFoxtonsParsing:
    @pytest.fixture(scope="class")
    def spider(self):
        return FoxtonsSpider()

    @pytest.fixture(scope="class")
    def props(self):
        return load_fixture("foxtons_properties.json")

    def test_fixture_present(self, props):
        assert isinstance(props, list) and len(props) >= 1

    def test_required_fields_populated(self, spider, props):
        for prop in props:
            item = spider.parse_property(prop, "Chelsea")
            assert item is not None
            assert item["source"] == "foxtons"
            assert item["property_id"]
            assert item["url"].startswith("https://www.foxtons.co.uk/")
            assert item["price_pcm"] and item["price_pcm"] > 0

    def test_sqft_high_coverage(self, spider, props):
        """Foxtons consistently includes sqft (~98%)."""
        items = [spider.parse_property(p, "Chelsea") for p in props]
        with_sqft = [i for i in items if i.get("size_sqft")]
        assert len(with_sqft) >= 1, "foxtons fixture yielded no sqft — JSON shape changed"

    def test_beds_int_or_none(self, spider, props):
        for prop in props:
            item = spider.parse_property(prop, "Chelsea")
            beds = item.get("bedrooms")
            assert beds is None or isinstance(beds, int)


# ---------------------------------------------------------------------------
# OpenRent — parse_property_card(card, listing_id, area) from the card-extraction JS
# ---------------------------------------------------------------------------
class TestOpenRentParsing:
    @pytest.fixture(scope="class")
    def spider(self):
        return OpenRentSpider()

    @pytest.fixture(scope="class")
    def cards(self):
        return load_fixture("openrent_cards.json")

    def test_required_fields_populated(self, spider, cards):
        for entry in cards:
            item = spider.parse_property_card(entry["card"], entry["listing_id"], "Chelsea")
            assert item["source"] == "openrent"
            assert item["property_id"] == entry["listing_id"]
            assert item["url"].startswith("https://www.openrent.co.uk/property-to-rent/")
            assert item["price_pcm"] and item["price_pcm"] > 0

    def test_studio_is_zero_beds(self, spider, cards):
        studio = next(c for c in cards if "Studio" in c["card"]["alt"])
        item = spider.parse_property_card(studio["card"], studio["listing_id"], "Chelsea")
        assert item["bedrooms"] == 0

    def test_weekly_price_converted_to_pcm(self, spider, cards):
        weekly = next(c for c in cards if "/week" in c["card"]["fullText"])
        item = spider.parse_property_card(weekly["card"], weekly["listing_id"], "Chelsea")
        assert item["price_period"] == "pw"
        assert item["price_pcm"] == int(item["price"] * 52 / 12)

    def test_postcode_district_extracted(self, spider, cards):
        for entry in cards:
            item = spider.parse_property_card(entry["card"], entry["listing_id"], "Chelsea")
            assert item["postcode"], f"no postcode from {entry['card']['alt']}"


# ---------------------------------------------------------------------------
# Selector-regression — frozen snapshots. If site JSON/card shape drifts so the
# parser yields different values for a KNOWN fixture, these fail loudly in CI.
# ---------------------------------------------------------------------------
@pytest.mark.selector
class TestSelectorRegression:
    def test_savills_card_exact_extraction(self):
        """Frozen: the 3 savills fixture cards must parse to these exact values."""
        spider = SavillsSpider()
        cards = load_fixture("savills_card_data.json")
        got = [
            (c_item["property_id"], c_item["price_pcm"], c_item["bedrooms"],
             c_item["postcode"], c_item["size_sqft"])
            for c_item in (spider.parse_card_data(c) for c in cards)
        ]
        expected = [
            ("gbssresll250139l", 4723, 2, "SW1X 8EA", 721),
            ("gbbyrebyl220056l", 2102, 1, "SW6 6AH", 565),
            ("gbwlrewll250021l", 3601, 4, "SW19 8JH", 1039),
        ]
        assert got == expected, (
            "Savills card parsing drifted — either the spider changed or the "
            "fixture was re-captured with a different shape. Review before updating."
        )

    def test_rightmove_property_shape_invariants(self):
        """The __NEXT_DATA__ search property must still carry the keys we parse."""
        props = load_fixture("rightmove_search_properties.json")
        required_keys = {"id", "propertyUrl", "price", "displayAddress",
                         "bedrooms", "location"}
        for prop in props:
            missing = required_keys - set(prop.keys())
            assert not missing, (
                f"Rightmove search property missing keys {missing} — site JSON "
                f"shape changed; rightmove_spider.parse_property will break."
            )
        # price sub-shape
        for prop in props:
            assert "amount" in prop["price"] and "frequency" in prop["price"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
