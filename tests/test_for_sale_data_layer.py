"""test_for_sale_data_layer.py — TDD contract for the SEPARATE "FOR SALE" vertical
(increment #1: scraped for-sale ASKING-listing data layer).

ARCHITECTURE (per the project directive — read this before changing the test)
----------------------------------------------------------------------------
The for-sale vertical REUSES THE EXISTING SCRAPING INFRASTRUCTURE pointed at the
realtor sites' FOR-SALE sections (e.g. rightmove /property-for-sale/) instead of the
rental sections. It is NOT a Land Registry / sold-price project — the data is the
SCRAPED FOR-SALE ASKING price from the SAME spiders, in a for-sale MODE behind a
`listing_type=sale` flag that leaves the default RENTAL behaviour untouched. (Land
Registry is a possible FUTURE enrichment/validation, out of scope here.)

This increment proves the FOUNDATION the user asked for:
  1. The spiders can extract for-sale listings (Rightmove `__NEXT_DATA__` seam) into a
     for-sale item whose key field is the ASKING SALE PRICE — a lump sum (£100k–£50M),
     a DIFFERENT MAGNITUDE from rental price_pcm, kept in a DISTINCT field
     (`asking_price`), never `price_pcm`.
  2. A for-sale DATA LAYER (for_sale/sale_data.py) — schema + a separate `sale_listings`
     table — ISOLATED from the rental `listings` table, so a sale price can never leak
     into the rent magnitude and vice-versa.

THE FIXTURE is REAL captured data: tests/fixtures/for_sale/rightmove_for_sale_properties.json
is the actual `props.pageProps.searchResults.properties` array from a live Rightmove
/property-for-sale/ Chelsea search (6 prime-London sale listings). No network in CI —
the parse seam runs against this committed sample, exactly like test_spider_parsing.py
does for the rental seam.

CI-SAFETY: marker `for_sale` (registered under strict_markers); no network, no live DB,
no binary deps. Runs on every PR and is on the anti-silent-skip allowlist.

ZERO RENTAL REGRESSION: nothing here imports or mutates rental_price_models_v20,
canonical_predict, the parity-gated xgboost.js, or the rental `listings` table. The
for-sale mode is additive (a flag); the default rental parse path is asserted unchanged
(test_default_spider_is_rental_unchanged) and the existing tests/test_spider_parsing.py
is untouched.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

pytestmark = pytest.mark.for_sale

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "for_sale" / "rightmove_for_sale_properties.json"


# ── The committed for-sale fixture (REAL __NEXT_DATA__ sale-property shape) ───────

def test_committed_for_sale_fixture_exists_and_nonempty():
    assert FIXTURE.exists(), (
        "committed Rightmove for-sale fixture missing — it is the CI-safe sample the "
        "for-sale parse seam is exercised against (no live network in CI)."
    )
    props = json.loads(FIXTURE.read_text())
    assert isinstance(props, list) and len(props) >= 5


def test_fixture_has_sale_price_shape_not_rental():
    """A for-sale __NEXT_DATA__ property carries a lump-sum price.amount with a BUY
    channel and a sale qualifier (Guide Price / blank) — NOT a weekly/monthly rental
    frequency. This pins the shape the parser must handle."""
    props = json.loads(FIXTURE.read_text())
    for p in props:
        assert p.get("channel") == "BUY"
        price = p["price"]
        assert "amount" in price
        # rental fixtures use frequency monthly/weekly; sale does not.
        assert price.get("frequency", "") not in ("monthly", "weekly")
    # The common London sale qualifier must be present in the sample.
    quals = {
        dp.get("displayPriceQualifier", "")
        for p in props
        for dp in p["price"].get("displayPrices", [])
    }
    assert "Guide Price" in quals


# ── The for-sale parse seam on the Rightmove spider (listing_type=sale mode) ──────

@pytest.fixture(scope="module")
def sale_spider():
    """The SAME RightmoveSpider, constructed in for-sale mode via listing_type=sale.
    Constructor runs standalone (no crawl). Default (no flag) must stay rental."""
    from property_scraper.spiders.rightmove_spider import RightmoveSpider
    return RightmoveSpider(
        areas="Chelsea", max_pages="1", fetch_details="false", listing_type="sale"
    )


@pytest.fixture(scope="module")
def rent_spider():
    from property_scraper.spiders.rightmove_spider import RightmoveSpider
    return RightmoveSpider(areas="Chelsea", max_pages="1", fetch_details="false")


@pytest.fixture(scope="module")
def props():
    return json.loads(FIXTURE.read_text())


def test_default_spider_is_rental_unchanged(rent_spider):
    """ZERO-REGRESSION GUARD: with no listing_type flag the spider stays RENT — same
    rental URL section, so existing rental scrapes/behaviour are untouched."""
    assert getattr(rent_spider, "listing_type", "rent") == "rent"
    assert "property-to-rent" in rent_spider.start_url_for("Chelsea")
    assert "property-for-sale" not in rent_spider.start_url_for("Chelsea")


def test_sale_mode_uses_for_sale_url_section(sale_spider):
    """For-sale mode points the SAME spider at the site's /property-for-sale/ section —
    reusing the infra, just a different part of the realtor site."""
    assert sale_spider.listing_type == "sale"
    url = sale_spider.start_url_for("Chelsea")
    assert "property-for-sale" in url
    assert "property-to-rent" not in url


def test_sale_parse_extracts_asking_price_not_pcm(sale_spider, props):
    """The for-sale parse seam yields the ASKING SALE PRICE in a DISTINCT field and
    NEVER populates the rental price_pcm magnitude (isolation at the item level)."""
    parsed = [sale_spider.parse_for_sale_property(p, "Chelsea") for p in props]
    parsed = [it for it in parsed if it is not None]
    assert len(parsed) == len(props)
    for it in parsed:
        assert it["source"] == "rightmove"
        assert it["listing_type"] == "sale"
        assert it["property_id"]
        assert it["url"].startswith("https://www.rightmove.co.uk/properties/")
        # No rental magnitude leaked.
        assert not it.get("price_pcm")
        assert not it.get("price_pw")
    # Every priced row carries a lump-sum asking price in a sane London sale range.
    for it in parsed:
        assert it.get("asking_price")
        assert 100_000 < it["asking_price"] < 50_000_000


def test_sale_parse_keeps_shared_attributes(sale_spider, props):
    """The SAME extracted attributes the rental item has (beds/baths/sqft/property_type/
    coords) carry across to the sale item — only the price field differs. Asserted
    against the REAL Lower Sloane Street listing (id 89029677)."""
    by_id = {str(p["id"]): sale_spider.parse_for_sale_property(p, "Chelsea") for p in props}
    sloane = by_id["89029677"]
    assert sloane["bedrooms"] == 1
    assert sloane["bathrooms"] == 1
    assert sloane["size_sqft"] == 466          # parsed from "466 sq. ft." (periods)
    assert sloane["property_type"] == "Flat"   # propertySubType preferred
    assert sloane["latitude"] and sloane["longitude"]
    assert sloane["asking_price"] == 875000


def test_sale_parse_sqft_handles_sq_ft_with_periods(sale_spider, props):
    """Rightmove for-sale displaySize is "11,316 sq. ft." (commas + periods) — the
    parser must strip both. Asserted on the Hans Place listing (id 88781454)."""
    by_id = {str(p["id"]): sale_spider.parse_for_sale_property(p, "Chelsea") for p in props}
    assert by_id["88781454"]["size_sqft"] == 11316


def test_sale_parse_extracts_outcode_when_full_postcode_absent(sale_spider, props):
    """Most for-sale addresses lack a full postcode; the parser must still pull the
    outcode when present (Manresa Road ... SW3) and tolerate its absence (Lower Sloane
    Street has none) without crashing."""
    by_id = {str(p["id"]): sale_spider.parse_for_sale_property(p, "Chelsea") for p in props}
    assert by_id["87951738"]["postcode"] == "SW3"     # outcode at end of address
    assert by_id["89029677"]["postcode"] in (None, "")  # no postcode in address


def test_sale_parse_captures_price_qualifier(sale_spider, props):
    """Sale listings carry a human qualifier (Guide Price / blank) the buyer-side tool
    needs to reason about 'over/under asking'. The parser keeps it verbatim; it is
    sale-specific and has no rental analogue."""
    by_id = {str(p["id"]): sale_spider.parse_for_sale_property(p, "Chelsea") for p in props}
    assert by_id["89029677"]["price_qualifier"] == "Guide Price"
    assert by_id["87951738"]["price_qualifier"] == ""  # blank qualifier preserved


def test_sale_parse_handles_poa_without_crash(sale_spider):
    """A POA (price-on-application, amount 0) row must parse without crashing and
    surface asking_price as None — not 0 and not a rental price — so downstream code
    can skip unpriced comps cleanly (the single-row class of bug the rental side hit).
    The captured fixture has no POA row, so this exercises a synthetic POA property of
    the same shape."""
    poa = {
        "id": 99999999,
        "propertyUrl": "/properties/99999999#/?channel=RES_BUY",
        "price": {"amount": 0, "frequency": "not specified",
                  "displayPrices": [{"displayPrice": "POA", "displayPriceQualifier": "POA"}]},
        "displayAddress": "Eaton Square, Belgravia, London, SW1W",
        "bedrooms": 4, "bathrooms": 3, "propertySubType": "Penthouse",
        "displaySize": "", "location": {"latitude": 51.49, "longitude": -0.15},
        "channel": "BUY",
    }
    it = sale_spider.parse_for_sale_property(poa, "Chelsea")
    assert it is not None
    assert it["asking_price"] is None
    assert it["price_qualifier"] == "POA"
    assert not it.get("price_pcm")


def test_sale_parse_missing_id_returns_none(sale_spider):
    assert sale_spider.parse_for_sale_property({"price": {"amount": 1000000}}, "Chelsea") is None


# ── The isolated for-sale DATA LAYER (for_sale/sale_data.py) ─────────────────────

@pytest.fixture(scope="module")
def sale_data():
    from for_sale import sale_data
    return sale_data


def test_sale_schema_is_separate_table_with_asking_price(sale_data):
    """The for-sale data layer creates its OWN `sale_listings` table with an
    `asking_price` column — NOT the rental `listings` table and NOT a `price_pcm`
    column — so the two magnitudes are physically isolated."""
    conn = sqlite3.connect(":memory:")
    try:
        sale_data.create_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        assert "sale_listings" in tables
        assert "listings" not in tables, "for-sale layer must not create the rental table"
        cur.execute("PRAGMA table_info(sale_listings)")
        cols = {r[1] for r in cur.fetchall()}
        assert "asking_price" in cols
        assert "price_pcm" not in cols, "rental rent magnitude must not exist in sale table"
        # Shared attributes that DO carry across.
        for shared in ("source", "property_id", "postcode", "bedrooms", "size_sqft",
                       "property_type", "price_qualifier", "address_fingerprint"):
            assert shared in cols
    finally:
        conn.close()


def test_insert_and_read_back_sale_listing(sale_data, sale_spider, props):
    """A parsed for-sale item round-trips through the data layer and reads back with the
    asking price intact and in sale magnitude — exercised on a REAL fixture row."""
    item = sale_spider.parse_for_sale_property(props[0], "Chelsea")  # Lower Sloane St
    conn = sqlite3.connect(":memory:")
    try:
        sale_data.create_schema(conn)
        sale_data.upsert_sale_listing(conn, item)
        rows = sale_data.fetch_sale_listings(conn)
        assert len(rows) == 1
        r = rows[0]
        assert r["asking_price"] == 875000
        assert r["bedrooms"] == 1
        assert r["size_sqft"] == 466
        # fingerprint derived by the data layer (reuses the pattern, see isolation test).
        assert r["address_fingerprint"]
    finally:
        conn.close()


def test_upsert_is_idempotent_on_source_property_id(sale_data):
    """Re-ingesting the same (source, property_id) updates in place — no duplicate row,
    mirroring the rental pipeline's UNIQUE(source, property_id) identity."""
    conn = sqlite3.connect(":memory:")
    try:
        sale_data.create_schema(conn)
        base = {
            "source": "rightmove", "property_id": "X1",
            "url": "https://www.rightmove.co.uk/properties/X1",
            "asking_price": 1000000, "postcode": "SW3", "bedrooms": 2,
            "address": "1 Test Street, Chelsea, London, SW3",
        }
        sale_data.upsert_sale_listing(conn, base)
        sale_data.upsert_sale_listing(conn, {**base, "asking_price": 950000})
        rows = sale_data.fetch_sale_listings(conn)
        assert len(rows) == 1
        assert rows[0]["asking_price"] == 950000  # updated, not duplicated
    finally:
        conn.close()


def test_sale_status_flags_round_trip(sale_data, sale_spider):
    """The for-sale parse seam extracts SALE-SPECIFIC status the rent vertical has no
    analogue for — is_under_offer (SSTC comps must be excludable) and is_new_build (a
    known price-magnitude outlier the sale model will treat specially in Inc3). Those
    flags must SURVIVE persistence, not be silently dropped by the data layer. This
    pins the schema/parser column-set agreement (the cross-module drift the rental
    dedupe incident warned about)."""
    poa = {
        "id": 55555555,
        "propertyUrl": "/properties/55555555#/?channel=RES_BUY",
        "price": {"amount": 2_500_000, "frequency": "not specified",
                  "displayPrices": [{"displayPriceQualifier": "Guide Price"}]},
        "displayAddress": "1 New Wharf, Chelsea, London, SW10",
        "bedrooms": 3, "bathrooms": 2, "propertySubType": "Flat",
        "propertyTypeFullDescription": "New home Flat",
        "displaySize": "1,200 sq. ft.", "location": {"latitude": 51.48, "longitude": -0.18},
        "displayStatus": "Under offer", "channel": "BUY",
    }
    item = sale_spider.parse_for_sale_property(poa, "Chelsea")
    assert item["is_under_offer"] == 1
    assert item["is_new_build"] == 1
    conn = sqlite3.connect(":memory:")
    try:
        sale_data.create_schema(conn)
        # The schema must expose these sale-status columns.
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(sale_listings)")
        cols = {r[1] for r in cur.fetchall()}
        assert "is_under_offer" in cols
        assert "is_new_build" in cols
        sale_data.upsert_sale_listing(conn, item)
        r = sale_data.fetch_sale_listings(conn)[0]
        assert r["is_under_offer"] == 1
        assert r["is_new_build"] == 1
    finally:
        conn.close()


def test_data_layer_never_stores_rental_magnitude(sale_data):
    """Belt-and-braces isolation: even if a caller passes a stray price_pcm key, the
    sale layer must not store it — the sale table has no rent column and the upsert
    drops unknown keys."""
    conn = sqlite3.connect(":memory:")
    try:
        sale_data.create_schema(conn)
        sale_data.upsert_sale_listing(conn, {
            "source": "rightmove", "property_id": "Y1",
            "url": "https://www.rightmove.co.uk/properties/Y1",
            "asking_price": 800000, "postcode": "E14", "bedrooms": 1,
            "address": "2 Wharf Road, Poplar, London, E14",
            "price_pcm": 3500,  # rental field — must be ignored, not stored
        })
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(sale_listings)")
        cols = {r[1] for r in cur.fetchall()}
        assert "price_pcm" not in cols
        # And the row is stored at sale magnitude only.
        rows = sale_data.fetch_sale_listings(conn)
        assert rows[0]["asking_price"] == 800000
    finally:
        conn.close()


# ── ISOLATION GUARD — the for-sale vertical must not touch the rental stack ───────

def test_for_sale_module_does_not_import_rental_model():
    """Hard isolation: for_sale/* must not import the parity-gated rental chain
    (rental_price_models_v20 / canonical_predict). It may RE-IMPLEMENT shared patterns
    (a postcode regex, the fingerprint primitive) but must not couple to it, so a rental
    retrain can never perturb for-sale and vice-versa."""
    import for_sale.sale_data as m
    src = Path(m.__file__).read_text()
    for banned in (
        "import rental_price_models_v20",
        "from rental_price_models_v20",
        "import canonical_predict",
        "from canonical_predict",
    ):
        assert banned not in src, f"for_sale.sale_data illegally couples to rental: {banned}"


def test_sale_data_layer_does_not_write_rental_listings_table(sale_data):
    """The for-sale data layer must never CREATE or write the rental `listings` table —
    physical guarantee the two verticals share no table."""
    conn = sqlite3.connect(":memory:")
    try:
        sale_data.create_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='listings'")
        assert cur.fetchone() is None
    finally:
        conn.close()
