"""
for_sale.listing_parse — pure parse seams for SCRAPED FOR-SALE listings.

These are the for-sale analogues of the rental spiders' parse_property seams. They
REUSE the rental extraction patterns (the same postcode/sqft regexes, the same
id/url/address/beds/baths/type fields) and differ ONLY in:
  * which __NEXT_DATA__ price field holds the asking price, and
  * routing that price to the SALE schema (asking_price) instead of price_pcm.

They are PURE (dict in → SaleListingItem out), so they can be unit-tested against
committed for-sale fixtures with no network — exactly like tests/test_spider_parsing.py
tests the rental seams. A for-sale spider MODE (listing_type=sale) calls these from the
existing spiders' parse_search once the for-sale start URLs are wired; this module is
the reusable, independently-tested core of that mode.

No import of rental_price_models_v20 / canonical_predict (isolation; see the guard test).
"""
from __future__ import annotations

import re
from datetime import datetime

from for_sale.items import SaleListingItem, SALE_PRICE_FIELD

# Same postcode patterns the rental rightmove_spider uses (re-implemented here, not
# imported, to keep the for-sale vertical decoupled).
_FULL_POSTCODE = re.compile(r"([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})")
_OUTCODE = re.compile(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(?:,|$)")
_SQFT = re.compile(r"([\d,]+)\s*sq\.?\s*ft", re.I)


def extract_postcode(address: str | None) -> str | None:
    """Pull a UK postcode (full, else outcode) from an address. Mirrors the rental seam.
    Returns None when the address carries no postcode (common on Rightmove for-sale)."""
    if not address:
        return None
    up = address.upper()
    m = _FULL_POSTCODE.search(up)
    if m:
        return m.group(1).replace(" ", "")
    m = _OUTCODE.search(up)
    return m.group(1) if m else None


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def parse_rightmove_for_sale(prop: dict, area: str) -> SaleListingItem | None:
    """Parse one Rightmove FOR-SALE search property (from __NEXT_DATA__
    searchResults.properties). Same shape as the rental search property; the only
    difference is price.frequency == 'not specified' and the amount is a sale price.
    """
    prop_id = prop.get("id")
    if not prop_id:
        return None

    item = SaleListingItem()
    item["source"] = "rightmove"
    item["listing_type"] = "sale"
    item["property_id"] = str(prop_id)
    # propertyUrl looks like '/properties/89029677#/?channel=RES_BUY' — strip the fragment.
    url_path = (prop.get("propertyUrl") or "").split("#")[0]
    item["url"] = f"https://www.rightmove.co.uk{url_path}"
    item["area"] = area

    price_data = prop.get("price", {}) or {}
    item[SALE_PRICE_FIELD] = _to_int(price_data.get("amount")) or 0
    display_prices = price_data.get("displayPrices") or [{}]
    item["price_qualifier"] = display_prices[0].get("displayPriceQualifier", "") if display_prices else ""

    item["address"] = prop.get("displayAddress", "")
    item["postcode"] = extract_postcode(item["address"])
    location = prop.get("location", {}) or {}
    item["latitude"] = location.get("latitude")
    item["longitude"] = location.get("longitude")

    item["bedrooms"] = prop.get("bedrooms")
    item["bathrooms"] = prop.get("bathrooms")
    item["property_type"] = prop.get("propertySubType") or prop.get("propertyType") or ""

    size = prop.get("displaySize") or ""
    sm = _SQFT.search(size)
    item["size_sqft"] = int(sm.group(1).replace(",", "")) if sm else None

    status = (prop.get("displayStatus") or "").lower()
    item["is_under_offer"] = 1 if ("under offer" in status or "sold stc" in status or "sstc" in status) else 0
    item["is_new_build"] = 1 if (prop.get("propertyTypeFullDescription", "") or "").lower().startswith("new") else 0

    customer = prop.get("customer", {}) or {}
    item["agent_name"] = customer.get("branchDisplayName", "")
    item["summary"] = prop.get("summary", "")
    item["added_date"] = prop.get("addedOrReduced", "")
    item["scraped_at"] = datetime.utcnow().isoformat()
    item["is_active"] = 1
    return item


def parse_foxtons_for_sale(prop: dict, area: str) -> SaleListingItem | None:
    """Parse one Foxtons FOR-SALE search property (from __NEXT_DATA__
    pageProps.pageData.data.data). instructionType == 'sale'; the asking price is in
    priceFrom (pricePcm is junk/annualized for sales and must be ignored)."""
    prop_ref = prop.get("propertyReference") or prop.get("propertyId")
    if not prop_ref:
        return None

    item = SaleListingItem()
    item["source"] = "foxtons"
    item["listing_type"] = "sale"
    item["property_id"] = str(prop_ref)

    postcode_short = prop.get("postcodeShort") or ""
    slug = postcode_short.lower() if postcode_short else area.lower()
    item["url"] = f"https://www.foxtons.co.uk/properties-for-sale/{slug}/{str(prop_ref).lower()}"
    item["area"] = area

    # Sale price = priceFrom (== priceTo when single price). NEVER pricePcm.
    item[SALE_PRICE_FIELD] = _to_int(prop.get("priceFrom")) or 0
    item["price_qualifier"] = ""

    street = prop.get("streetName") or ""
    loc = prop.get("locationName") or ""
    item["address"] = ", ".join(x for x in (street, loc, postcode_short) if x)
    # Prefer the explicit district; else try to pull one from the assembled address.
    item["postcode"] = postcode_short or extract_postcode(item["address"])
    item["latitude"] = None
    item["longitude"] = None

    item["bedrooms"] = prop.get("bedrooms")
    item["bathrooms"] = prop.get("bathrooms")
    item["property_type"] = prop.get("searchPropertyType") or prop.get("typeGroup") or ""

    blob = prop.get("propertyBlob") or {}
    item["size_sqft"] = _to_int(blob.get("floorArea")) if blob.get("floorArea") else None

    item["is_new_build"] = 1 if prop.get("isNewHome") else 0
    item["is_under_offer"] = 1 if prop.get("isUnderOffer") else 0
    item["agent_name"] = prop.get("officeName", "")
    item["summary"] = ""
    item["scraped_at"] = datetime.utcnow().isoformat()
    item["is_active"] = 1
    return item
