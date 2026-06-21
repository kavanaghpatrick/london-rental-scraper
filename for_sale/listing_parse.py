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

import hashlib
import re
from datetime import datetime

from for_sale.items import SaleListingItem, SALE_PRICE_FIELD

# Same postcode patterns the rental rightmove_spider uses (re-implemented here, not
# imported, to keep the for-sale vertical decoupled).
_FULL_POSTCODE = re.compile(r"([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})")
_OUTCODE = re.compile(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(?:,|$)")
_SQFT = re.compile(r"([\d,]+)\s*sq\.?\s*ft", re.I)
# Sale cards carry a headline lump-sum £ with NO Monthly/Weekly token (unlike rentals).
_SALE_PRICE = re.compile(r"£\s*([\d,]+)")


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


# Price-qualifier phrases, in match priority. "Offers in Excess of" / "Offers in the
# Region of" must be checked before bare "Offers Over" so the longer phrase wins.
_SAVILLS_QUALIFIERS = (
    "Guide Price",
    "Offers in Excess of",
    "Offers in the Region of",
    "Offers Over",
)
_KF_QUALIFIERS = ("Guide Price", "Offers in Excess of", "Offers Over")
_CHEST_QUALIFIERS = ("Guide Price", "Offers in Excess of", "Offers Over")


def _price_qualifier(text: str, phrases) -> str:
    """Return the first qualifier phrase present in `text` (case-insensitive), else ''."""
    low = (text or "").lower()
    for phrase in phrases:
        if phrase.lower() in low:
            return phrase
    return ""


def parse_savills_for_sale(card_data: dict, area: str) -> SaleListingItem | None:
    """Parse one Savills FOR-SALE card (the dict savills_spider's page.evaluate() emits:
    {href, text, address, sqft, price, beds, baths, postcode, furnished}). In sale mode
    `price` is the raw lump-sum int (NOT ×52/12). REUSES the rental extraction patterns;
    routes the headline £ into asking_price (never price_pcm/price_pw)."""
    href = card_data.get("href", "") or ""
    text = card_data.get("text", "") or ""

    if not href and not text:
        return None

    item = SaleListingItem()
    item["source"] = "savills"
    item["listing_type"] = "sale"

    id_match = re.search(r"/property-detail/([^/?#]+)", href)
    if id_match:
        item["property_id"] = id_match.group(1)
    else:
        item["property_id"] = f"savills_{hashlib.sha256(text.encode()).hexdigest()[:16]}"

    item["url"] = href

    # Asking price: explicit card price first; else the £ in the card text. 0 = POA.
    price = _to_int(card_data.get("price"))
    if not price:
        pm = _SALE_PRICE.search(text)
        price = _to_int(pm.group(1).replace(",", "")) if pm else None
    item[SALE_PRICE_FIELD] = price or 0

    item["price_qualifier"] = _price_qualifier(text, _SAVILLS_QUALIFIERS)

    address = card_data.get("address") or ""
    if not address:
        for line in text.split("\n"):
            if re.search(r"[A-Z]{1,2}\d", line):
                address = line.strip()
                break
    item["address"] = address

    item["postcode"] = card_data.get("postcode") or extract_postcode(address)
    item["latitude"] = None
    item["longitude"] = None

    beds = _to_int(card_data.get("beds"))
    if beds is None:
        bm = re.search(r"(\d+)\s*Bedrooms?", text)
        beds = _to_int(bm.group(1)) if bm else None
    item["bedrooms"] = beds

    baths = _to_int(card_data.get("baths"))
    if baths is None:
        bm = re.search(r"(\d+)\s*Bathrooms?", text)
        baths = _to_int(bm.group(1)) if bm else None
    item["bathrooms"] = baths

    item["property_type"] = _infer_type(text)

    size = _to_int(card_data.get("sqft"))
    if size is None:
        sm = _SQFT.search(text)
        size = _to_int(sm.group(1).replace(",", "")) if sm else None
    item["size_sqft"] = size

    low = text.lower()
    item["is_under_offer"] = 1 if any(t in low for t in ("under offer", "sold stc", "sstc")) else 0
    item["is_new_build"] = 1 if re.search(r"\bnew\b", low) else 0

    item["agent_name"] = "Savills"
    item["summary"] = text[:500]
    item["scraped_at"] = datetime.utcnow().isoformat()
    item["is_active"] = 1
    return item


def parse_knightfrank_for_sale(card_data: dict, area: str) -> SaleListingItem | None:
    """Parse one Knight Frank FOR-SALE card (the dict knightfrank_spider's
    page.evaluate() emits: {text, href}). Everything is regex-parsed from `text`.
    Routes the headline £ into asking_price (never price_pcm/price_pw)."""
    text = card_data.get("text", "") or ""
    href = card_data.get("href", "") or ""

    if not text and not href:
        return None

    item = SaleListingItem()
    item["source"] = "knightfrank"
    item["listing_type"] = "sale"

    id_match = re.search(r"/([a-z]{3}\d+)$", href, re.I)
    if id_match:
        item["property_id"] = id_match.group(1)
    else:
        item["property_id"] = f"kf_{hashlib.sha256(text.encode()).hexdigest()[:16]}"

    item["url"] = href

    pm = _SALE_PRICE.search(text)
    item[SALE_PRICE_FIELD] = (_to_int(pm.group(1).replace(",", "")) if pm else None) or 0

    item["price_qualifier"] = _price_qualifier(text, _KF_QUALIFIERS)

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Address = first line carrying a postcode-like token.
    address = ""
    address_line_idx = 0
    for i, line in enumerate(lines):
        if re.search(r"[A-Z]{1,2}\d{1,2}", line):
            address = line
            address_line_idx = i
            break
    item["address"] = address

    pc_match = re.search(r"([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d?[A-Z]{0,2}$", address.upper())
    item["postcode"] = pc_match.group(1) if pc_match else extract_postcode(address)
    item["latitude"] = None
    item["longitude"] = None

    # Beds/baths from the positional single-digit sequence AFTER the address line.
    post_address_text = "\n".join(lines[address_line_idx + 1:])
    nums = re.findall(r"\b(\d)\b", post_address_text)
    if len(nums) >= 3:
        item["bedrooms"] = _to_int(nums[-3])
        item["bathrooms"] = _to_int(nums[-2])
    else:
        item["bedrooms"] = None
        item["bathrooms"] = None

    prop_type = ""
    for line in lines:
        if line.lower() in ("flat", "house", "apartment", "maisonette", "studio"):
            prop_type = line.lower()
            break
    item["property_type"] = prop_type

    sm = re.search(r"([\d,]+)\s*sqft", text, re.I)
    item["size_sqft"] = _to_int(sm.group(1).replace(",", "")) if sm else None

    low = text.lower()
    item["is_under_offer"] = 1 if any(t in low for t in ("under offer", "sstc", "sold stc", "agreed")) else 0
    item["is_new_build"] = 1 if "new" in low else 0

    item["agent_name"] = "Knight Frank"
    item["summary"] = ""
    item["scraped_at"] = datetime.utcnow().isoformat()
    item["is_active"] = 1
    return item


def parse_chestertons_for_sale(card_data: dict, area: str) -> SaleListingItem | None:
    """Parse one Chestertons FOR-SALE card (the dict chestertons_spider's
    page.evaluate() emits: {href, address, letType, textContent}). The /sales/ id seam
    accepts BOTH /sales/ and /lettings/ (rental regex was /lettings/ only). Routes the
    headline £ into asking_price (never price_pcm/price_pw)."""
    href = card_data.get("href", "") or ""
    address = card_data.get("address", "") or ""
    let_type = card_data.get("letType", "") or ""
    text = card_data.get("textContent", "") or ""

    if not href and not text:
        return None

    item = SaleListingItem()
    item["source"] = "chestertons"
    item["listing_type"] = "sale"

    # MANDATORY /sales/ FIX: accept /sales/ as well as /lettings/ so the sale URL yields
    # the STABLE "<num>_<REF>" id (not a content hash).
    id_match = re.search(r"/properties/(\d+)/(?:sales|lettings)/(\w+)", href)
    if id_match:
        item["property_id"] = f"{id_match.group(1)}_{id_match.group(2)}"
    else:
        item["property_id"] = f"chestertons_{hashlib.sha256(text.encode()).hexdigest()[:16]}"

    if href:
        item["url"] = f"https://www.chestertons.co.uk{href}" if href.startswith("/") else href
    else:
        item["url"] = ""

    pm = _SALE_PRICE.search(text)
    item[SALE_PRICE_FIELD] = (_to_int(pm.group(1).replace(",", "")) if pm else None) or 0

    item["price_qualifier"] = _price_qualifier(text, _CHEST_QUALIFIERS)

    item["address"] = address

    pc_match = re.search(r"([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d?[A-Z]{0,2}$", address.upper())
    item["postcode"] = pc_match.group(1) if pc_match else extract_postcode(address)
    item["latitude"] = None
    item["longitude"] = None

    # Beds/baths positional: first textContent line with comma+alpha is the address line.
    lines = text.strip().split("\n")
    address_line_idx = 0
    for i, line in enumerate(lines):
        if "," in line and any(c.isalpha() for c in line):
            address_line_idx = i
            break
    post_address_text = "\n".join(lines[address_line_idx + 1:])
    nums = re.findall(r"\b(\d)\b", post_address_text)
    item["bedrooms"] = _to_int(nums[0]) if len(nums) >= 1 else None
    item["bathrooms"] = _to_int(nums[1]) if len(nums) >= 2 else None

    item["property_type"] = _infer_type(text)

    sm = re.search(r"(\d{3,5})\s*ft", text)
    item["size_sqft"] = _to_int(sm.group(1)) if sm else None

    low = (let_type + " " + text).lower()
    item["is_under_offer"] = 1 if any(t in low for t in ("under offer", "sold stc", "sstc")) else 0
    item["is_new_build"] = 1 if "new" in low else 0

    item["agent_name"] = "Chestertons"
    item["summary"] = ""
    item["scraped_at"] = datetime.utcnow().isoformat()
    item["is_active"] = 1
    return item


def _infer_type(text: str) -> str:
    """Keyword-infer a property_type from card text (apartment/flat/house/studio/...)."""
    low = (text or "").lower()
    if "penthouse" in low:
        return "penthouse"
    if "maisonette" in low:
        return "maisonette"
    if "studio" in low:
        return "studio"
    if "apartment" in low or "flat" in low:
        return "flat"
    if "house" in low:
        return "house"
    return ""
