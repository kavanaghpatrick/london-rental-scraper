"""
for_sale.items — the SCRAPED FOR-SALE listing schema, isolated from the rental stack.

A sale ASKING price (£100k–£40M+) is a different magnitude from a rental price_pcm
(£100–£500k/mo). Storing it in the rental PropertyItem.price_pcm would (a) be
semantically wrong and (b) trip the rental validator's "price_pcm suspiciously high
(>£500k)" guard. So the for-sale vertical gets its OWN item with a sale-named price
field and its OWN validator tuned to sale magnitudes.

This module REUSES the rental item's *shape* (Scrapy Item, the same beds/baths/sqft/
postcode field names so downstream feature-engineering can mirror the rental code) but
does NOT import or depend on the rental model chain.
"""
from __future__ import annotations

import scrapy

# The canonical sale-price field name. Sale-named (not pcm/pw) so it can never be
# confused with the rental price_pcm column. Exposed as a constant so tests, the
# parse seams, and a future pipeline all agree on one name.
SALE_PRICE_FIELD = "asking_price"


class SaleListingItem(scrapy.Item):
    """A scraped FOR-SALE listing. Mirrors PropertyItem field names where they carry
    the SAME meaning (beds/baths/sqft/postcode/type), but the price channel is the
    sale asking price — there is no price_pcm/price_pw on a sale item."""

    # Identity
    source = scrapy.Field()
    property_id = scrapy.Field()
    url = scrapy.Field()
    area = scrapy.Field()
    listing_type = scrapy.Field()  # always "sale" for this item (vs rental "rent")

    # Pricing — SALE asking price only (no pcm/pw).
    asking_price = scrapy.Field()          # int £, the headline asking/guide price
    price_qualifier = scrapy.Field()       # "Guide Price" / "Offers Over" / "" etc.

    # Location (same extraction as rentals)
    address = scrapy.Field()
    postcode = scrapy.Field()
    latitude = scrapy.Field()
    longitude = scrapy.Field()

    # Property details (same extraction as rentals)
    bedrooms = scrapy.Field()
    bathrooms = scrapy.Field()
    property_type = scrapy.Field()
    size_sqft = scrapy.Field()

    # Sale-specific status
    is_new_build = scrapy.Field()          # 1 if a new-home/development listing
    is_under_offer = scrapy.Field()        # 1 if SSTC / under offer

    # Agent / content
    agent_name = scrapy.Field()
    summary = scrapy.Field()

    # Bookkeeping (mirrors the rental historical-tracking columns)
    address_fingerprint = scrapy.Field()
    scraped_at = scrapy.Field()
    added_date = scrapy.Field()
    is_active = scrapy.Field()


# Sale-price sanity bounds (London prime-central residential asking prices).
# Floor guards against a monthly RENT (£~500–£20k) leaking into the sale table;
# ceiling is generous (trophy mansions reach tens of millions).
MIN_SALE_PRICE = 50_000
MAX_SALE_PRICE = 250_000_000


def validate_sale_item(item, logger=None) -> tuple[bool, list[str]]:
    """Validate a SaleListingItem. Returns (is_valid, issues).

    Mirrors rental items.validate_item but with SALE-magnitude price bounds: it
    ACCEPTS £875k–£40M (which the rental validator would reject as "price_pcm
    suspiciously high") and REJECTS rental-magnitude values as not-a-sale-price.
    """
    issues: list[str] = []
    data = dict(item) if hasattr(item, "items") else item

    if not data.get("source"):
        issues.append("missing source")
    if not data.get("property_id"):
        issues.append("missing property_id")

    price = data.get(SALE_PRICE_FIELD, 0)
    if not price or price <= 0:
        issues.append("missing/invalid asking_price")
    elif price < MIN_SALE_PRICE:
        issues.append(f"asking_price too low for a sale ({price}) — looks like a rent")
    elif price > MAX_SALE_PRICE:
        issues.append(f"asking_price implausibly high ({price})")

    url = data.get("url", "")
    if not url:
        issues.append("missing url")
    elif not url.startswith("http"):
        issues.append(f"invalid url format ({url[:30]}...)")

    beds = data.get("bedrooms")
    if beds is not None and (beds < 0 or beds > 30):
        issues.append(f"invalid bedrooms ({beds})")

    sqft = data.get("size_sqft")
    if sqft is not None and (sqft < 50 or sqft > 100_000):
        issues.append(f"implausible sqft ({sqft})")

    is_valid = len(issues) == 0
    if not is_valid and logger:
        logger.warning(f"[SALE-VALIDATION] {data.get('property_id', '?')}: {', '.join(issues)}")
    return is_valid, issues
