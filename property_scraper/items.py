"""
Scrapy Item definitions for property listings.
"""

import scrapy
from itemloaders.processors import TakeFirst, MapCompose, Join
from w3lib.html import remove_tags


def clean_price(value):
    """Extract numeric price from text."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        cleaned = value.replace('£', '').replace(',', '').strip()
        try:
            return int(float(cleaned))
        except (ValueError, TypeError):
            return None
    return None


def clean_text(value):
    """Clean whitespace from text."""
    if value:
        return ' '.join(value.split())
    return value


class PropertyItem(scrapy.Item):
    """Item for a rental property listing."""

    # Identity
    source = scrapy.Field()
    property_id = scrapy.Field()
    url = scrapy.Field()
    area = scrapy.Field()

    # Pricing
    price = scrapy.Field()
    price_pw = scrapy.Field()
    price_pcm = scrapy.Field()
    price_period = scrapy.Field()

    # Location
    address = scrapy.Field()
    postcode = scrapy.Field()
    latitude = scrapy.Field()
    longitude = scrapy.Field()

    # Property details
    bedrooms = scrapy.Field()
    bathrooms = scrapy.Field()
    reception_rooms = scrapy.Field()  # Living/reception room count
    property_type = scrapy.Field()
    size_sqft = scrapy.Field()
    size_sqm = scrapy.Field()  # Square meters (from floorplans)
    furnished = scrapy.Field()
    epc_rating = scrapy.Field()  # Energy rating A-G

    # Floorplan
    floorplan_url = scrapy.Field()  # Direct URL to floorplan image
    room_details = scrapy.Field()  # JSON: room dimensions from floorplan

    # Binary floor data (for ML model training)
    has_basement = scrapy.Field()       # 1 if has basement
    has_lower_ground = scrapy.Field()   # 1 if has lower ground floor
    has_ground = scrapy.Field()         # 1 if has ground floor
    has_mezzanine = scrapy.Field()      # 1 if has mezzanine
    has_first_floor = scrapy.Field()    # 1 if has first floor
    has_second_floor = scrapy.Field()   # 1 if has second floor
    has_third_floor = scrapy.Field()    # 1 if has third floor
    has_fourth_plus = scrapy.Field()    # 1 if has 4th floor or higher
    has_roof_terrace = scrapy.Field()   # 1 if has roof/roof terrace
    floor_count = scrapy.Field()        # Total number of floors
    property_levels = scrapy.Field()    # single_floor, duplex, triplex, multi_floor

    # Status
    let_agreed = scrapy.Field()

    # Agent
    agent_name = scrapy.Field()
    agent_phone = scrapy.Field()

    # Content
    summary = scrapy.Field()
    description = scrapy.Field()
    features = scrapy.Field()

    # Dates
    added_date = scrapy.Field()
    scraped_at = scrapy.Field()

    # Historical tracking (PRD-003)
    address_fingerprint = scrapy.Field()  # SHA256 hash for cross-source dedup
    first_seen = scrapy.Field()           # ISO timestamp of first scrape
    last_seen = scrapy.Field()            # ISO timestamp of most recent scrape
    is_active = scrapy.Field()            # 1 if still available, 0 if delisted
    price_change_count = scrapy.Field()   # Number of price changes detected


# =============================================================================
# Issue #20 FIX: Item Validation
# =============================================================================

def validate_item(item, logger=None) -> tuple[bool, list[str]]:
    """Validate a PropertyItem before yielding.

    Args:
        item: PropertyItem or dict to validate
        logger: Optional logger for warnings

    Returns:
        (is_valid, list_of_issues) tuple

    Example:
        >>> is_valid, issues = validate_item(item, self.logger)
        >>> if not is_valid:
        ...     self.logger.warning(f"Skipping invalid item: {issues}")
        ...     return None
        >>> yield item
    """
    issues = []

    # Convert to dict if needed
    data = dict(item) if hasattr(item, 'items') else item

    # Required fields
    if not data.get('source'):
        issues.append('missing source')
    if not data.get('property_id'):
        issues.append('missing property_id')

    # Price validation
    price_pcm = data.get('price_pcm', 0)
    if not price_pcm or price_pcm <= 0:
        issues.append('missing/invalid price_pcm')
    elif price_pcm < 100:
        issues.append(f'price_pcm too low ({price_pcm})')
    elif price_pcm > 500000:
        issues.append(f'price_pcm suspiciously high ({price_pcm})')

    # Address validation
    address = data.get('address', '')
    if not address or len(address) < 5:
        issues.append('missing/invalid address')

    # URL validation
    url = data.get('url', '')
    if not url:
        issues.append('missing url')
    elif not url.startswith('http'):
        issues.append(f'invalid url format ({url[:30]}...)')

    # Bedrooms sanity check
    bedrooms = data.get('bedrooms')
    if bedrooms is not None:
        if bedrooms < 0 or bedrooms > 20:
            issues.append(f'invalid bedrooms ({bedrooms})')

    # Sqft sanity check
    sqft = data.get('size_sqft')
    if sqft is not None:
        if sqft < 50:
            issues.append(f'sqft too small ({sqft})')
        elif sqft > 50000:
            issues.append(f'sqft too large ({sqft})')

    is_valid = len(issues) == 0

    if not is_valid and logger:
        prop_id = data.get('property_id', 'unknown')
        logger.warning(f"[VALIDATION] {prop_id}: {', '.join(issues)}")

    return is_valid, issues


def validate_and_yield(item, logger=None, strict=False):
    """Validate item and return it if valid, None otherwise.

    Args:
        item: PropertyItem to validate
        logger: Optional logger for warnings
        strict: If True, return None for any validation issue.
                If False (default), only return None for critical issues.

    Returns:
        item if valid, None otherwise

    Example:
        >>> yield validate_and_yield(item, self.logger)  # Yields None if invalid
    """
    is_valid, issues = validate_item(item, logger)

    if strict:
        return item if is_valid else None

    # Non-strict: only reject items missing critical fields
    critical_issues = [i for i in issues if any(x in i for x in
                       ['missing source', 'missing property_id', 'missing url'])]

    if critical_issues:
        return None

    return item
