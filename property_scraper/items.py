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
