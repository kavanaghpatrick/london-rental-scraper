# property_scraper/utils/__init__.py
"""Utility modules for property scraping."""

from .floorplan_extractor import (
    FloorplanExtractor,
    FloorplanData,
    FloorData,
    Room,
    OutdoorSpace,
    extract_from_floorplan,
)

from .postcode import (
    extract_full_postcode,
    extract_postcode_district,
    extract_postcode_area,
    is_valid_postcode_district,
    normalize_postcode,
    is_prime_london,
    PRIME_LONDON_DISTRICTS,
)

from .url_validation import (
    validate_url,
    validate_image_url,
    validate_floorplan_url,
)

__all__ = [
    # Floorplan extraction
    'FloorplanExtractor',
    'FloorplanData',
    'FloorData',
    'Room',
    'OutdoorSpace',
    'extract_from_floorplan',
    # Postcode utilities (Issue #22)
    'extract_full_postcode',
    'extract_postcode_district',
    'extract_postcode_area',
    'is_valid_postcode_district',
    'normalize_postcode',
    'is_prime_london',
    'PRIME_LONDON_DISTRICTS',
    # URL validation (Issue #25)
    'validate_url',
    'validate_image_url',
    'validate_floorplan_url',
]
