"""
Centralized UK postcode extraction utilities.

Issue #22 FIX: Provides consistent postcode extraction across all spiders.

UK Postcode Format:
- Outward code (district): 2-4 chars (e.g., SW1, W1A, EC1A)
- Inward code: 3 chars (e.g., 1AA)
- Full format: SW1A 1AA

This module provides utilities to extract:
1. Full postcodes (SW1A 1AA)
2. Postcode districts (SW1A, SW1, W1)
3. Postcode areas (SW, W, EC)
"""

import re
from typing import Optional

# UK postcode patterns
# Full postcode: SW1A 1AA, W1A 1AA, EC1A 1BB
FULL_POSTCODE_PATTERN = re.compile(
    r'([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d[A-Z]{2})',
    re.IGNORECASE
)

# Outward code only (district): SW1A, SW1, W1, EC1A
DISTRICT_PATTERN = re.compile(
    r'([A-Z]{1,2}\d{1,2}[A-Z]?)',
    re.IGNORECASE
)

# Area only: SW, W, EC, NW
AREA_PATTERN = re.compile(
    r'^([A-Z]{1,2})',
    re.IGNORECASE
)


def extract_full_postcode(text: str) -> Optional[str]:
    """Extract full UK postcode from text.

    Args:
        text: Text containing a postcode (e.g., address)

    Returns:
        Full postcode in uppercase with space (e.g., "SW1A 1AA") or None

    Example:
        >>> extract_full_postcode("123 King's Road, London SW3 4PL")
        'SW3 4PL'
    """
    if not text:
        return None
    match = FULL_POSTCODE_PATTERN.search(text.upper())
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None


def extract_postcode_district(text: str) -> Optional[str]:
    """Extract postcode district (outward code) from text.

    The district is the first part of a postcode (e.g., SW3 from SW3 4PL).
    This is the most commonly used for grouping properties by area.

    Args:
        text: Text containing a postcode

    Returns:
        Postcode district in uppercase (e.g., "SW3") or None

    Example:
        >>> extract_postcode_district("Flat 2, 100 King's Road, Chelsea SW3 4PL")
        'SW3'
    """
    if not text:
        return None

    # First try to find a full postcode and extract district
    full_match = FULL_POSTCODE_PATTERN.search(text.upper())
    if full_match:
        return full_match.group(1)

    # Fall back to just finding a district pattern
    # Search from end of string first (postcodes usually at end of addresses)
    text_upper = text.upper()

    # Try to find district at end of string (most reliable for addresses)
    end_match = re.search(r'([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d?[A-Z]{0,2}\s*$', text_upper)
    if end_match:
        return end_match.group(1)

    # Fall back to any district pattern
    match = DISTRICT_PATTERN.search(text_upper)
    if match:
        return match.group(1)

    return None


def extract_postcode_area(text: str) -> Optional[str]:
    """Extract postcode area (first 1-2 letters) from text.

    The area is the letter prefix (e.g., SW from SW3 4PL).

    Args:
        text: Text containing a postcode

    Returns:
        Postcode area in uppercase (e.g., "SW") or None
    """
    district = extract_postcode_district(text)
    if district:
        match = AREA_PATTERN.match(district)
        if match:
            return match.group(1)
    return None


def is_valid_postcode_district(district: str) -> bool:
    """Check if a string is a valid UK postcode district format.

    Args:
        district: String to validate (e.g., "SW3", "W1A")

    Returns:
        True if valid district format, False otherwise
    """
    if not district:
        return False
    return bool(re.match(r'^[A-Z]{1,2}\d{1,2}[A-Z]?$', district.upper()))


def normalize_postcode(postcode: str) -> Optional[str]:
    """Normalize a postcode to standard format.

    Args:
        postcode: Postcode in any format

    Returns:
        Normalized postcode (uppercase, proper spacing) or None if invalid

    Example:
        >>> normalize_postcode("sw3 4pl")
        'SW3 4PL'
        >>> normalize_postcode("SW34PL")
        'SW3 4PL'
    """
    if not postcode:
        return None

    # Remove all spaces and convert to uppercase
    clean = re.sub(r'\s+', '', postcode.upper())

    # Try to parse as full postcode
    # Format: 2-4 chars for outward, 3 chars for inward
    if len(clean) >= 5:
        # Inward code is always last 3 chars
        inward = clean[-3:]
        outward = clean[:-3]

        # Validate inward code format: digit + 2 letters
        if re.match(r'^\d[A-Z]{2}$', inward):
            if is_valid_postcode_district(outward):
                return f"{outward} {inward}"

    return None


# Prime London postcodes commonly used in the scraper
PRIME_LONDON_DISTRICTS = {
    'SW1', 'SW1A', 'SW1E', 'SW1H', 'SW1P', 'SW1V', 'SW1W', 'SW1X', 'SW1Y',
    'SW3', 'SW5', 'SW6', 'SW7', 'SW10', 'SW11',
    'W1', 'W1A', 'W1B', 'W1C', 'W1D', 'W1F', 'W1G', 'W1H', 'W1J', 'W1K',
    'W1S', 'W1T', 'W1U', 'W1W',
    'W2', 'W8', 'W11', 'W14',
    'NW1', 'NW3', 'NW8',
    'WC1', 'WC2',
    'EC1', 'EC2', 'EC3', 'EC4',
}


def is_prime_london(text: str) -> bool:
    """Check if address/postcode is in Prime Central London.

    Args:
        text: Address or postcode text

    Returns:
        True if in prime London area
    """
    district = extract_postcode_district(text)
    if not district:
        return False

    # Check exact match first
    if district in PRIME_LONDON_DISTRICTS:
        return True

    # Check prefix (e.g., SW1 matches SW1A, SW1E, etc.)
    for prime in PRIME_LONDON_DISTRICTS:
        if district.startswith(prime) or prime.startswith(district):
            return True

    return False
