#!/usr/bin/env python3
"""
Address fingerprinting service for cross-source deduplication.

Generates deterministic hash identifiers from property addresses,
enabling O(1) duplicate detection at scrape time.

Addresses Gemini review feedback:
- Canonicalizes suffixes instead of stripping (prevents collisions)
- Handles "St" vs "Saint" properly
- Splits letter suffixes from numbers (123a -> 123 + unit a)
- Supports textual units (Basement, Ground Floor)
"""

import hashlib
import re
from typing import Optional, Tuple

# Textual unit descriptors (no number required)
# IMPORTANT: Use tuple (ordered) instead of frozenset to ensure deterministic
# iteration order across Python runs with different PYTHONHASHSEED values.
# Ordered alphabetically for consistent fingerprint generation.
# (Fix for Codex review: frozenset iteration was non-deterministic)
TEXTUAL_UNITS = (
    'basement', 'ground', 'lower', 'maisonette', 'mews',
    'penthouse', 'studio', 'top', 'upper',
)

# Unit prefixes that expect a number
# IMPORTANT: Use tuple for deterministic iteration order.
UNIT_PREFIXES = (
    'apartment', 'apt', 'flat', 'floor', 'level', 'room', 'suite', 'unit',
)

# Suffix canonicalization map (normalize to full form)
SUFFIX_CANONICAL = {
    'st': 'street', 'str': 'street', 'street': 'street',
    'rd': 'road', 'road': 'road',
    'ln': 'lane', 'lane': 'lane',
    'ave': 'avenue', 'av': 'avenue', 'avenue': 'avenue',
    'dr': 'drive', 'drv': 'drive', 'drive': 'drive',
    'pl': 'place', 'place': 'place',
    'ct': 'court', 'court': 'court',
    'gdns': 'gardens', 'gdn': 'gardens', 'gardens': 'gardens',
    'ter': 'terrace', 'terrace': 'terrace',
    'cl': 'close', 'close': 'close',
    'cres': 'crescent', 'crescent': 'crescent',
    'sq': 'square', 'square': 'square',
    'way': 'way',
    'mews': 'mews',
    'row': 'row',
    'hill': 'hill',
    'park': 'park',
    'grove': 'grove',
    'walk': 'walk',
    'passage': 'passage',
}


def normalize_postcode(postcode: Optional[str]) -> str:
    """Normalize UK postcode to lowercase without spaces."""
    if not postcode:
        return ""
    return re.sub(r'\s+', '', postcode.lower().strip())


def expand_saint(text: str) -> str:
    """
    Expand 'St' to 'Saint' when it appears as a place name prefix.

    "St John's Wood" -> "Saint John's Wood"
    "High St" -> "High St" (suffix, not expanded)
    """
    # Pattern: St followed by uppercase letter (place name) or at start
    # But NOT at end of string (which would be suffix)
    text = re.sub(r'\bSt\.?\s+([A-Z])', r'Saint \1', text)
    text = re.sub(r'\bst\.?\s+([a-z])', r'saint \1', text.lower())
    return text


def extract_street_number_and_unit(address: str) -> Tuple[str, str]:
    """
    Extract street number and unit from address.

    Returns (street_number, unit_identifier).

    Handles:
    - "123 High Street" -> ("123", "")
    - "123a High Street" -> ("123", "a")
    - "Flat 1, 123 High Street" -> ("123", "1")
    - "Basement, 10 High St" -> ("10", "basement")
    """
    if not address:
        return "", ""

    address_lower = address.lower()
    unit = ""

    # Check for textual units first (Basement, Ground Floor, etc.)
    for textual in TEXTUAL_UNITS:
        if re.search(rf'\b{textual}\b', address_lower):
            unit = textual
            address_lower = re.sub(rf'\b{textual}\s*(flat|floor)?\s*[,/]?\s*', '', address_lower)
            break

    # Check for numbered units (Flat 1, Apartment 5, etc.)
    if not unit:
        for prefix in UNIT_PREFIXES:
            match = re.search(rf'\b{prefix}\s*(\d+[a-z]?)\b', address_lower)
            if match:
                unit = match.group(1)
                address_lower = re.sub(rf'\b{prefix}\s*\d+[a-z]?\s*[,/]?\s*', '', address_lower)
                break

    # Now find street number (first number after removing unit prefix)
    # Look for pattern: number possibly followed by letter
    match = re.search(r'\b(\d+)([a-z])?\b', address_lower)
    if match:
        street_num = match.group(1)
        letter_suffix = match.group(2) or ""

        # If we found a letter suffix and no unit yet, use it as unit
        if letter_suffix and not unit:
            unit = letter_suffix

        return street_num, unit

    return "", unit


def canonicalize_suffix(word: str) -> Optional[str]:
    """
    Return canonical form of street suffix, or None if not a suffix.
    """
    return SUFFIX_CANONICAL.get(word.lower())


def normalize_street_name(address: str) -> str:
    """
    Normalize street name, preserving canonical suffix.

    "123 High Street" -> "high street"
    "456 Victoria Rd" -> "victoria road"
    "Flat 1, St John's Wood Road" -> "saint johns wood road"
    """
    if not address:
        return ""

    # Expand Saint first
    address = expand_saint(address)
    address_lower = address.lower()

    # Remove unit prefix and number
    for prefix in UNIT_PREFIXES:
        address_lower = re.sub(rf'\b{prefix}\s*\d*[a-z]?\s*[,/]?\s*', '', address_lower)

    # Remove textual units
    for textual in TEXTUAL_UNITS:
        address_lower = re.sub(rf'\b{textual}\s*(flat|floor)?\s*[,/]?\s*', '', address_lower)

    # Remove house number at start
    address_lower = re.sub(r'^\s*\d+[a-z]?(?:-\d+[a-z]?)?\s*[,/]?\s*', '', address_lower)

    # Remove punctuation except apostrophes
    address_lower = re.sub(r"[^\w\s'-]", ' ', address_lower)

    # Split into words
    words = address_lower.split()

    # Process words: keep significant ones, canonicalize suffixes
    result = []
    for word in words:
        # Skip very short words and pure numbers
        if len(word) <= 1 or word.isdigit():
            continue

        # Clean apostrophes
        word = word.replace("'", "")

        # Check if it's a suffix
        canonical = canonicalize_suffix(word)
        if canonical:
            result.append(canonical)
        elif word not in {'the', 'and', 'of', 'in', 'at', 'london'}:
            result.append(word)

    # Take first 3 significant words (name + suffix usually)
    return ' '.join(result[:3])


def generate_fingerprint(
    address: str,
    postcode: str,
    include_unit: bool = True
) -> str:
    """
    Generate a deterministic fingerprint for a property address.

    Args:
        address: Full address string
        postcode: UK postcode
        include_unit: If True, include unit number in fingerprint

    Returns:
        16-character hex string fingerprint
    """
    pc = normalize_postcode(postcode)
    street_num, unit = extract_street_number_and_unit(address)
    street_name = normalize_street_name(address)

    # Build fingerprint key
    components = [pc, street_num, street_name]

    if include_unit and unit:
        components.append(unit)

    key = '_'.join(c for c in components if c)

    # Hash and truncate
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def parse_address(address: str, postcode: str) -> dict:
    """
    Parse address into normalized components.

    Useful for debugging and testing.
    """
    street_num, unit = extract_street_number_and_unit(address)
    return {
        'postcode': normalize_postcode(postcode),
        'street_number': street_num,
        'street_name': normalize_street_name(address),
        'unit': unit,
        'fingerprint': generate_fingerprint(address, postcode),
    }
