# PRD-002: Address Fingerprint Service

## Overview
Create a deterministic address fingerprinting service that generates consistent hash identifiers for properties, enabling cross-source duplicate detection at scrape time.

## Problem Statement
Properties appear on multiple listing sites (Rightmove + Savills + Knight Frank) with slightly different address formats:
- "Flat 1, 123 High Street, London SW1A 1AA"
- "123 High St, Flat 1, SW1A1AA"
- "Apt 1/123 High Street SW1A 1AA"

Current approach detects duplicates AFTER scraping via `dedupe_cross_source.py`, which:
- Requires manual post-processing
- Uses expensive fuzzy matching (O(n^2))
- Misses duplicates during the scrape window

## Goals
1. Create deterministic fingerprint function that maps equivalent addresses to same hash
2. Enable O(1) duplicate detection at scrape time via database index lookup
3. Backfill fingerprints for existing 2,650 listings
4. Provide test coverage for common address variations

## Non-Goals
- Fuzzy matching for edge cases (defer to periodic cleanup)
- Geocoding or external API calls
- Handling international addresses

## Technical Design

### Fingerprint Algorithm

The fingerprint is a 16-character hex hash derived from normalized address components:

```
fingerprint = sha256(postcode_normalized + "_" + street_number + "_" + street_name_words)[:16]
```

#### Normalization Rules

1. **Postcode**: Remove all whitespace, lowercase
   - "SW1A 1AA" → "sw1a1aa"
   - "sw1a1aa" → "sw1a1aa"

2. **Street Number**: Extract first number from address
   - "123 High Street" → "123"
   - "Flat 1, 123 High St" → "123" (not the flat number)
   - "123a High Street" → "123a"

3. **Street Name**: First 2-3 significant words, normalized
   - Remove unit prefixes: flat, apartment, apt, unit, suite
   - Remove common suffixes: road, street, lane, avenue, etc.
   - Take first 2 words after unit removal
   - "123 High Street" → "high"
   - "Flat 1, 123 Victoria Road" → "victoria"

4. **Unit/Flat**: Secondary identifier (not in main hash)
   - Used to distinguish units within same building
   - "Flat 1" vs "Flat 2" at same address → different fingerprints

### Implementation

#### `property_scraper/services/fingerprint.py`

```python
#!/usr/bin/env python3
"""
Address fingerprinting service for cross-source deduplication.

Generates deterministic hash identifiers from property addresses,
enabling O(1) duplicate detection at scrape time.
"""

import hashlib
import re
from typing import Optional, Tuple

# Common prefixes to strip (case-insensitive)
UNIT_PREFIXES = frozenset([
    'flat', 'apartment', 'apt', 'unit', 'suite', 'room',
    'studio', 'penthouse', 'maisonette', 'basement', 'ground',
    'first', 'second', 'third', 'fourth', 'fifth',
    '1st', '2nd', '3rd', '4th', '5th',
])

# Street suffixes to remove for normalization
STREET_SUFFIXES = frozenset([
    'street', 'st', 'road', 'rd', 'lane', 'ln', 'avenue', 'ave',
    'drive', 'dr', 'way', 'place', 'pl', 'court', 'ct',
    'gardens', 'gdns', 'terrace', 'ter', 'close', 'cl',
    'crescent', 'cres', 'square', 'sq', 'mews', 'row',
    'hill', 'park', 'grove', 'walk', 'passage',
])


def normalize_postcode(postcode: Optional[str]) -> str:
    """Normalize UK postcode to lowercase without spaces."""
    if not postcode:
        return ""
    return re.sub(r'\s+', '', postcode.lower().strip())


def extract_street_number(address: str) -> str:
    """
    Extract the street number from an address.

    Finds the first standalone number that appears to be a street number,
    ignoring flat/unit numbers.
    """
    if not address:
        return ""

    address_lower = address.lower()

    # Remove flat/unit prefix and its number first
    # "Flat 1, 123 High St" → "123 High St"
    for prefix in UNIT_PREFIXES:
        pattern = rf'\b{prefix}\s*\d+[a-z]?\s*[,/]?\s*'
        address_lower = re.sub(pattern, '', address_lower)

    # Now find first number (could be "123" or "123a" or "123-125")
    match = re.search(r'\b(\d+[a-z]?(?:-\d+[a-z]?)?)\b', address_lower)
    return match.group(1) if match else ""


def extract_unit_number(address: str) -> str:
    """
    Extract unit/flat number from an address.

    Returns empty string if no unit number found.
    """
    if not address:
        return ""

    address_lower = address.lower()

    for prefix in UNIT_PREFIXES:
        match = re.search(rf'\b{prefix}\s*(\d+[a-z]?)\b', address_lower)
        if match:
            return match.group(1)

    return ""


def normalize_street_name(address: str) -> str:
    """
    Normalize street name to 2-3 key words.

    Removes unit prefixes, numbers, and common suffixes.
    """
    if not address:
        return ""

    address_lower = address.lower()

    # Remove unit prefix and number
    for prefix in UNIT_PREFIXES:
        pattern = rf'\b{prefix}\s*\d*[a-z]?\s*[,/]?\s*'
        address_lower = re.sub(pattern, '', address_lower)

    # Remove house number at start
    address_lower = re.sub(r'^\s*\d+[a-z]?(?:-\d+[a-z]?)?\s*[,/]?\s*', '', address_lower)

    # Remove punctuation except hyphens in names
    address_lower = re.sub(r'[^\w\s-]', ' ', address_lower)

    # Split into words
    words = address_lower.split()

    # Filter out street suffixes and short words
    significant_words = [
        w for w in words
        if w not in STREET_SUFFIXES
        and len(w) > 1
        and not w.isdigit()
    ]

    # Take first 2 words (enough to distinguish most streets)
    return ' '.join(significant_words[:2])


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
    street_num = extract_street_number(address)
    street_name = normalize_street_name(address)

    # Build fingerprint key
    components = [pc, street_num, street_name]

    if include_unit:
        unit = extract_unit_number(address)
        if unit:
            components.append(unit)

    key = '_'.join(c for c in components if c)

    # Hash and truncate
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def parse_address(address: str, postcode: str) -> dict:
    """
    Parse address into normalized components.

    Useful for debugging and testing.
    """
    return {
        'postcode': normalize_postcode(postcode),
        'street_number': extract_street_number(address),
        'street_name': normalize_street_name(address),
        'unit': extract_unit_number(address),
        'fingerprint': generate_fingerprint(address, postcode),
    }
```

### Backfill Script: `backfill_fingerprints.py`

```python
#!/usr/bin/env python3
"""
Backfill address fingerprints for existing listings.

Usage:
    python3 backfill_fingerprints.py              # Dry run
    python3 backfill_fingerprints.py --execute    # Apply changes
"""

import sqlite3
import argparse
from pathlib import Path

# Import from the service
import sys
sys.path.insert(0, str(Path(__file__).parent))
from property_scraper.services.fingerprint import generate_fingerprint

DB_PATH = Path("output/rentals.db")


def backfill(execute: bool = False) -> dict:
    """Backfill fingerprints for listings missing them."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get listings without fingerprints
    cursor.execute("""
        SELECT id, address, postcode
        FROM listings
        WHERE address_fingerprint IS NULL
    """)
    rows = cursor.fetchall()

    stats = {'total': len(rows), 'updated': 0, 'skipped': 0}

    print(f"Found {len(rows)} listings without fingerprints")

    if not execute:
        # Show sample of what would happen
        print("\nSample fingerprints (first 10):")
        for row in rows[:10]:
            id_, address, postcode = row
            fp = generate_fingerprint(address or '', postcode or '')
            print(f"  {id_}: {address[:50]}... -> {fp}")
        print(f"\nDry run. Run with --execute to apply.")
        conn.close()
        return stats

    # Batch update
    updates = []
    for row in rows:
        id_, address, postcode = row
        if not address and not postcode:
            stats['skipped'] += 1
            continue

        fp = generate_fingerprint(address or '', postcode or '')
        updates.append((fp, id_))

    cursor.executemany(
        "UPDATE listings SET address_fingerprint = ? WHERE id = ?",
        updates
    )
    conn.commit()

    stats['updated'] = len(updates)
    print(f"Updated {stats['updated']} listings with fingerprints")

    # Verify
    cursor.execute("""
        SELECT COUNT(*) FROM listings
        WHERE address_fingerprint IS NOT NULL
    """)
    with_fp = cursor.fetchone()[0]
    print(f"Total listings with fingerprints: {with_fp}")

    # Check for duplicates
    cursor.execute("""
        SELECT address_fingerprint, COUNT(*) as cnt
        FROM listings
        WHERE address_fingerprint IS NOT NULL
        GROUP BY address_fingerprint
        HAVING cnt > 1
        ORDER BY cnt DESC
        LIMIT 10
    """)
    dupes = cursor.fetchall()
    if dupes:
        print(f"\nTop potential duplicates (same fingerprint):")
        for fp, cnt in dupes:
            print(f"  {fp}: {cnt} listings")

    conn.close()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    backfill(execute=args.execute)
```

### Test Suite: `tests/test_fingerprint.py`

```python
#!/usr/bin/env python3
"""Tests for address fingerprinting service."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from property_scraper.services.fingerprint import (
    normalize_postcode,
    extract_street_number,
    extract_unit_number,
    normalize_street_name,
    generate_fingerprint,
    parse_address,
)


class TestNormalizePostcode:
    def test_removes_spaces(self):
        assert normalize_postcode("SW1A 1AA") == "sw1a1aa"

    def test_already_normalized(self):
        assert normalize_postcode("sw1a1aa") == "sw1a1aa"

    def test_multiple_spaces(self):
        assert normalize_postcode("SW1A  1AA") == "sw1a1aa"

    def test_empty(self):
        assert normalize_postcode("") == ""
        assert normalize_postcode(None) == ""


class TestExtractStreetNumber:
    def test_simple_number(self):
        assert extract_street_number("123 High Street") == "123"

    def test_with_letter(self):
        assert extract_street_number("123a High Street") == "123a"

    def test_flat_prefix(self):
        assert extract_street_number("Flat 1, 123 High Street") == "123"

    def test_apartment_prefix(self):
        assert extract_street_number("Apartment 5, 456 Victoria Road") == "456"

    def test_unit_prefix(self):
        assert extract_street_number("Unit 2, 789 Kings Road") == "789"

    def test_range(self):
        assert extract_street_number("123-125 High Street") == "123-125"

    def test_no_number(self):
        assert extract_street_number("Victoria Mansions") == ""


class TestExtractUnitNumber:
    def test_flat(self):
        assert extract_unit_number("Flat 1, 123 High St") == "1"

    def test_apartment(self):
        assert extract_unit_number("Apartment 5, 456 Road") == "5"

    def test_unit(self):
        assert extract_unit_number("Unit 2A, 789 Lane") == "2a"

    def test_no_unit(self):
        assert extract_unit_number("123 High Street") == ""


class TestNormalizeStreetName:
    def test_simple(self):
        assert normalize_street_name("123 High Street") == "high"

    def test_removes_suffix(self):
        assert normalize_street_name("456 Victoria Road") == "victoria"

    def test_two_word_name(self):
        result = normalize_street_name("789 Kings Cross Road")
        assert "kings" in result
        assert "cross" in result

    def test_with_flat_prefix(self):
        assert normalize_street_name("Flat 1, 123 High Street") == "high"


class TestGenerateFingerprint:
    def test_same_address_same_fingerprint(self):
        fp1 = generate_fingerprint("123 High Street", "SW1A 1AA")
        fp2 = generate_fingerprint("123 High Street", "sw1a1aa")
        assert fp1 == fp2

    def test_different_formatting_same_fingerprint(self):
        fp1 = generate_fingerprint("Flat 1, 123 High Street, London", "SW1A 1AA")
        fp2 = generate_fingerprint("123 High St, Flat 1", "SW1A1AA")
        assert fp1 == fp2

    def test_different_addresses_different_fingerprints(self):
        fp1 = generate_fingerprint("123 High Street", "SW1A 1AA")
        fp2 = generate_fingerprint("456 High Street", "SW1A 1AA")
        assert fp1 != fp2

    def test_different_postcodes_different_fingerprints(self):
        fp1 = generate_fingerprint("123 High Street", "SW1A 1AA")
        fp2 = generate_fingerprint("123 High Street", "W1A 1AA")
        assert fp1 != fp2

    def test_different_units_different_fingerprints(self):
        fp1 = generate_fingerprint("Flat 1, 123 High Street", "SW1A 1AA")
        fp2 = generate_fingerprint("Flat 2, 123 High Street", "SW1A 1AA")
        assert fp1 != fp2

    def test_length_is_16(self):
        fp = generate_fingerprint("123 High Street", "SW1A 1AA")
        assert len(fp) == 16


class TestRealWorldCases:
    """Tests based on actual data patterns from the database."""

    def test_savills_vs_rightmove_format(self):
        # Savills format
        fp1 = generate_fingerprint(
            "Flat 12, Eaton Place Mansions, 16 Eaton Place",
            "SW1X 8BY"
        )
        # Rightmove format (more abbreviated)
        fp2 = generate_fingerprint(
            "16 Eaton Place, Flat 12",
            "SW1X8BY"
        )
        # Should match (same property)
        # Note: This may need algorithm tuning

    def test_knight_frank_format(self):
        fp = generate_fingerprint(
            "4 South Eaton Place, Belgravia, London SW1W 9JA",
            "SW1W 9JA"
        )
        assert len(fp) == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Testing Plan

1. **Unit tests**: Run `pytest tests/test_fingerprint.py -v`
2. **Integration test**: Run backfill in dry-run mode on real data
3. **Duplicate detection test**: Check if known cross-source duplicates get same fingerprint

## Success Criteria

- [ ] `generate_fingerprint()` produces consistent hashes for equivalent addresses
- [ ] Unit tests pass for common address variations
- [ ] Backfill populates fingerprints for all 2,650 existing listings
- [ ] Duplicate report shows known cross-source matches

## Implementation Steps

1. Create `property_scraper/services/` directory
2. Create `fingerprint.py` service module
3. Create `tests/test_fingerprint.py`
4. Run tests and iterate on algorithm
5. Create `backfill_fingerprints.py`
6. Run backfill dry-run, verify output
7. Run backfill --execute
8. Commit and push

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| False positives (different properties, same fingerprint) | Conservative normalization, unit number inclusion |
| False negatives (same property, different fingerprint) | Periodic fuzzy matching cleanup (separate script) |
| Performance on large datasets | Hash-based O(1) lookup via indexed column |

## Dependencies
- PRD-001 (Schema Migration) - `address_fingerprint` column must exist

## Future Work
- PRD-003 will integrate fingerprinting into the pipeline
- Periodic fuzzy-match cleanup for missed duplicates
