#!/usr/bin/env python3
"""
Export active listings for similarity search in the Chrome extension API.
Outputs a compact JSON file with key properties for matching.
"""

import json
import re
import sqlite3
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'rentals.db')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chrome-extension', 'api', 'similar_listings.json')


def export_listings():
    """Export active listings with key features for similarity matching."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Query active listings with essential fields
    query = """
    SELECT
        source || ':' || property_id as id,
        url,
        address,
        postcode,
        bedrooms,
        bathrooms,
        size_sqft,
        price_pcm,
        latitude,
        longitude,
        property_type_std,
        features,
        description
    FROM listings
    WHERE is_active = 1
      AND price_pcm IS NOT NULL
      AND price_pcm > 0
      AND bedrooms IS NOT NULL
    ORDER BY last_seen DESC
    """

    cursor = conn.execute(query)
    rows = cursor.fetchall()

    print(f"Found {len(rows)} active listings")

    # Parse amenities from features/description
    amenity_keywords = {
        'balcony': ['balcony'],
        'terrace': ['terrace'],
        'garden': ['garden'],
        'porter': ['porter', 'concierge'],
        'gym': ['gym', 'fitness'],
        'pool': ['pool', 'swimming'],
        'parking': ['parking', 'garage'],
        'lift': ['lift', 'elevator'],
        'ac': ['air con', 'a/c', 'aircon'],
    }

    def parse_amenities(features_str, description_str):
        """Extract amenity flags from text."""
        text = ''
        if features_str:
            text += features_str.lower() + ' '
        if description_str:
            text += description_str.lower()

        amenities = []
        for amenity, keywords in amenity_keywords.items():
            if any(kw in text for kw in keywords):
                amenities.append(amenity)
        return amenities

    # A UK postcode is "OUTWARD INWARD" where the inward code is always a digit
    # followed by two letters (e.g. "9HD"). The outward code IS the district we
    # want (e.g. "SW3", "W8", "SW1X"). A bare district has no inward code.
    _INWARD_RE = re.compile(r'[0-9][A-Z]{2}$')              # trailing inward code
    _DISTRICT_RE = re.compile(r'^[A-Z]{1,2}[0-9][0-9A-Z]?$')  # a clean outward code
    # Full postcode embedded anywhere in free text (for address backfill).
    _FULL_PC_RE = re.compile(r'\b([A-Z]{1,2}[0-9][0-9A-Z]?)\s*[0-9][A-Z]{2}\b')

    def get_postcode_district(postcode, address=''):
        """Extract the postcode district (e.g. 'SW3' from 'SW3 4AA').

        Handles three shapes the DB actually contains:
          - spaced full postcode  "SW3 6SN" -> "SW3"
          - space-less full postcode "SW36SN" -> "SW3" (strip the inward code;
            a naive split-on-space left these as garbage districts before)
          - bare district          "SW3" / "W8" / "SW1X" -> unchanged
        Falls back to the LAST full postcode found in the address when the
        postcode field is missing/unusable. Returns '' if nothing district-like
        is recoverable (do NOT guess a district from a bare token in free text).
        """
        if postcode:
            pc = re.sub(r'\s+', ' ', str(postcode).strip().upper())
            if ' ' in pc:
                out = pc.split(' ')[0]
                if _DISTRICT_RE.match(out):
                    return out
            elif _DISTRICT_RE.match(pc):
                return pc  # already a clean district
            else:
                stripped = _INWARD_RE.sub('', pc)  # "SW36SN" -> "SW3"
                if _DISTRICT_RE.match(stripped):
                    return stripped
        # Backfill from a full postcode embedded in the address (last one wins —
        # the postcode is usually at the end of the address string).
        if address:
            matches = _FULL_PC_RE.findall(address.upper())
            if matches and _DISTRICT_RE.match(matches[-1]):
                return matches[-1]
        return ''

    # Build compact export
    listings = {}
    for row in rows:
        row_dict = dict(row)

        # Parse amenities
        amenities = parse_amenities(row_dict.get('features'), row_dict.get('description'))

        # Build compact record
        # Keys: u=url, a=address, p=postcode district, b=beds, ba=baths, s=sqft,
        #       pr=price, la=lat, lo=lon, t=type, am=amenities
        record = {
            'u': row_dict['url'],
            'a': row_dict['address'] or '',
            'p': get_postcode_district(row_dict['postcode'], row_dict.get('address') or ''),
            'b': row_dict['bedrooms'] or 0,
            'ba': row_dict['bathrooms'] or 1,
            's': row_dict['size_sqft'] or 0,
            'pr': row_dict['price_pcm'],
            't': (row_dict['property_type_std'] or 'flat').lower(),
        }

        # Only include lat/lon if present
        if row_dict['latitude'] and row_dict['longitude']:
            record['la'] = round(row_dict['latitude'], 4)
            record['lo'] = round(row_dict['longitude'], 4)

        # Only include amenities if any detected
        if amenities:
            record['am'] = amenities

        listings[row_dict['id']] = record

    # Write output
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(listings, f, separators=(',', ':'))  # Compact JSON

    file_size = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"Exported {len(listings)} listings to {OUTPUT_PATH} ({file_size:.1f} KB)")

    conn.close()
    return len(listings)


if __name__ == '__main__':
    export_listings()
