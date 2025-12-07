#!/usr/bin/env python3
"""
Database Standardization Script

Normalizes categorical fields to help ML models learn better patterns:
- Property types: Consolidate 27 variants into 6 standard categories
- Let types: Extract from property_type into separate field
- Postcodes: Normalize format, infer from area when missing
- Agent brands: Extract main brand from branch names

Usage:
    python standardize_data.py                    # Process output/rentals.db
    python standardize_data.py --dry-run          # Show what would change
    python standardize_data.py --db path/to.db    # Custom database
"""

import sqlite3
import argparse
import re
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROPERTY TYPE MAPPINGS
# ============================================================================
PROPERTY_TYPE_MAP = {
    # Flats/Apartments
    'flat': 'flat',
    'apartment': 'flat',
    'ground flat': 'flat',
    'flat share': 'flat',

    # Studios
    'studio': 'studio',

    # Houses
    'house': 'house',
    'terraced': 'house',
    'end of terrace': 'house',
    'detached': 'house',
    'semi-detached': 'house',
    'town house': 'house',
    'mews': 'house',
    'link detached house': 'house',
    'barn conversion': 'house',
    'house share': 'house',

    # Maisonettes
    'maisonette': 'maisonette',
    'duplex': 'maisonette',

    # Penthouses (premium)
    'penthouse': 'penthouse',

    # Special/Other
    'serviced apartments': 'serviced',
    'house boat': 'other',
    'parking': 'other',
    'garages': 'other',
    'land': 'other',

    # Let types (will be extracted separately)
    'long let': None,  # Will use default
    'short let': None,

    # Empty/Unknown
    '': None,
    'not specified': None,
}

# ============================================================================
# LET TYPE EXTRACTION
# ============================================================================
def extract_let_type(property_type: str) -> str:
    """Extract let type from property_type field."""
    if not property_type:
        return 'unknown'
    pt_lower = property_type.lower()
    if 'short' in pt_lower:
        return 'short'
    elif 'long' in pt_lower:
        return 'long'
    return 'unknown'


# ============================================================================
# POSTCODE NORMALIZATION
# ============================================================================
# Area to typical postcode mapping (for inference)
AREA_TO_POSTCODE = {
    'Belgravia': 'SW1',
    'Chelsea': 'SW3',
    'Earls Court': 'SW5',
    'South Kensington': 'SW7',
    'Knightsbridge': 'SW1X',
    'Kensington': 'W8',
    'Notting Hill': 'W11',
    'Bayswater': 'W2',
    'St Johns Wood': 'NW8',
    'Hampstead': 'NW3',
    'Mayfair': 'W1',
}

def normalize_postcode(postcode: str) -> str:
    """Normalize postcode to outcode format (e.g., SW1A)."""
    if not postcode:
        return None

    # Clean up
    pc = postcode.upper().strip()

    # Extract outcode (first part)
    # Full format: SW1A 1AA or SW1A1AA
    match = re.match(r'^([A-Z]{1,2}\d{1,2}[A-Z]?)', pc)
    if match:
        return match.group(1)

    return None


def infer_postcode_from_area(area: str) -> str:
    """Infer postcode district from area name."""
    if not area:
        return None
    return AREA_TO_POSTCODE.get(area)


# ============================================================================
# AGENT BRAND EXTRACTION
# ============================================================================
KNOWN_BRANDS = [
    'Knight Frank', 'Chestertons', 'Foxtons', 'Savills', 'Hamptons',
    'Dexters', 'Marsh & Parsons', 'John D Wood', 'Harrods Estates',
    'JLL', 'Carter Jonas', 'Domus Nova', 'Sotheby', 'OpenRent',
    'Winkworth', 'Strutt & Parker', 'Kinleigh Folkard & Hayward',
    'KFH', 'Draker', 'Lurot Brand', 'Beauchamp Estates'
]

def extract_agent_brand(agent_name: str) -> str:
    """Extract main brand from agent name."""
    if not agent_name:
        return 'unknown'

    agent_lower = agent_name.lower()

    for brand in KNOWN_BRANDS:
        if brand.lower() in agent_lower:
            return brand

    # Fallback: use first word before comma
    if ',' in agent_name:
        return agent_name.split(',')[0].strip()

    return agent_name


# ============================================================================
# MAIN STANDARDIZATION
# ============================================================================
def standardize_database(db_path: str, dry_run: bool = False):
    """Run all standardization routines."""

    logger.info("=" * 70)
    logger.info("DATABASE STANDARDIZATION")
    logger.info("=" * 70)
    logger.info(f"Database: {db_path}")
    logger.info(f"Dry run: {dry_run}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Add new columns if they don't exist
    new_columns = [
        ('property_type_std', 'TEXT'),
        ('let_type', 'TEXT'),
        ('postcode_normalized', 'TEXT'),
        ('postcode_inferred', 'INTEGER DEFAULT 0'),
        ('agent_brand', 'TEXT'),
    ]

    for col_name, col_type in new_columns:
        try:
            cursor.execute(f'ALTER TABLE listings ADD COLUMN {col_name} {col_type}')
            logger.info(f"[SCHEMA] Added column: {col_name}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Get all listings
    cursor.execute('SELECT id, property_type, area, postcode, agent_name FROM listings')
    rows = cursor.fetchall()

    stats = {
        'total': len(rows),
        'property_type_mapped': 0,
        'let_type_extracted': 0,
        'postcode_normalized': 0,
        'postcode_inferred': 0,
        'agent_brand_extracted': 0,
    }

    updates = []

    for row in rows:
        update = {'id': row['id']}

        # 1. Property type standardization
        original_type = row['property_type'] or ''
        mapped_type = PROPERTY_TYPE_MAP.get(original_type.lower(), 'other')
        if mapped_type is None:
            mapped_type = 'flat'  # Default for long let/short let
        update['property_type_std'] = mapped_type
        if mapped_type != original_type.lower():
            stats['property_type_mapped'] += 1

        # 2. Let type extraction
        let_type = extract_let_type(original_type)
        update['let_type'] = let_type
        if let_type != 'unknown':
            stats['let_type_extracted'] += 1

        # 3. Postcode normalization
        normalized_pc = normalize_postcode(row['postcode'])
        inferred = 0

        if not normalized_pc:
            # Try to infer from area
            normalized_pc = infer_postcode_from_area(row['area'])
            if normalized_pc:
                inferred = 1
                stats['postcode_inferred'] += 1
        else:
            stats['postcode_normalized'] += 1

        update['postcode_normalized'] = normalized_pc
        update['postcode_inferred'] = inferred

        # 4. Agent brand extraction
        brand = extract_agent_brand(row['agent_name'])
        update['agent_brand'] = brand
        stats['agent_brand_extracted'] += 1

        updates.append(update)

    # Apply updates
    if not dry_run:
        for update in updates:
            cursor.execute('''
                UPDATE listings SET
                    property_type_std = ?,
                    let_type = ?,
                    postcode_normalized = ?,
                    postcode_inferred = ?,
                    agent_brand = ?
                WHERE id = ?
            ''', (
                update['property_type_std'],
                update['let_type'],
                update['postcode_normalized'],
                update['postcode_inferred'],
                update['agent_brand'],
                update['id']
            ))

        conn.commit()

    # Print results
    logger.info("")
    logger.info("=" * 70)
    logger.info("STANDARDIZATION RESULTS")
    logger.info("=" * 70)

    logger.info(f"\n[PROPERTY TYPES]")
    logger.info(f"  Total records: {stats['total']}")
    logger.info(f"  Types mapped: {stats['property_type_mapped']}")

    if not dry_run:
        cursor.execute('''
            SELECT property_type_std, COUNT(*) as cnt
            FROM listings GROUP BY property_type_std ORDER BY cnt DESC
        ''')
        for row in cursor.fetchall():
            logger.info(f"    {row[0]}: {row[1]}")

    logger.info(f"\n[LET TYPES]")
    logger.info(f"  Extracted: {stats['let_type_extracted']}")

    if not dry_run:
        cursor.execute('''
            SELECT let_type, COUNT(*) as cnt
            FROM listings GROUP BY let_type ORDER BY cnt DESC
        ''')
        for row in cursor.fetchall():
            logger.info(f"    {row[0]}: {row[1]}")

    logger.info(f"\n[POSTCODES]")
    logger.info(f"  Normalized: {stats['postcode_normalized']}")
    logger.info(f"  Inferred from area: {stats['postcode_inferred']}")

    if not dry_run:
        cursor.execute('''
            SELECT COUNT(*) FROM listings WHERE postcode_normalized IS NOT NULL
        ''')
        has_pc = cursor.fetchone()[0]
        logger.info(f"  Total with postcode: {has_pc}/{stats['total']} ({100*has_pc/stats['total']:.1f}%)")

    logger.info(f"\n[AGENT BRANDS]")
    logger.info(f"  Extracted: {stats['agent_brand_extracted']}")

    if not dry_run:
        cursor.execute('''
            SELECT agent_brand, COUNT(*) as cnt
            FROM listings GROUP BY agent_brand ORDER BY cnt DESC LIMIT 15
        ''')
        for row in cursor.fetchall():
            logger.info(f"    {row[0]}: {row[1]}")

    conn.close()

    logger.info("")
    logger.info("=" * 70)
    if dry_run:
        logger.info("DRY RUN COMPLETE - No changes made")
    else:
        logger.info("STANDARDIZATION COMPLETE")
    logger.info("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Standardize database fields')
    parser.add_argument('--db', default='output/rentals.db', help='Database path')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change')

    args = parser.parse_args()

    standardize_database(args.db, args.dry_run)


if __name__ == '__main__':
    main()
