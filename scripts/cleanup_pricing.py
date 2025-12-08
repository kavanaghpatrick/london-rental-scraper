#!/usr/bin/env python3
"""
Pricing Cleanup Script - Fix verified pricing issues

Based on Playwright verification of suspicious listings, this script:
1. Clears size_sqft for listings with extreme ppsf (size errors)
2. Converts weekly to monthly pricing for verified Savills listings

Usage:
    python scripts/cleanup_pricing.py --analyze         # Show what would be fixed
    python scripts/cleanup_pricing.py --fix-sizes       # Clear bad sizes (dry-run)
    python scripts/cleanup_pricing.py --fix-sizes --execute
    python scripts/cleanup_pricing.py --fix-weekly      # Fix weekly pricing (dry-run)
    python scripts/cleanup_pricing.py --fix-weekly --execute
"""

import sqlite3
import argparse
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / 'output' / 'rentals.db'

# Verified weekly pricing errors (from Playwright verification)
WEEKLY_PRICING_IDS = [
    'savills_925418',  # £850 pw shown as £850 pcm - Pimlico
    'savills_579040',  # £1,330 pw shown as £1,330 pcm - Queensway
]

# Extreme ppsf thresholds for size errors
PPSF_LOW_THRESHOLD = 1.5   # Below this = size likely overestimated
PPSF_HIGH_THRESHOLD = 50   # Above this = size likely underestimated


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def analyze():
    """Show current state of suspicious listings."""
    conn = get_connection()
    cursor = conn.cursor()

    print("=" * 70)
    print("PRICING ANALYSIS")
    print("=" * 70)
    print()

    # Count by ppsf range
    cursor.execute('''
        SELECT
            CASE
                WHEN ppsf < 1.5 THEN '< 1.5 (extreme low)'
                WHEN ppsf < 2.5 THEN '1.5-2.5 (low)'
                WHEN ppsf > 50 THEN '> 50 (extreme high)'
                WHEN ppsf > 25 THEN '25-50 (high)'
                ELSE '2.5-25 (normal)'
            END as ppsf_range,
            COUNT(*) as count
        FROM (
            SELECT ROUND(price_pcm * 1.0 / size_sqft, 2) as ppsf
            FROM listings
            WHERE is_active = 1 AND size_sqft > 0 AND price_pcm > 0
        )
        GROUP BY ppsf_range
        ORDER BY ppsf_range
    ''')
    print("PPSF Distribution:")
    for row in cursor.fetchall():
        print(f"  {row['ppsf_range']}: {row['count']}")
    print()

    # Extreme low ppsf (size overestimated)
    cursor.execute('''
        SELECT source, property_id, price_pcm, size_sqft,
               ROUND(price_pcm * 1.0 / size_sqft, 2) as ppsf,
               address
        FROM listings
        WHERE is_active = 1 AND size_sqft > 0 AND price_pcm > 0
          AND price_pcm * 1.0 / size_sqft < ?
        ORDER BY ppsf
    ''', (PPSF_LOW_THRESHOLD,))

    rows = cursor.fetchall()
    print(f"EXTREME LOW PPSF (< {PPSF_LOW_THRESHOLD}) - {len(rows)} listings:")
    print("  These have SIZE likely OVERESTIMATED")
    print("-" * 70)
    for row in rows[:10]:
        print(f"  {row['source']}/{row['property_id']}")
        print(f"    £{row['price_pcm']:,} pcm, {row['size_sqft']:,} sqft = £{row['ppsf']}/sqft")
        print(f"    {row['address'][:60]}...")
    print()

    # Extreme high ppsf (size underestimated)
    cursor.execute('''
        SELECT source, property_id, price_pcm, size_sqft,
               ROUND(price_pcm * 1.0 / size_sqft, 2) as ppsf,
               address
        FROM listings
        WHERE is_active = 1 AND size_sqft > 0 AND price_pcm > 0
          AND price_pcm * 1.0 / size_sqft > ?
        ORDER BY ppsf DESC
    ''', (PPSF_HIGH_THRESHOLD,))

    rows = cursor.fetchall()
    print(f"EXTREME HIGH PPSF (> {PPSF_HIGH_THRESHOLD}) - {len(rows)} listings:")
    print("  These have SIZE likely UNDERESTIMATED")
    print("-" * 70)
    for row in rows[:10]:
        print(f"  {row['source']}/{row['property_id']}")
        print(f"    £{row['price_pcm']:,} pcm, {row['size_sqft']:,} sqft = £{row['ppsf']}/sqft")
        print(f"    {row['address'][:60]}...")
    print()

    # Known weekly pricing errors
    cursor.execute('''
        SELECT source, property_id, price_pcm, size_sqft,
               ROUND(price_pcm * 1.0 / size_sqft, 2) as ppsf,
               CAST(price_pcm * 52.0 / 12 AS INTEGER) as corrected_pcm,
               address
        FROM listings
        WHERE property_id IN ({})
    '''.format(','.join('?' * len(WEEKLY_PRICING_IDS))), WEEKLY_PRICING_IDS)

    rows = cursor.fetchall()
    print(f"VERIFIED WEEKLY PRICING ERRORS - {len(rows)} listings:")
    print("-" * 70)
    for row in rows:
        corrected_ppsf = row['corrected_pcm'] / row['size_sqft'] if row['size_sqft'] else 0
        print(f"  {row['source']}/{row['property_id']}")
        print(f"    Current: £{row['price_pcm']:,} pcm = £{row['ppsf']}/sqft")
        print(f"    Corrected: £{row['corrected_pcm']:,} pcm = £{corrected_ppsf:.2f}/sqft")
        print(f"    {row['address'][:60]}...")
    print()

    conn.close()


def fix_sizes(execute=False):
    """Clear size_sqft for listings with extreme ppsf."""
    conn = get_connection()
    cursor = conn.cursor()

    # Find listings with extreme ppsf
    cursor.execute('''
        SELECT id, source, property_id, price_pcm, size_sqft,
               ROUND(price_pcm * 1.0 / size_sqft, 2) as ppsf
        FROM listings
        WHERE is_active = 1 AND size_sqft > 0 AND price_pcm > 0
          AND (
            price_pcm * 1.0 / size_sqft < ?
            OR price_pcm * 1.0 / size_sqft > ?
          )
    ''', (PPSF_LOW_THRESHOLD, PPSF_HIGH_THRESHOLD))

    rows = cursor.fetchall()

    print("=" * 70)
    print(f"FIX SIZES - {'EXECUTING' if execute else 'DRY RUN'}")
    print("=" * 70)
    print(f"Found {len(rows)} listings with extreme ppsf")
    print()

    if not rows:
        print("Nothing to fix!")
        conn.close()
        return

    # Show what will be fixed
    low_count = sum(1 for r in rows if r['ppsf'] < PPSF_LOW_THRESHOLD)
    high_count = sum(1 for r in rows if r['ppsf'] > PPSF_HIGH_THRESHOLD)
    print(f"  LOW ppsf (< {PPSF_LOW_THRESHOLD}): {low_count} listings - size overestimated")
    print(f"  HIGH ppsf (> {PPSF_HIGH_THRESHOLD}): {high_count} listings - size underestimated")
    print()

    print("Sample of listings to fix:")
    for row in rows[:10]:
        issue = "LOW" if row['ppsf'] < PPSF_LOW_THRESHOLD else "HIGH"
        print(f"  [{issue}] {row['source']}/{row['property_id']}: {row['size_sqft']:,} sqft @ £{row['ppsf']}/sqft")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more")
    print()

    if execute:
        ids = [r['id'] for r in rows]
        placeholders = ','.join('?' * len(ids))
        cursor.execute(f'''
            UPDATE listings
            SET size_sqft = NULL
            WHERE id IN ({placeholders})
        ''', ids)
        conn.commit()
        print(f"EXECUTED: Cleared size_sqft for {cursor.rowcount} listings")
        print("These listings will be re-enriched on next floorplan run")
    else:
        print("DRY RUN - no changes made. Add --execute to apply fixes.")

    conn.close()


def fix_weekly(execute=False):
    """Convert weekly to monthly pricing for verified listings."""
    conn = get_connection()
    cursor = conn.cursor()

    print("=" * 70)
    print(f"FIX WEEKLY PRICING - {'EXECUTING' if execute else 'DRY RUN'}")
    print("=" * 70)
    print(f"Verified weekly pricing errors: {len(WEEKLY_PRICING_IDS)}")
    print()

    # Get current state
    cursor.execute('''
        SELECT id, source, property_id, price_pcm, size_sqft,
               ROUND(price_pcm * 1.0 / size_sqft, 2) as ppsf,
               CAST(price_pcm * 52.0 / 12 AS INTEGER) as corrected_pcm,
               address
        FROM listings
        WHERE property_id IN ({})
    '''.format(','.join('?' * len(WEEKLY_PRICING_IDS))), WEEKLY_PRICING_IDS)

    rows = cursor.fetchall()

    if not rows:
        print("No matching listings found (may already be fixed)")
        conn.close()
        return

    for row in rows:
        corrected_ppsf = row['corrected_pcm'] / row['size_sqft'] if row['size_sqft'] else 0
        print(f"  {row['property_id']}:")
        print(f"    Current: £{row['price_pcm']:,} pcm = £{row['ppsf']}/sqft")
        print(f"    Will be: £{row['corrected_pcm']:,} pcm = £{corrected_ppsf:.2f}/sqft")
    print()

    if execute:
        cursor.execute('''
            UPDATE listings
            SET price_pcm = CAST(price_pcm * 52.0 / 12 AS INTEGER)
            WHERE property_id IN ({})
        '''.format(','.join('?' * len(WEEKLY_PRICING_IDS))), WEEKLY_PRICING_IDS)
        conn.commit()
        print(f"EXECUTED: Updated pricing for {cursor.rowcount} listings")
    else:
        print("DRY RUN - no changes made. Add --execute to apply fixes.")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Cleanup pricing issues')
    parser.add_argument('--analyze', action='store_true', help='Analyze current state')
    parser.add_argument('--fix-sizes', action='store_true', help='Clear bad sizes')
    parser.add_argument('--fix-weekly', action='store_true', help='Fix weekly pricing')
    parser.add_argument('--execute', action='store_true', help='Actually execute changes')
    args = parser.parse_args()

    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")
    print()

    if args.analyze or not (args.fix_sizes or args.fix_weekly):
        analyze()

    if args.fix_sizes:
        fix_sizes(execute=args.execute)

    if args.fix_weekly:
        fix_weekly(execute=args.execute)


if __name__ == '__main__':
    main()
