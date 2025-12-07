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
import sys

# Import from the service
sys.path.insert(0, str(Path(__file__).parent))
from property_scraper.services.fingerprint import generate_fingerprint, parse_address

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
            parsed = parse_address(address or '', postcode or '')
            print(f"  ID {id_}:")
            print(f"    Address: {(address or '')[:60]}...")
            print(f"    Parsed: num={parsed['street_number']}, street={parsed['street_name']}, unit={parsed['unit']}")
            print(f"    Fingerprint: {parsed['fingerprint']}")
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

    # Check for potential duplicates (same fingerprint, different source)
    cursor.execute("""
        SELECT address_fingerprint, COUNT(*) as cnt,
               GROUP_CONCAT(DISTINCT source) as sources
        FROM listings
        WHERE address_fingerprint IS NOT NULL
        GROUP BY address_fingerprint
        HAVING cnt > 1
        ORDER BY cnt DESC
        LIMIT 15
    """)
    dupes = cursor.fetchall()
    if dupes:
        print(f"\nPotential cross-source duplicates (same fingerprint):")
        for fp, cnt, sources in dupes:
            cursor.execute("""
                SELECT source, address, postcode, price_pcm
                FROM listings
                WHERE address_fingerprint = ?
                LIMIT 3
            """, (fp,))
            examples = cursor.fetchall()
            print(f"\n  Fingerprint {fp}: {cnt} listings from [{sources}]")
            for src, addr, pc, price in examples:
                print(f"    [{src}] {addr[:50]}... {pc} - {price}/pcm")

    conn.close()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    backfill(execute=args.execute)
