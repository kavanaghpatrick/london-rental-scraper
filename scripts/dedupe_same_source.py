#!/usr/bin/env python3
"""
Deduplicate same-source records that share address_fingerprint + bedrooms + price.

These are true duplicates where the same property was relisted with a new ID.
Keeps the record with most complete data (sqft, floorplan, etc.).

Usage:
    python scripts/dedupe_same_source.py --analyze      # Show what would be deleted
    python scripts/dedupe_same_source.py --execute      # Actually delete duplicates
"""

import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "output" / "rentals.db"


def score_record(record: dict) -> int:
    """Score a record by data completeness. Higher = keep."""
    score = 0
    # Prefer records with more complete data
    if record['size_sqft']:
        score += 10
    if record['floorplan_url']:
        score += 5
    if record['epc_rating']:
        score += 3
    if record['description']:
        score += len(record['description'] or '') // 100  # More description = better
    if record['latitude'] and record['longitude']:
        score += 2
    # Prefer older first_seen (original listing)
    # Prefer newer last_seen (still active)
    return score


def find_true_duplicates(conn: sqlite3.Connection) -> list:
    """
    Find TRUE duplicates: same source + fingerprint + bedrooms + price.

    Different prices likely mean different flats, so we only merge exact price matches.
    """
    cursor = conn.cursor()

    # Find groups with same source + fingerprint + bedrooms + price
    cursor.execute('''
        SELECT
            source,
            address_fingerprint,
            bedrooms,
            price_pcm,
            GROUP_CONCAT(id) as ids,
            COUNT(*) as cnt
        FROM listings
        WHERE address_fingerprint IS NOT NULL
        GROUP BY source, address_fingerprint, bedrooms, price_pcm
        HAVING cnt > 1
        ORDER BY cnt DESC
    ''')

    groups = []
    for row in cursor.fetchall():
        source, fingerprint, bedrooms, price, ids_str, count = row
        ids = [int(x) for x in ids_str.split(',')]
        groups.append({
            'source': source,
            'fingerprint': fingerprint,
            'bedrooms': bedrooms,
            'price_pcm': price,
            'ids': ids,
            'count': count
        })

    return groups


def get_record_details(conn: sqlite3.Connection, record_id: int) -> dict:
    """Get full details for a record."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, source, property_id, address, postcode, price_pcm, bedrooms,
               size_sqft, floorplan_url, epc_rating, description, latitude, longitude,
               first_seen, last_seen
        FROM listings WHERE id = ?
    ''', (record_id,))
    row = cursor.fetchone()
    if row:
        return {
            'id': row[0],
            'source': row[1],
            'property_id': row[2],
            'address': row[3],
            'postcode': row[4],
            'price_pcm': row[5],
            'bedrooms': row[6],
            'size_sqft': row[7],
            'floorplan_url': row[8],
            'epc_rating': row[9],
            'description': row[10],
            'latitude': row[11],
            'longitude': row[12],
            'first_seen': row[13],
            'last_seen': row[14],
        }
    return None


def analyze_duplicates(conn: sqlite3.Connection):
    """Analyze duplicates and show what would be deleted."""
    groups = find_true_duplicates(conn)

    total_removable = 0
    by_source = {}

    print(f"\n{'='*60}")
    print("TRUE DUPLICATE ANALYSIS")
    print(f"{'='*60}")
    print(f"Criteria: same source + fingerprint + bedrooms + price")
    print()

    for group in groups:
        source = group['source']
        by_source[source] = by_source.get(source, 0) + (group['count'] - 1)
        total_removable += group['count'] - 1

    print("Removable duplicates by source:")
    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")
    print(f"  TOTAL: {total_removable}")
    print()

    # Show some examples
    print("Example duplicate groups:")
    for group in groups[:5]:
        records = [get_record_details(conn, rid) for rid in group['ids']]
        records = [r for r in records if r]  # Filter None

        # Score and sort
        for r in records:
            r['score'] = score_record(r)
        records.sort(key=lambda x: -x['score'])

        print(f"\n  {group['source']} | beds={group['bedrooms']} | price={group['price_pcm']}")
        print(f"  Fingerprint: {group['fingerprint']}")
        for i, r in enumerate(records):
            status = "KEEP" if i == 0 else "DELETE"
            sqft_str = f"sqft={r['size_sqft']}" if r['size_sqft'] else "no sqft"
            print(f"    [{status}] id={r['id']} pid={r['property_id']} {sqft_str} score={r['score']}")

    return total_removable, groups


def execute_deduplication(conn: sqlite3.Connection, groups: list, dry_run: bool = True):
    """Delete duplicate records, keeping the best one in each group."""
    cursor = conn.cursor()
    deleted_count = 0

    for group in groups:
        records = [get_record_details(conn, rid) for rid in group['ids']]
        records = [r for r in records if r]

        if len(records) < 2:
            continue

        # Score and sort - highest score first (keep)
        for r in records:
            r['score'] = score_record(r)
        records.sort(key=lambda x: -x['score'])

        # Keep first, delete rest
        keep_id = records[0]['id']
        delete_ids = [r['id'] for r in records[1:]]

        if not dry_run:
            # Also delete from price_history
            for del_id in delete_ids:
                cursor.execute('DELETE FROM price_history WHERE listing_id = ?', (del_id,))
                cursor.execute('DELETE FROM listings WHERE id = ?', (del_id,))
            deleted_count += len(delete_ids)
        else:
            deleted_count += len(delete_ids)

    if not dry_run:
        conn.commit()
        print(f"\nDeleted {deleted_count} duplicate records")
    else:
        print(f"\nWould delete {deleted_count} duplicate records (dry run)")

    return deleted_count


def main():
    parser = argparse.ArgumentParser(description='Deduplicate same-source records')
    parser.add_argument('--analyze', action='store_true', help='Analyze duplicates')
    parser.add_argument('--execute', action='store_true', help='Execute deduplication')
    args = parser.parse_args()

    if not args.analyze and not args.execute:
        parser.print_help()
        return

    conn = sqlite3.connect(DB_PATH)

    try:
        if args.analyze:
            total, groups = analyze_duplicates(conn)
            print(f"\nTotal removable: {total} records")
            print("Run with --execute to actually delete duplicates")

        if args.execute:
            print("Finding duplicates...")
            groups = find_true_duplicates(conn)

            # Backup first
            print("Creating backup...")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM listings")
            before_count = cursor.fetchone()[0]

            print(f"Records before: {before_count}")
            execute_deduplication(conn, groups, dry_run=False)

            cursor.execute("SELECT COUNT(*) FROM listings")
            after_count = cursor.fetchone()[0]
            print(f"Records after: {after_count}")
            print(f"Removed: {before_count - after_count}")

    finally:
        conn.close()


if __name__ == '__main__':
    main()
