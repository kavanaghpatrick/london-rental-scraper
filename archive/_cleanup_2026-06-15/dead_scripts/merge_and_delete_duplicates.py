#!/usr/bin/env python3
"""
Merge Duplicate Data and Delete

For each duplicate group:
1. Find all fields where canonical is NULL but a duplicate has data
2. Merge the best data into the canonical record
3. Delete the duplicate records

This ensures no data is lost when removing duplicates.
"""

import sqlite3
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'output', 'rentals.db')

# Fields that can be merged from duplicates to canonical
MERGEABLE_FIELDS = [
    'size_sqft',
    'bathrooms',
    'latitude',
    'longitude',
    'property_type',
    'furnished',
    'features',
    'summary',
    'description',
    'agent_name',
    'added_date',
]


def get_duplicate_groups(conn):
    """Get all duplicate groups with canonical and duplicate IDs."""
    cursor = conn.cursor()

    # Get unique canonical IDs
    cursor.execute('SELECT DISTINCT canonical_id FROM duplicate_groups')
    canonical_ids = [row[0] for row in cursor.fetchall()]

    groups = []
    for canonical_id in canonical_ids:
        cursor.execute(
            'SELECT duplicate_id FROM duplicate_groups WHERE canonical_id = ?',
            (canonical_id,)
        )
        duplicate_ids = [row[0] for row in cursor.fetchall()]
        groups.append({
            'canonical_id': canonical_id,
            'duplicate_ids': duplicate_ids
        })

    return groups


def get_record(conn, record_id):
    """Get a full record by ID."""
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM listings WHERE id = ?', (record_id,))
    row = cursor.fetchone()
    if row:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    return None


def merge_group(conn, group, dry_run=True):
    """Merge data from duplicates into canonical record."""
    canonical_id = group['canonical_id']
    duplicate_ids = group['duplicate_ids']

    # Get canonical record
    canonical = get_record(conn, canonical_id)
    if not canonical:
        return {'merged_fields': [], 'error': 'Canonical not found'}

    # Get all duplicate records
    duplicates = []
    for dup_id in duplicate_ids:
        dup = get_record(conn, dup_id)
        if dup:
            duplicates.append(dup)

    if not duplicates:
        return {'merged_fields': [], 'error': 'No duplicates found'}

    # Find fields to merge
    updates = {}
    merge_sources = {}

    for field in MERGEABLE_FIELDS:
        canonical_value = canonical.get(field)

        # Check if canonical is missing this field
        if canonical_value is None or canonical_value == '' or canonical_value == 0:
            # Look for best value in duplicates
            best_value = None
            best_source = None

            for dup in duplicates:
                dup_value = dup.get(field)

                if dup_value is not None and dup_value != '' and dup_value != 0:
                    # For text fields, prefer longer values
                    if isinstance(dup_value, str):
                        if best_value is None or len(str(dup_value)) > len(str(best_value)):
                            best_value = dup_value
                            best_source = dup.get('id')
                    # For numeric fields, prefer non-zero values
                    else:
                        if best_value is None:
                            best_value = dup_value
                            best_source = dup.get('id')

            if best_value is not None:
                updates[field] = best_value
                merge_sources[field] = best_source

    # Apply updates
    if updates and not dry_run:
        cursor = conn.cursor()
        set_clause = ', '.join(f'{field} = ?' for field in updates.keys())
        values = list(updates.values()) + [canonical_id]
        cursor.execute(
            f'UPDATE listings SET {set_clause} WHERE id = ?',
            values
        )

    return {
        'canonical_id': canonical_id,
        'canonical_address': canonical.get('address', '')[:50],
        'merged_fields': list(updates.keys()),
        'merge_sources': merge_sources,
        'duplicate_count': len(duplicates)
    }


def delete_duplicates(conn, dry_run=True):
    """Delete all duplicate records."""
    cursor = conn.cursor()

    # Get count first
    cursor.execute('SELECT COUNT(*) FROM duplicate_groups')
    count = cursor.fetchone()[0]

    if not dry_run:
        cursor.execute('''
            DELETE FROM listings
            WHERE id IN (SELECT duplicate_id FROM duplicate_groups)
        ''')
        conn.commit()

    return count


def analyze_merge_potential(conn):
    """Analyze what data can be merged from duplicates."""
    groups = get_duplicate_groups(conn)

    print(f"\n{'='*70}")
    print("MERGE POTENTIAL ANALYSIS")
    print(f"{'='*70}")
    print(f"\nAnalyzing {len(groups)} duplicate groups...")

    field_stats = {field: 0 for field in MERGEABLE_FIELDS}
    total_merges = 0

    for group in groups:
        result = merge_group(conn, group, dry_run=True)
        for field in result.get('merged_fields', []):
            field_stats[field] += 1
            total_merges += 1

    print(f"\nFields that can be enriched from duplicates:")
    print(f"{'Field':<20} {'Records to Enrich':>20}")
    print("-" * 45)

    for field, count in sorted(field_stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"{field:<20} {count:>20}")

    print(f"\nTotal field values to merge: {total_merges}")

    return field_stats


def run_merge_and_delete(conn, dry_run=True):
    """Run the full merge and delete process."""
    groups = get_duplicate_groups(conn)

    print(f"\n{'='*70}")
    print(f"MERGE AND DELETE {'(DRY RUN)' if dry_run else '(EXECUTING)'}")
    print(f"{'='*70}")

    # Phase 1: Merge data
    print(f"\nPhase 1: Merging data from {len(groups)} duplicate groups...")

    merge_count = 0
    field_totals = {field: 0 for field in MERGEABLE_FIELDS}

    for i, group in enumerate(groups):
        result = merge_group(conn, group, dry_run=dry_run)

        if result.get('merged_fields'):
            merge_count += 1
            for field in result['merged_fields']:
                field_totals[field] += 1

            if merge_count <= 5:  # Show first 5 examples
                print(f"\n  [{result['canonical_id']}] {result['canonical_address']}")
                print(f"      Merged: {', '.join(result['merged_fields'])}")

    if merge_count > 5:
        print(f"\n  ... and {merge_count - 5} more merges")

    print(f"\nMerge summary:")
    for field, count in sorted(field_totals.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {field}: {count} records enriched")

    if not dry_run:
        conn.commit()
        print("\nMerges committed to database.")

    # Phase 2: Delete duplicates
    print(f"\nPhase 2: Deleting duplicate records...")

    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM duplicate_groups')
    delete_count = cursor.fetchone()[0]

    if not dry_run:
        cursor.execute('''
            DELETE FROM listings
            WHERE id IN (SELECT duplicate_id FROM duplicate_groups)
        ''')
        conn.commit()
        print(f"Deleted {delete_count} duplicate records.")

        # Clean up the duplicate_groups table
        cursor.execute('DELETE FROM duplicate_groups')
        conn.commit()
        print("Cleared duplicate_groups table.")
    else:
        print(f"Would delete {delete_count} duplicate records.")

    # Final count
    cursor.execute('SELECT COUNT(*) FROM listings')
    final_count = cursor.fetchone()[0]

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Groups processed:     {len(groups)}")
    print(f"Records enriched:     {merge_count}")
    print(f"Duplicates deleted:   {delete_count if not dry_run else f'{delete_count} (pending)'}")
    print(f"Final listing count:  {final_count}")

    return {
        'groups': len(groups),
        'merged': merge_count,
        'deleted': delete_count if not dry_run else 0,
        'final_count': final_count
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Merge duplicate data and delete')
    parser.add_argument('--analyze', action='store_true', help='Only analyze merge potential')
    parser.add_argument('--execute', action='store_true', help='Actually execute (default is dry-run)')

    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    if args.analyze:
        analyze_merge_potential(conn)
    else:
        run_merge_and_delete(conn, dry_run=not args.execute)

    conn.close()


if __name__ == '__main__':
    main()
