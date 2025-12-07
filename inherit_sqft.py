#!/usr/bin/env python3
"""
Cross-Source Square Footage Inheritance

Propagates sqft data from agency listings (Knight Frank, Chestertons, Foxtons)
to their duplicate Rightmove listings. Since agencies list on Rightmove but
include sqft in their own listings, we can inherit this data.

Usage:
    python inherit_sqft.py                    # Process output/rentals.db
    python inherit_sqft.py --db path/to.db    # Custom database path
    python inherit_sqft.py --dry-run          # Show what would be updated
"""

import sqlite3
import argparse
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def inherit_sqft(db_path: str, dry_run: bool = False):
    """Inherit sqft from agency listings to their Rightmove duplicates."""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # First, find all duplicate groups where at least one listing has sqft
    # and at least one doesn't
    cursor.execute('''
        SELECT
            dg.canonical_id,
            dg.duplicate_id,
            dg.match_reason,
            l1.source as canonical_source,
            l1.size_sqft as canonical_sqft,
            l1.address as canonical_address,
            l2.source as dup_source,
            l2.size_sqft as dup_sqft
        FROM duplicate_groups dg
        JOIN listings l1 ON l1.id = dg.canonical_id
        JOIN listings l2 ON l2.id = dg.duplicate_id
    ''')

    # Build complete picture of each group
    # A group includes the canonical and all its duplicates
    groups = defaultdict(list)

    for row in cursor.fetchall():
        canonical_id = row['canonical_id']
        groups[canonical_id].append({
            'id': row['canonical_id'],
            'source': row['canonical_source'],
            'sqft': row['canonical_sqft'],
            'address': row['canonical_address'],
        })
        groups[canonical_id].append({
            'id': row['duplicate_id'],
            'source': row['dup_source'],
            'sqft': row['dup_sqft'],
            'address': None,
        })

    # Dedupe within groups (canonical appears multiple times)
    for canonical_id in groups:
        seen_ids = set()
        unique = []
        for item in groups[canonical_id]:
            if item['id'] not in seen_ids:
                seen_ids.add(item['id'])
                unique.append(item)
        groups[canonical_id] = unique

    stats = {
        'groups_analyzed': len(groups),
        'groups_with_sqft_gap': 0,
        'listings_updated': 0,
        'by_source': defaultdict(int),
        'inherited_from': defaultdict(int),
    }

    updates = []

    for canonical_id, members in groups.items():
        # Find listings with sqft
        with_sqft = [m for m in members if m['sqft'] and m['sqft'] > 0]
        without_sqft = [m for m in members if not m['sqft'] or m['sqft'] == 0]

        if with_sqft and without_sqft:
            stats['groups_with_sqft_gap'] += 1

            # Use the first sqft value (could be smarter - average, max, etc.)
            # Prefer agency sqft over Rightmove
            best_sqft = None
            best_source = None
            for m in with_sqft:
                if m['source'] in ('knightfrank', 'foxtons', 'chestertons'):
                    best_sqft = m['sqft']
                    best_source = m['source']
                    break
            if not best_sqft:
                best_sqft = with_sqft[0]['sqft']
                best_source = with_sqft[0]['source']

            # Update listings missing sqft
            for m in without_sqft:
                updates.append({
                    'id': m['id'],
                    'sqft': best_sqft,
                    'source': m['source'],
                    'inherited_from': best_source,
                    'address': members[0]['address'],
                })
                stats['listings_updated'] += 1
                stats['by_source'][m['source']] += 1
                stats['inherited_from'][best_source] += 1

    # Log summary
    logger.info("=" * 70)
    logger.info("CROSS-SOURCE SQFT INHERITANCE")
    logger.info("=" * 70)
    logger.info(f"[ANALYSIS] Duplicate groups: {stats['groups_analyzed']}")
    logger.info(f"[ANALYSIS] Groups with sqft gap: {stats['groups_with_sqft_gap']}")
    logger.info(f"[ANALYSIS] Listings to update: {stats['listings_updated']}")

    logger.info(f"\n[BY TARGET SOURCE]")
    for source, count in sorted(stats['by_source'].items(), key=lambda x: -x[1]):
        logger.info(f"  {source}: {count} listings")

    logger.info(f"\n[INHERITED FROM]")
    for source, count in sorted(stats['inherited_from'].items(), key=lambda x: -x[1]):
        logger.info(f"  {source}: {count} sqft values")

    # Show examples
    logger.info(f"\n[EXAMPLES]")
    for u in updates[:10]:
        logger.info(f"  {u['address'][:50] if u['address'] else 'N/A'}...")
        logger.info(f"    {u['source']} <- {u['sqft']} sqft from {u['inherited_from']}")

    if dry_run:
        logger.info(f"\n[DRY-RUN] Would update {len(updates)} listings")
    else:
        # Apply updates
        for u in updates:
            cursor.execute(
                'UPDATE listings SET size_sqft = ? WHERE id = ?',
                (u['sqft'], u['id'])
            )

        conn.commit()
        logger.info(f"\n[COMPLETE] Updated {len(updates)} listings with inherited sqft")

    # Show new coverage
    logger.info(f"\n[NEW SQFT COVERAGE]")
    cursor.execute('''
        SELECT source,
               COUNT(*) as total,
               SUM(CASE WHEN size_sqft > 0 THEN 1 ELSE 0 END) as has_sqft
        FROM listings GROUP BY source
    ''')
    for row in cursor.fetchall():
        pct = 100 * row['has_sqft'] / row['total'] if row['total'] else 0
        logger.info(f"  {row['source']}: {row['has_sqft']}/{row['total']} ({pct:.1f}%)")

    conn.close()
    logger.info("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Inherit sqft across duplicate listings')
    parser.add_argument('--db', default='output/rentals.db', help='Database path')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated')

    args = parser.parse_args()

    inherit_sqft(args.db, args.dry_run)


if __name__ == '__main__':
    main()
