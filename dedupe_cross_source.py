#!/usr/bin/env python3
"""
Cross-Source Deduplication Script

Identifies and handles duplicate properties listed on multiple sources
(e.g., same property on Rightmove AND Savills).

Strategy:
1. Find duplicates using multiple matching criteria
2. Merge data - prefer agent source for sqft, keep best data from each
3. Mark duplicates with canonical_id pointing to the "best" record
4. Optionally remove duplicates, keeping only the record with most data

Usage:
    python3 dedupe_cross_source.py --analyze          # Just show duplicates
    python3 dedupe_cross_source.py --merge            # Merge sqft from agents to rightmove
    python3 dedupe_cross_source.py --mark             # Add canonical_id column
    python3 dedupe_cross_source.py --remove           # Remove duplicates (keep best)
"""

import sqlite3
import argparse
from collections import defaultdict

DB_PATH = 'output/rentals.db'

# Priority order for sources (higher = preferred for canonical)
SOURCE_PRIORITY = {
    'savills': 5,      # Best sqft coverage
    'knightfrank': 4,
    'foxtons': 3,
    'chestertons': 2,
    'rightmove': 1,    # Aggregator, often missing sqft
}


# NOTE: the old fuzzy normalize_address() + address_similarity() (SequenceMatcher)
# identity has been RETIRED (#10). Cross-source identity now comes from the single
# structural-fingerprint core in scripts/dedupe_postgres.py, which find_duplicates()
# below delegates to. This removes the divergent fuzzy path that caused the 39.5%
# over-delete (#5).


def find_duplicates(conn, threshold=0.85):
    """Find potential duplicate properties across sources.

    DELEGATES to the single structural-fingerprint identity core
    (scripts/dedupe_postgres.group_cross_source_clusters) so there is ONE
    cross-source dedupe rule everywhere (#5/#10). The old fuzzy SequenceMatcher
    path that over-clustered distinct same-street properties (39.5% over-delete)
    has been retired.

    `threshold` is accepted for backwards-compatible call sites but is no longer
    used — identity is now address_fingerprint + bedrooms + price±5%, not a
    free-text similarity ratio.

    Returns the same pairwise shape the existing CLI consumers
    (analyze/merge/mark/remove) expect: a list of
    {record1, record2, similarity} dicts. A structural cluster of N members is
    expanded into N-1 pairs, each pairing the canonical (best) record with one
    other member; `similarity` is 1.0 because a fingerprint match is exact.
    """
    # Lazy import to avoid a module-level import cycle: dedupe_postgres imports
    # SOURCE_PRIORITY from this module at its top, so importing it here (inside
    # the function) keeps both import orders clean.
    import os
    import sys
    _scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from dedupe_postgres import group_cross_source_clusters

    cursor = conn.cursor()
    # property_id is required by the fingerprint service's vague-address fallback.
    cursor.execute('''
        SELECT id, source, property_id, address, postcode, price_pcm,
               bedrooms, bathrooms, size_sqft, url
        FROM listings
        WHERE address != '' OR postcode != ''
        ORDER BY source, id
    ''')
    rows = cursor.fetchall()
    print(f"Analyzing {len(rows)} listings for cross-source duplicates...")

    # Adapt DB rows -> the row dicts group_cross_source_clusters expects.
    listings = [
        {
            'id': r[0], 'source': r[1], 'property_id': r[2], 'address': r[3],
            'postcode': r[4], 'price_pcm': r[5], 'bedrooms': r[6],
            'size_sqft': r[8],
        }
        for r in rows
    ]

    def _rec(m):
        return {
            'id': m['id'], 'source': m['source'], 'property_id': m['property_id'],
            'address': m['address'], 'postcode': m['postcode'],
            'price_pcm': m['price_pcm'], 'bedrooms': m['bedrooms'],
            'size_sqft': m['size_sqft'],
        }

    duplicates = []
    for cluster in group_cross_source_clusters(listings):
        # Canonical = highest source priority, then has-sqft, then lowest id —
        # the same rule find_cross_source_duplicate_ids keeps.
        canonical = max(
            cluster,
            key=lambda m: (
                SOURCE_PRIORITY.get(m['source'], 0),
                1 if (m.get('size_sqft') or 0) > 0 else 0,
                -m['id'],
            ),
        )
        for m in cluster:
            if m['id'] == canonical['id']:
                continue
            duplicates.append({
                'record1': _rec(canonical),
                'record2': _rec(m),
                'similarity': 1.0,  # structural fingerprint match is exact
            })

    return duplicates


def analyze_duplicates(conn):
    """Analyze and report on duplicates."""
    duplicates = find_duplicates(conn)

    print(f"\n{'='*70}")
    print(f"CROSS-SOURCE DUPLICATE ANALYSIS")
    print(f"{'='*70}")
    print(f"Found {len(duplicates)} duplicate pairs\n")

    # Count by source pair
    source_pairs = defaultdict(int)
    sqft_opportunities = 0

    for dup in duplicates:
        r1, r2 = dup['record1'], dup['record2']
        pair = tuple(sorted([r1['source'], r2['source']]))
        source_pairs[pair] += 1

        # Check if we can fill sqft
        if (r1['size_sqft'] or 0) == 0 and (r2['size_sqft'] or 0) > 0:
            sqft_opportunities += 1
        elif (r2['size_sqft'] or 0) == 0 and (r1['size_sqft'] or 0) > 0:
            sqft_opportunities += 1

    print("Duplicates by source pair:")
    for pair, count in sorted(source_pairs.items(), key=lambda x: -x[1]):
        print(f"  {pair[0]} ↔ {pair[1]}: {count}")

    print(f"\nSqft enrichment opportunities: {sqft_opportunities}")
    print(f"(Cases where one source has sqft and the other doesn't)")

    # Show sample
    print(f"\n{'='*70}")
    print("SAMPLE DUPLICATES:")
    print(f"{'='*70}")
    for dup in duplicates[:5]:
        r1, r2 = dup['record1'], dup['record2']
        print(f"\n[{r1['source']}] {r1['address']}")
        print(f"  Price: £{r1['price_pcm']:,}, Beds: {r1['bedrooms']}, Sqft: {r1['size_sqft'] or 'MISSING'}")
        print(f"[{r2['source']}] {r2['address']}")
        print(f"  Price: £{r2['price_pcm']:,}, Beds: {r2['bedrooms']}, Sqft: {r2['size_sqft'] or 'MISSING'}")
        print(f"  Similarity: {dup['similarity']:.0%}")

    return duplicates


def merge_sqft(conn, dry_run=True):
    """Merge sqft data from agent sources to Rightmove records."""
    duplicates = find_duplicates(conn)
    cursor = conn.cursor()

    updates = []
    for dup in duplicates:
        r1, r2 = dup['record1'], dup['record2']

        # Find the record missing sqft and the one with sqft
        if (r1['size_sqft'] or 0) == 0 and (r2['size_sqft'] or 0) > 0:
            target, source = r1, r2
        elif (r2['size_sqft'] or 0) == 0 and (r1['size_sqft'] or 0) > 0:
            target, source = r2, r1
        else:
            continue  # Both have or both missing sqft

        updates.append({
            'target_id': target['id'],
            'target_source': target['source'],
            'target_address': target['address'],
            'source_sqft': source['size_sqft'],
            'source_source': source['source'],
        })

    print(f"\n{'='*70}")
    print(f"SQFT MERGE {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*70}")
    print(f"Found {len(updates)} records to update with sqft data\n")

    # Group by target source
    by_source = defaultdict(list)
    for u in updates:
        by_source[u['target_source']].append(u)

    for source, items in sorted(by_source.items()):
        print(f"  {source}: {len(items)} records to enrich")

    if not dry_run and updates:
        print("\nApplying updates...")
        for u in updates:
            cursor.execute(
                'UPDATE listings SET size_sqft = ? WHERE id = ?',
                (u['source_sqft'], u['target_id'])
            )
        conn.commit()
        print(f"Updated {len(updates)} records with sqft data")

    return updates


def mark_canonical(conn, dry_run=True):
    """Add canonical_id to link duplicate records."""
    # First, ensure column exists
    cursor = conn.cursor()

    try:
        cursor.execute('ALTER TABLE listings ADD COLUMN canonical_id INTEGER')
        conn.commit()
        print("Added canonical_id column")
    except sqlite3.OperationalError:
        pass  # Column already exists

    duplicates = find_duplicates(conn)

    # Build groups of duplicates
    groups = []
    id_to_group = {}

    for dup in duplicates:
        id1 = dup['record1']['id']
        id2 = dup['record2']['id']

        if id1 in id_to_group and id2 in id_to_group:
            # Merge groups
            g1 = id_to_group[id1]
            g2 = id_to_group[id2]
            if g1 != g2:
                groups[g1].update(groups[g2])
                for gid in groups[g2]:
                    id_to_group[gid] = g1
                groups[g2] = set()
        elif id1 in id_to_group:
            groups[id_to_group[id1]].add(id2)
            id_to_group[id2] = id_to_group[id1]
        elif id2 in id_to_group:
            groups[id_to_group[id2]].add(id1)
            id_to_group[id1] = id_to_group[id2]
        else:
            new_group = {id1, id2}
            groups.append(new_group)
            id_to_group[id1] = len(groups) - 1
            id_to_group[id2] = len(groups) - 1

    # Remove empty groups
    groups = [g for g in groups if g]

    print(f"\n{'='*70}")
    print(f"CANONICAL MARKING {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*70}")
    print(f"Found {len(groups)} duplicate groups\n")

    updates = 0
    for group in groups:
        if len(group) < 2:
            continue

        # Get full records for this group
        placeholders = ','.join('?' * len(group))
        cursor.execute(f'''
            SELECT id, source, size_sqft FROM listings
            WHERE id IN ({placeholders})
        ''', list(group))
        records = cursor.fetchall()

        # Find canonical (highest priority source with sqft)
        best = None
        best_score = -1
        for rec in records:
            priority = SOURCE_PRIORITY.get(rec[1], 0)
            has_sqft = 1 if (rec[2] or 0) > 0 else 0
            score = priority * 10 + has_sqft
            if score > best_score:
                best_score = score
                best = rec[0]

        if not dry_run:
            for rec in records:
                if rec[0] != best:
                    cursor.execute(
                        'UPDATE listings SET canonical_id = ? WHERE id = ?',
                        (best, rec[0])
                    )
                    updates += 1

    if not dry_run:
        conn.commit()
        print(f"Marked {updates} records with canonical_id")
    else:
        print(f"Would mark {sum(len(g)-1 for g in groups)} records as duplicates")

    return groups


def remove_duplicates(conn, dry_run=True):
    """Remove duplicate records, keeping only the canonical one."""
    groups = mark_canonical(conn, dry_run=True)  # Just analyze
    cursor = conn.cursor()

    to_delete = []
    for group in groups:
        if len(group) < 2:
            continue

        # Get records with data completeness
        placeholders = ','.join('?' * len(group))
        cursor.execute(f'''
            SELECT id, source, size_sqft,
                   (CASE WHEN size_sqft > 0 THEN 1 ELSE 0 END +
                    CASE WHEN bedrooms IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN bathrooms IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN postcode != '' THEN 1 ELSE 0 END) as completeness
            FROM listings
            WHERE id IN ({placeholders})
            ORDER BY completeness DESC,
                     (CASE source
                        WHEN 'savills' THEN 5
                        WHEN 'knightfrank' THEN 4
                        WHEN 'foxtons' THEN 3
                        WHEN 'chestertons' THEN 2
                        ELSE 1 END) DESC
        ''', list(group))
        records = cursor.fetchall()

        # Keep first (best), delete rest
        for rec in records[1:]:
            to_delete.append(rec[0])

    print(f"\n{'='*70}")
    print(f"DUPLICATE REMOVAL {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*70}")
    print(f"Would delete {len(to_delete)} duplicate records")
    print(f"Keeping {len(groups)} canonical records\n")

    if not dry_run and to_delete:
        # === DESTRUCTIVE-OP GUARD (#37): delta-abort + backup before delete ===
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
        from _safe_delete import guarded_delete, SafeDeleteAborted
        total = cursor.execute("SELECT COUNT(*) FROM listings").fetchone()[0]

        def _do_delete(ids):
            ph = ','.join('?' * len(ids))
            cursor.execute(f'DELETE FROM listings WHERE id IN ({ph})', list(ids))

        try:
            guarded_delete(
                cursor, "listings", to_delete,
                total_rows=total, do_delete=_do_delete,
                label="cross-source dedupe --remove",
            )
            conn.commit()
            print(f"Deleted {len(to_delete)} duplicate records (backed up first)")
        except SafeDeleteAborted as e:
            conn.rollback()
            print(f"[ABORTED] {e}")
            raise SystemExit(1)

    return to_delete


def main():
    parser = argparse.ArgumentParser(description='Cross-source deduplication')
    parser.add_argument('--analyze', action='store_true', help='Analyze duplicates')
    parser.add_argument('--merge', action='store_true', help='Merge sqft from agents to rightmove')
    parser.add_argument('--mark', action='store_true', help='Mark duplicates with canonical_id')
    parser.add_argument('--remove', action='store_true', help='Remove duplicates (keep best)')
    parser.add_argument('--execute', action='store_true', help='Actually execute (default is dry-run)')

    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    if args.analyze or not any([args.merge, args.mark, args.remove]):
        analyze_duplicates(conn)

    if args.merge:
        merge_sqft(conn, dry_run=not args.execute)

    if args.mark:
        mark_canonical(conn, dry_run=not args.execute)

    if args.remove:
        remove_duplicates(conn, dry_run=not args.execute)

    conn.close()


if __name__ == '__main__':
    main()
