#!/usr/bin/env python3
"""
Property Listing Deduplication Engine

Identifies and merges duplicate property listings that appear under
different property IDs (e.g., same property listed by multiple agents).

Matching Strategies (in priority order):
1. Coordinate Match - Same lat/lng within 10m radius
2. Exact Match - Same price + bedrooms + normalized address prefix
3. Fuzzy Match - Similar address (>85%) + price within 5% + same bedrooms

Usage:
    python dedupe.py                    # Process output/rentals.db
    python dedupe.py --db path/to.db    # Custom database path
    python dedupe.py --dry-run          # Show what would be deduplicated
"""

import sqlite3
import argparse
import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class DedupeConfig:
    """Configuration for deduplication thresholds."""
    # Coordinate matching
    coord_tolerance_meters: float = 15.0  # Properties within 15m = same location

    # Price matching
    price_tolerance_pct: float = 0.05  # 5% price difference allowed

    # Address matching
    address_prefix_len: int = 25  # Characters to compare for exact match
    fuzzy_threshold: float = 0.85  # 85% similarity for fuzzy match

    # Selection criteria for "best" listing
    prefer_recent: bool = True
    prefer_complete: bool = True


@dataclass
class Listing:
    """A property listing."""
    id: int
    property_id: str
    source: str
    area: str
    address: str
    price_pcm: int
    bedrooms: Optional[int]
    bathrooms: Optional[int]
    latitude: Optional[float]
    longitude: Optional[float]
    property_type: str
    size_sqft: Optional[int]
    agent_name: str
    scraped_at: str
    url: str

    # Computed fields
    address_normalized: str = field(default='', init=False)
    completeness_score: int = field(default=0, init=False)

    def __post_init__(self):
        self.address_normalized = self._normalize_address(self.address)
        self.completeness_score = self._calc_completeness()

    def _normalize_address(self, addr: str) -> str:
        """Normalize address for comparison."""
        if not addr:
            return ''
        # Lowercase, remove extra spaces
        addr = ' '.join(addr.lower().split())
        # Remove common suffixes that vary
        for suffix in [', london', ', sw1', ', sw3', ', sw5', ', sw7', ', w8', ', w11']:
            addr = addr.replace(suffix, '')
        # Remove punctuation
        addr = re.sub(r'[,\.\-]', ' ', addr)
        addr = ' '.join(addr.split())
        return addr

    def _calc_completeness(self) -> int:
        """Score how complete this listing's data is."""
        score = 0
        if self.bedrooms is not None:
            score += 1
        if self.bathrooms is not None:
            score += 1
        if self.size_sqft is not None:
            score += 2  # Size is valuable
        if self.latitude and self.longitude:
            score += 1
        if self.property_type:
            score += 1
        if self.agent_name:
            score += 1
        return score


@dataclass
class DuplicateGroup:
    """A group of duplicate listings."""
    canonical_id: int  # The "best" listing ID
    duplicate_ids: list  # Other listing IDs that are duplicates
    match_reason: str  # Why they were matched
    confidence: float  # 0-1 confidence score


class DeduplicationEngine:
    """
    Identifies and resolves duplicate property listings.
    """

    def __init__(self, db_path: str, config: Optional[DedupeConfig] = None):
        self.db_path = db_path
        self.config = config or DedupeConfig()
        self.listings: list[Listing] = []
        self.groups: list[DuplicateGroup] = []
        self.stats = {
            'total_listings': 0,
            'duplicate_groups': 0,
            'duplicates_found': 0,
            'unique_after_dedupe': 0,
            'by_strategy': defaultdict(int),
            'by_area': defaultdict(lambda: {'before': 0, 'after': 0}),
            'start_time': time.time(),
        }

    def load_listings(self):
        """Load all listings from database."""
        logger.info(f"[LOAD] Reading from {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, property_id, source, area, address, price_pcm,
                   bedrooms, bathrooms, latitude, longitude, property_type,
                   size_sqft, agent_name, scraped_at, url
            FROM listings
            ORDER BY area, price_pcm
        ''')

        for row in cursor.fetchall():
            listing = Listing(
                id=row['id'],
                property_id=row['property_id'],
                source=row['source'],
                area=row['area'],
                address=row['address'] or '',
                price_pcm=row['price_pcm'] or 0,
                bedrooms=row['bedrooms'],
                bathrooms=row['bathrooms'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                property_type=row['property_type'] or '',
                size_sqft=row['size_sqft'],
                agent_name=row['agent_name'] or '',
                scraped_at=row['scraped_at'] or '',
                url=row['url'] or '',
            )
            self.listings.append(listing)
            self.stats['by_area'][listing.area]['before'] += 1

        conn.close()
        self.stats['total_listings'] = len(self.listings)
        logger.info(f"[LOAD] Loaded {len(self.listings)} listings")

    def _haversine_distance(self, lat1: float, lon1: float,
                            lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in meters."""
        R = 6371000  # Earth's radius in meters

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (math.sin(delta_phi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _address_similarity(self, addr1: str, addr2: str) -> float:
        """Calculate similarity between two addresses (0-1)."""
        if not addr1 or not addr2:
            return 0.0
        return SequenceMatcher(None, addr1, addr2).ratio()

    def _price_similar(self, price1: int, price2: int) -> bool:
        """Check if two prices are within tolerance."""
        if price1 == 0 or price2 == 0:
            return False
        diff = abs(price1 - price2) / max(price1, price2)
        return diff <= self.config.price_tolerance_pct

    def _is_match_coords(self, a: Listing, b: Listing) -> bool:
        """Strategy 1: Match by coordinates."""
        if not all([a.latitude, a.longitude, b.latitude, b.longitude]):
            return False

        distance = self._haversine_distance(
            a.latitude, a.longitude, b.latitude, b.longitude
        )

        # Same location AND same bedrooms AND similar price
        if distance <= self.config.coord_tolerance_meters:
            if a.bedrooms == b.bedrooms and self._price_similar(a.price_pcm, b.price_pcm):
                return True

        return False

    def _is_match_exact(self, a: Listing, b: Listing) -> bool:
        """Strategy 2: Exact match on price + beds + address prefix."""
        if a.price_pcm != b.price_pcm:
            return False
        if a.bedrooms != b.bedrooms:
            return False

        prefix_len = self.config.address_prefix_len
        addr_a = a.address_normalized[:prefix_len]
        addr_b = b.address_normalized[:prefix_len]

        if len(addr_a) < 10 or len(addr_b) < 10:
            return False

        return addr_a == addr_b

    def _is_match_fuzzy(self, a: Listing, b: Listing) -> bool:
        """Strategy 3: Fuzzy address match + similar price."""
        if a.bedrooms != b.bedrooms:
            return False

        if not self._price_similar(a.price_pcm, b.price_pcm):
            return False

        similarity = self._address_similarity(
            a.address_normalized, b.address_normalized
        )

        return similarity >= self.config.fuzzy_threshold

    def _select_canonical(self, listings: list[Listing]) -> Listing:
        """Select the best listing from a group of duplicates."""
        # Sort by completeness (desc), then by scraped_at (desc for most recent)
        sorted_listings = sorted(
            listings,
            key=lambda x: (x.completeness_score, x.scraped_at),
            reverse=True
        )
        return sorted_listings[0]

    def find_duplicates(self):
        """Find all duplicate groups using multiple strategies."""
        logger.info("[DEDUPE] Starting duplicate detection...")
        logger.info(f"[CONFIG] Coord tolerance: {self.config.coord_tolerance_meters}m")
        logger.info(f"[CONFIG] Price tolerance: {self.config.price_tolerance_pct*100}%")
        logger.info(f"[CONFIG] Fuzzy threshold: {self.config.fuzzy_threshold*100}%")

        # Track which listings have been assigned to a group
        assigned = set()

        # Group by area first to reduce comparisons
        by_area = defaultdict(list)
        for listing in self.listings:
            by_area[listing.area].append(listing)

        for area, area_listings in by_area.items():
            logger.info(f"[DEDUPE] Processing {area}: {len(area_listings)} listings")

            # Compare each pair
            for i, a in enumerate(area_listings):
                if a.id in assigned:
                    continue

                group_members = [a]
                match_reasons = []

                for j, b in enumerate(area_listings[i+1:], start=i+1):
                    if b.id in assigned:
                        continue

                    # Try each strategy
                    if self._is_match_coords(a, b):
                        group_members.append(b)
                        assigned.add(b.id)
                        match_reasons.append('coords')
                        self.stats['by_strategy']['coords'] += 1
                    elif self._is_match_exact(a, b):
                        group_members.append(b)
                        assigned.add(b.id)
                        match_reasons.append('exact')
                        self.stats['by_strategy']['exact'] += 1
                    elif self._is_match_fuzzy(a, b):
                        group_members.append(b)
                        assigned.add(b.id)
                        match_reasons.append('fuzzy')
                        self.stats['by_strategy']['fuzzy'] += 1

                if len(group_members) > 1:
                    # Found duplicates
                    canonical = self._select_canonical(group_members)
                    duplicate_ids = [l.id for l in group_members if l.id != canonical.id]

                    # Determine primary match reason
                    reason_counts = defaultdict(int)
                    for r in match_reasons:
                        reason_counts[r] += 1
                    primary_reason = max(reason_counts, key=reason_counts.get)

                    # Confidence based on strategy
                    confidence = {'coords': 0.95, 'exact': 0.90, 'fuzzy': 0.75}[primary_reason]

                    self.groups.append(DuplicateGroup(
                        canonical_id=canonical.id,
                        duplicate_ids=duplicate_ids,
                        match_reason=primary_reason,
                        confidence=confidence,
                    ))

                    self.stats['duplicate_groups'] += 1
                    self.stats['duplicates_found'] += len(duplicate_ids)

                    logger.debug(
                        f"[GROUP] {len(group_members)} duplicates: "
                        f"{canonical.address[:40]}... ({primary_reason})"
                    )

        self.stats['unique_after_dedupe'] = (
            self.stats['total_listings'] - self.stats['duplicates_found']
        )

        logger.info(f"[DEDUPE] Found {self.stats['duplicate_groups']} duplicate groups")
        logger.info(f"[DEDUPE] {self.stats['duplicates_found']} duplicates identified")

    def save_results(self, dry_run: bool = False):
        """Save deduplicated results to database."""
        if dry_run:
            logger.info("[DRY-RUN] Would save results (skipping)")
            return

        logger.info("[SAVE] Writing deduplicated data...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create duplicate groups table
        cursor.execute('DROP TABLE IF EXISTS duplicate_groups')
        cursor.execute('''
            CREATE TABLE duplicate_groups (
                id INTEGER PRIMARY KEY,
                canonical_id INTEGER,
                duplicate_id INTEGER,
                match_reason TEXT,
                confidence REAL,
                FOREIGN KEY (canonical_id) REFERENCES listings(id),
                FOREIGN KEY (duplicate_id) REFERENCES listings(id)
            )
        ''')

        # Insert duplicate mappings
        for group in self.groups:
            for dup_id in group.duplicate_ids:
                cursor.execute('''
                    INSERT INTO duplicate_groups
                    (canonical_id, duplicate_id, match_reason, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (group.canonical_id, dup_id, group.match_reason, group.confidence))

        # Create clean listings view
        cursor.execute('DROP VIEW IF EXISTS listings_clean')
        cursor.execute('''
            CREATE VIEW listings_clean AS
            SELECT l.*
            FROM listings l
            WHERE l.id NOT IN (SELECT duplicate_id FROM duplicate_groups)
        ''')

        # Create area stats view
        cursor.execute('DROP VIEW IF EXISTS area_stats_clean')
        cursor.execute('''
            CREATE VIEW area_stats_clean AS
            SELECT
                area,
                COUNT(*) as count,
                ROUND(AVG(price_pcm)) as avg_pcm,
                MIN(price_pcm) as min_pcm,
                MAX(price_pcm) as max_pcm,
                ROUND(AVG(bedrooms), 1) as avg_beds
            FROM listings_clean
            GROUP BY area
            ORDER BY count DESC
        ''')

        # Store dedupe stats
        cursor.execute('DROP TABLE IF EXISTS dedupe_stats')
        cursor.execute('''
            CREATE TABLE dedupe_stats (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')

        stats_to_save = {
            'total_listings': str(self.stats['total_listings']),
            'duplicates_found': str(self.stats['duplicates_found']),
            'unique_after_dedupe': str(self.stats['unique_after_dedupe']),
            'duplicate_groups': str(self.stats['duplicate_groups']),
            'matches_by_coords': str(self.stats['by_strategy']['coords']),
            'matches_by_exact': str(self.stats['by_strategy']['exact']),
            'matches_by_fuzzy': str(self.stats['by_strategy']['fuzzy']),
            'dedupe_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        for key, value in stats_to_save.items():
            cursor.execute('INSERT INTO dedupe_stats VALUES (?, ?)', (key, value))

        conn.commit()
        conn.close()

        logger.info("[SAVE] Created tables: duplicate_groups, dedupe_stats")
        logger.info("[SAVE] Created views: listings_clean, area_stats_clean")

    def print_summary(self):
        """Print detailed summary of deduplication results."""
        elapsed = time.time() - self.stats['start_time']

        print("\n" + "=" * 70)
        print("DEDUPLICATION COMPLETE")
        print("=" * 70)

        print(f"\n[SUMMARY]")
        print(f"  Total listings:      {self.stats['total_listings']:,}")
        print(f"  Duplicates found:    {self.stats['duplicates_found']:,}")
        print(f"  Unique listings:     {self.stats['unique_after_dedupe']:,}")
        print(f"  Reduction:           {self.stats['duplicates_found']/self.stats['total_listings']*100:.1f}%")

        print(f"\n[MATCH STRATEGIES]")
        print(f"  Coordinate matches:  {self.stats['by_strategy']['coords']:,}")
        print(f"  Exact matches:       {self.stats['by_strategy']['exact']:,}")
        print(f"  Fuzzy matches:       {self.stats['by_strategy']['fuzzy']:,}")

        print(f"\n[BY AREA]")
        # Calculate after counts
        duplicate_ids = set()
        for group in self.groups:
            duplicate_ids.update(group.duplicate_ids)

        for listing in self.listings:
            if listing.id not in duplicate_ids:
                self.stats['by_area'][listing.area]['after'] += 1

        for area in sorted(self.stats['by_area'].keys()):
            before = self.stats['by_area'][area]['before']
            after = self.stats['by_area'][area]['after']
            reduction = before - after
            pct = reduction / before * 100 if before > 0 else 0
            print(f"  {area:20} {before:4} -> {after:4} (-{reduction}, {pct:.0f}%)")

        print(f"\n[PERFORMANCE]")
        print(f"  Duration:            {elapsed:.2f}s")

        print("=" * 70 + "\n")

    def show_examples(self, n: int = 5):
        """Show example duplicate groups."""
        print("\n[EXAMPLE DUPLICATE GROUPS]")

        # Get listing lookup
        listing_by_id = {l.id: l for l in self.listings}

        for i, group in enumerate(self.groups[:n]):
            canonical = listing_by_id[group.canonical_id]
            print(f"\n  Group {i+1} ({group.match_reason}, {group.confidence:.0%} confidence):")
            print(f"    Canonical: {canonical.address[:50]}")
            print(f"               £{canonical.price_pcm:,}/pcm, {canonical.bedrooms}bed")
            print(f"               Agent: {canonical.agent_name[:30]}")

            for dup_id in group.duplicate_ids[:3]:
                dup = listing_by_id[dup_id]
                print(f"    Duplicate: {dup.address[:50]}")
                print(f"               Agent: {dup.agent_name[:30]}")

            if len(group.duplicate_ids) > 3:
                print(f"    ... and {len(group.duplicate_ids) - 3} more duplicates")


def main():
    parser = argparse.ArgumentParser(description='Deduplicate property listings')
    parser.add_argument('--db', default='output/rentals.db', help='Database path')
    parser.add_argument('--dry-run', action='store_true', help='Show results without saving')
    parser.add_argument('--examples', type=int, default=5, help='Number of example groups to show')
    parser.add_argument('--coord-tolerance', type=float, default=15.0,
                        help='Coordinate tolerance in meters')
    parser.add_argument('--price-tolerance', type=float, default=0.05,
                        help='Price tolerance as fraction (0.05 = 5%)')
    parser.add_argument('--fuzzy-threshold', type=float, default=0.85,
                        help='Fuzzy address match threshold (0-1)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = DedupeConfig(
        coord_tolerance_meters=args.coord_tolerance,
        price_tolerance_pct=args.price_tolerance,
        fuzzy_threshold=args.fuzzy_threshold,
    )

    engine = DeduplicationEngine(args.db, config)
    engine.load_listings()
    engine.find_duplicates()
    engine.save_results(dry_run=args.dry_run)
    engine.print_summary()

    if args.examples > 0:
        engine.show_examples(args.examples)


if __name__ == '__main__':
    main()
