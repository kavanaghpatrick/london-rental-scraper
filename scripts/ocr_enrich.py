#!/usr/bin/env python3
"""
OCR Enrichment Script - Extract floor data from floorplan images

This script processes listings that have floorplan URLs but are missing
floor data (floor_count, property_levels, size_sqft).

It downloads floorplan images, runs OCR via FloorplanExtractor, and
updates the database with extracted data.

Usage:
    python scripts/ocr_enrich.py                    # Run on all sources
    python scripts/ocr_enrich.py --source rightmove # Single source
    python scripts/ocr_enrich.py --limit 100        # Limit processing
    python scripts/ocr_enrich.py --dry-run          # Preview only

Output fields updated:
    - size_sqft (if missing)
    - floor_count
    - property_levels (single_floor, duplex, triplex, multi_floor)
    - has_basement, has_lower_ground, has_ground, has_mezzanine
    - has_first_floor, has_second_floor, has_third_floor, has_fourth_plus
    - has_roof_terrace
"""

import sqlite3
import argparse
import requests
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from property_scraper.utils.floorplan_extractor import FloorplanExtractor, OCR_AVAILABLE

if not OCR_AVAILABLE:
    print("ERROR: pytesseract not available. Install with: pip install pytesseract")
    print("Also ensure tesseract is installed: brew install tesseract")
    sys.exit(1)


DB_PATH = Path(__file__).parent.parent / 'output' / 'rentals.db'


def get_listings_needing_ocr(source=None, limit=None):
    """Get listings with floorplan URLs but missing floor data."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = '''
        SELECT
            id, source, property_id, floorplan_url, address,
            size_sqft, floor_count, property_levels
        FROM listings
        WHERE is_active = 1
          AND floorplan_url IS NOT NULL
          AND floorplan_url != ''
          AND (
              floor_count IS NULL OR floor_count = 0
              OR size_sqft IS NULL OR size_sqft = 0
          )
    '''

    params = []
    if source:
        query += ' AND source = ?'
        params.append(source)

    query += ' ORDER BY source, id'

    if limit:
        query += f' LIMIT {int(limit)}'

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def download_image(url, timeout=30):
    """Download image from URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'image/*,*/*;q=0.8',
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return response.content
    except Exception as e:
        pass
    return None


def process_listing(listing, extractor):
    """Process a single listing - download floorplan and run OCR."""
    result = {
        'id': listing['id'],
        'property_id': listing['property_id'],
        'source': listing['source'],
        'success': False,
        'sqft': None,
        'floor_count': None,
        'property_levels': None,
        'floor_data': None,
        'error': None,
    }

    try:
        # Download image
        image_bytes = download_image(listing['floorplan_url'])
        if not image_bytes or len(image_bytes) < 1000:
            result['error'] = 'download_failed'
            return result

        # Run OCR - returns FloorplanData dataclass
        data = extractor.extract_from_bytes(image_bytes)
        if not data:
            result['error'] = 'ocr_failed'
            return result

        # Check for placeholder images (e.g., foxtons "Unable to find")
        if data.raw_text and 'unable to find' in data.raw_text.lower():
            result['error'] = 'placeholder_image'
            return result

        # Check extraction confidence
        if data.extraction_confidence < 0.1:
            result['error'] = 'low_confidence'
            return result

        # Extract results
        result['success'] = True

        # Extract sqft (FloorplanData is a dataclass, use attribute access)
        if data.total_sqft and data.total_sqft > 100:
            result['sqft'] = data.total_sqft

        # Extract floor data from FloorData dataclass
        if data.floor_data:
            fd = data.floor_data
            result['floor_data'] = {
                'has_basement': fd.has_basement,
                'has_lower_ground': fd.has_lower_ground,
                'has_ground': fd.has_ground,
                'has_mezzanine': fd.has_mezzanine,
                'has_first_floor': fd.has_first_floor,
                'has_second_floor': fd.has_second_floor,
                'has_third_floor': fd.has_third_floor,
                'has_fourth_plus': fd.has_fourth_plus,
                'has_roof_terrace': fd.has_roof_terrace,
            }

            # Use floor_count from FloorData if available
            if fd.floor_count and fd.floor_count > 0:
                result['floor_count'] = fd.floor_count
                result['property_levels'] = fd.property_levels

    except Exception as e:
        result['error'] = str(e)[:100]

    return result


def update_database(results, dry_run=False):
    """Update database with OCR results."""
    if dry_run:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    updated = 0

    for r in results:
        if not r['success']:
            continue

        updates = []
        params = []

        if r.get('sqft'):
            updates.append('size_sqft = ?')
            params.append(r['sqft'])

        if r.get('floor_count'):
            updates.append('floor_count = ?')
            params.append(r['floor_count'])

        if r.get('property_levels'):
            updates.append('property_levels = ?')
            params.append(r['property_levels'])

        if r.get('floor_data'):
            fd = r['floor_data']
            for flag in ['has_basement', 'has_lower_ground', 'has_ground', 'has_mezzanine',
                         'has_first_floor', 'has_second_floor', 'has_third_floor',
                         'has_fourth_plus', 'has_roof_terrace']:
                if fd.get(flag):
                    updates.append(f'{flag} = ?')
                    params.append(fd[flag])

        if updates:
            params.append(r['id'])
            query = f"UPDATE listings SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            updated += 1

    conn.commit()
    conn.close()
    return updated


def main():
    parser = argparse.ArgumentParser(description='OCR enrichment for floorplan images')
    parser.add_argument('--source', '-s', help='Specific source to process')
    parser.add_argument('--limit', '-l', type=int, help='Max listings to process')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Concurrent workers')
    parser.add_argument('--dry-run', action='store_true', help='Preview without updating DB')
    args = parser.parse_args()

    print("=" * 70)
    print("OCR ENRICHMENT - FLOORPLAN DATA EXTRACTION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.source:
        print(f"Source: {args.source}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print(f"Workers: {args.workers}")
    if args.dry_run:
        print("Mode: DRY RUN (no database updates)")
    print()

    # Get listings
    listings = get_listings_needing_ocr(args.source, args.limit)
    print(f"Found {len(listings)} listings needing OCR")

    if not listings:
        print("Nothing to process!")
        return

    # Show breakdown by source
    by_source = {}
    for l in listings:
        by_source[l['source']] = by_source.get(l['source'], 0) + 1
    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")
    print()

    # Initialize extractor
    extractor = FloorplanExtractor()

    # Process with thread pool
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_listing, l, extractor): l for l in listings}

        completed = 0
        successful = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            results.append(result)

            if result['success']:
                successful += 1

            if completed % 50 == 0 or completed == len(listings):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed}/{len(listings)} ({successful} success) | {rate:.1f}/sec")

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total processed: {len(results)}")
    print(f"Successful OCR: {sum(1 for r in results if r['success'])}")
    print(f"Downloaded failed: {sum(1 for r in results if r.get('error') == 'download_failed')}")
    print(f"OCR failed: {sum(1 for r in results if r.get('error') == 'ocr_failed')}")
    print(f"Other errors: {sum(1 for r in results if r.get('error') and r['error'] not in ['download_failed', 'ocr_failed'])}")
    print()

    # Count extractions
    sqft_extracted = sum(1 for r in results if r.get('sqft'))
    floor_count_extracted = sum(1 for r in results if r.get('floor_count'))
    levels_extracted = sum(1 for r in results if r.get('property_levels'))

    print(f"Sqft extracted: {sqft_extracted}")
    print(f"Floor count extracted: {floor_count_extracted}")
    print(f"Property levels extracted: {levels_extracted}")
    print()

    # Show sample results
    print("Sample extractions:")
    for r in results[:5]:
        if r['success']:
            print(f"  {r['property_id']}: sqft={r.get('sqft')}, floors={r.get('floor_count')}, levels={r.get('property_levels')}")
    print()

    # Update database
    if not args.dry_run:
        updated = update_database(results)
        print(f"Database updated: {updated} records")
    else:
        print("DRY RUN - no database updates made")

    print(f"\nDuration: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
