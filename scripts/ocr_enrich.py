#!/usr/bin/env python3
"""
OCR Enrichment Script - Extract floor data from floorplan images

This script processes listings that have floorplan URLs but are missing
floor data (floor_count, property_levels, size_sqft).

It downloads floorplan images, runs OCR via FloorplanExtractor, and
updates the database with extracted data.

Supports both SQLite (local) and Postgres (Vercel/GitHub Actions).
Set POSTGRES_URL environment variable to use Postgres.

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
import os
import signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime

# Force unbuffered output for CI/CD environments
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

def log(msg):
    """Print with immediate flush for CI/CD visibility."""
    print(msg, flush=True)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from property_scraper.utils.floorplan_extractor import FloorplanExtractor, OCR_AVAILABLE

if not OCR_AVAILABLE:
    log("ERROR: pytesseract not available. Install with: pip install pytesseract")
    log("Also ensure tesseract is installed: brew install tesseract")
    sys.exit(1)


DB_PATH = Path(__file__).parent.parent / 'output' / 'rentals.db'

# Database connection helpers
def get_postgres_url():
    """Get Postgres URL from environment."""
    return os.environ.get('POSTGRES_URL')

def get_connection():
    """Get database connection (Postgres if available, else SQLite)."""
    postgres_url = get_postgres_url()
    if postgres_url:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(postgres_url)
        return conn, 'postgres'
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn, 'sqlite'

def get_param_placeholder(db_type):
    """Get parameter placeholder for database type."""
    return '%s' if db_type == 'postgres' else '?'


def get_listings_needing_ocr(source=None, limit=None):
    """Get listings with floorplan URLs but missing floor data."""
    conn, db_type = get_connection()
    placeholder = get_param_placeholder(db_type)

    if db_type == 'postgres':
        import psycopg2.extras
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    else:
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
        query += f' AND source = {placeholder}'
        params.append(source)

    query += ' ORDER BY source, id'

    if limit:
        query += f' LIMIT {int(limit)}'

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Convert to list of dicts
    if db_type == 'postgres':
        return [dict(row) for row in rows]
    else:
        return [dict(row) for row in rows]


def transform_url(url):
    """Transform proxied URLs to direct CDN URLs for better access."""
    from urllib.parse import urlparse, parse_qs, unquote

    # Chestertons _next/image proxy -> direct homeflow-assets URL
    if '_next/image' in url and 'chestertons.co.uk' in url:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if 'url' in params:
            return unquote(params['url'][0])

    return url


def download_image(url, timeout=10):
    """Download image from URL with short timeout to skip slow CDNs."""
    # Transform proxied URLs to direct URLs
    url = transform_url(url)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'image/*,*/*;q=0.8',
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return response.content
        # Log non-200 responses (but not every one - too noisy)
        if response.status_code != 403:  # 403s are common for some CDNs
            log(f"  [WARN] Download returned {response.status_code} for {url[:60]}...")
    except requests.exceptions.Timeout:
        pass  # Expected for slow CDNs - don't log every one
    except requests.exceptions.ConnectionError:
        pass  # CDN blocking - expected
    except Exception as e:
        log(f"  [WARN] Download error: {str(e)[:50]}")
    return None


# Global Playwright browser instance for CDN-blocked sources
_playwright_browser = None
_playwright_context = None

def get_playwright_browser():
    """Get or create a Playwright browser instance."""
    global _playwright_browser, _playwright_context
    if _playwright_browser is None:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        _playwright_browser = pw.chromium.launch(headless=True)
        _playwright_context = _playwright_browser.new_context()
    return _playwright_context

def close_playwright_browser():
    """Close the Playwright browser."""
    global _playwright_browser, _playwright_context
    if _playwright_context:
        _playwright_context.close()
        _playwright_context = None
    if _playwright_browser:
        _playwright_browser.close()
        _playwright_browser = None

def download_image_playwright(url, timeout=15000):
    """Download image using Playwright browser (bypasses CDN blocks)."""
    try:
        context = get_playwright_browser()
        page = context.new_page()
        response = page.goto(url, timeout=timeout, wait_until='load')
        if response and response.ok:
            content = response.body()
            page.close()
            return content
        page.close()
    except Exception as e:
        log(f"  [WARN] Playwright download error: {str(e)[:50]}")
    return None


def process_listing(listing, extractor, use_playwright=False):
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
        source = listing['source']
        url = transform_url(listing['floorplan_url'])  # Transform proxied URLs first

        # Foxtons CDN blocks GitHub Actions IPs - use Playwright
        # Chestertons works with HTTP after URL transformation (direct CDN URL)
        if use_playwright or source == 'foxtons':
            image_bytes = download_image_playwright(url)
        else:
            image_bytes = download_image(url)
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

    conn, db_type = get_connection()
    cursor = conn.cursor()
    placeholder = get_param_placeholder(db_type)
    updated = 0

    for r in results:
        if not r['success']:
            continue

        updates = []
        params = []

        if r.get('sqft'):
            updates.append(f'size_sqft = {placeholder}')
            params.append(r['sqft'])

        if r.get('floor_count'):
            updates.append(f'floor_count = {placeholder}')
            params.append(r['floor_count'])

        if r.get('property_levels'):
            updates.append(f'property_levels = {placeholder}')
            params.append(r['property_levels'])

        if r.get('floor_data'):
            fd = r['floor_data']
            for flag in ['has_basement', 'has_lower_ground', 'has_ground', 'has_mezzanine',
                         'has_first_floor', 'has_second_floor', 'has_third_floor',
                         'has_fourth_plus', 'has_roof_terrace']:
                if fd.get(flag):
                    updates.append(f'{flag} = {placeholder}')
                    params.append(fd[flag])

        if updates:
            params.append(r['id'])
            query = f"UPDATE listings SET {', '.join(updates)} WHERE id = {placeholder}"
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
    parser.add_argument('--timeout', '-t', type=int, default=120, help='Timeout per listing (seconds)')
    args = parser.parse_args()

    # Check database type
    db_type = 'postgres' if get_postgres_url() else 'sqlite'

    log("=" * 70)
    log("OCR ENRICHMENT - FLOORPLAN DATA EXTRACTION")
    log("=" * 70)
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Database: {db_type.upper()}")
    if args.source:
        log(f"Source: {args.source}")
    if args.limit:
        log(f"Limit: {args.limit}")
    log(f"Workers: {args.workers}")
    log(f"Timeout per listing: {args.timeout}s")
    if args.dry_run:
        log("Mode: DRY RUN (no database updates)")
    log("")

    # Get listings
    log("Querying database for listings needing OCR...")
    listings = get_listings_needing_ocr(args.source, args.limit)
    log(f"Found {len(listings)} listings needing OCR")

    if not listings:
        log("Nothing to process!")
        return

    # Show breakdown by source
    by_source = {}
    for l in listings:
        by_source[l['source']] = by_source.get(l['source'], 0) + 1
    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        log(f"  {source}: {count}")
    log("")

    # Initialize extractor
    log("Initializing FloorplanExtractor...")
    extractor = FloorplanExtractor()
    log("FloorplanExtractor ready.")

    # Split listings by whether they need Playwright (CDN-blocked sources)
    # Foxtons CDN blocks GitHub Actions IPs - needs Playwright
    # Chestertons works with HTTP after URL transformation
    playwright_sources = {'foxtons'}
    playwright_listings = [l for l in listings if l['source'] in playwright_sources]
    regular_listings = [l for l in listings if l['source'] not in playwright_sources]

    if playwright_listings:
        log(f"Note: {len(playwright_listings)} Foxtons listings will use Playwright (CDN workaround)")

    # Process with thread pool
    results = []
    start_time = time.time()
    timeouts = 0

    # Process regular sources with ThreadPoolExecutor
    if regular_listings:
        log(f"Processing {len(regular_listings)} regular listings with {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_listing, l, extractor, False): l for l in regular_listings}

            completed = 0
            successful = 0
            for future in as_completed(futures):
                completed += 1
                listing = futures[future]
                try:
                    result = future.result(timeout=args.timeout)
                    results.append(result)
                    if result['success']:
                        successful += 1
                except TimeoutError:
                    timeouts += 1
                    results.append({'id': listing['id'], 'property_id': listing['property_id'],
                                    'source': listing['source'], 'success': False, 'error': 'timeout'})
                except Exception as e:
                    results.append({'id': listing['id'], 'property_id': listing['property_id'],
                                    'source': listing['source'], 'success': False, 'error': str(e)[:100]})

                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    log(f"Progress (regular): {completed}/{len(regular_listings)} ({successful} success) | {rate:.1f}/sec")

    # Process Playwright sources sequentially (not thread-safe)
    if playwright_listings:
        log(f"Processing {len(playwright_listings)} Playwright listings sequentially...")
        pw_start = time.time()
        pw_success = 0
        for i, listing in enumerate(playwright_listings):
            try:
                result = process_listing(listing, extractor, use_playwright=True)
                results.append(result)
                if result['success']:
                    pw_success += 1
            except Exception as e:
                results.append({'id': listing['id'], 'property_id': listing['property_id'],
                                'source': listing['source'], 'success': False, 'error': str(e)[:100]})

            if (i + 1) % 10 == 0:
                elapsed = time.time() - pw_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                log(f"Progress (Playwright): {i+1}/{len(playwright_listings)} ({pw_success} success) | {rate:.1f}/sec")

        # Cleanup Playwright
        close_playwright_browser()
        log("Playwright browser closed.")

    # Summary
    elapsed = time.time() - start_time
    log("")
    log("=" * 70)
    log("RESULTS")
    log("=" * 70)
    log(f"Total processed: {len(results)}")
    log(f"Successful OCR: {sum(1 for r in results if r['success'])}")
    log(f"Download failed: {sum(1 for r in results if r.get('error') == 'download_failed')}")
    log(f"OCR failed: {sum(1 for r in results if r.get('error') == 'ocr_failed')}")
    log(f"Timeouts: {sum(1 for r in results if r.get('error') == 'timeout')}")
    log(f"Other errors: {sum(1 for r in results if r.get('error') and r['error'] not in ['download_failed', 'ocr_failed', 'timeout'])}")
    log("")

    # Count extractions
    sqft_extracted = sum(1 for r in results if r.get('sqft'))
    floor_count_extracted = sum(1 for r in results if r.get('floor_count'))
    levels_extracted = sum(1 for r in results if r.get('property_levels'))

    log(f"Sqft extracted: {sqft_extracted}")
    log(f"Floor count extracted: {floor_count_extracted}")
    log(f"Property levels extracted: {levels_extracted}")
    log("")

    # Show sample results
    log("Sample extractions:")
    for r in results[:5]:
        if r['success']:
            log(f"  {r['property_id']}: sqft={r.get('sqft')}, floors={r.get('floor_count')}, levels={r.get('property_levels')}")
    log("")

    # Update database
    if not args.dry_run:
        log("Updating database...")
        updated = update_database(results)
        log(f"Database updated: {updated} records")
    else:
        log("DRY RUN - no database updates made")

    log(f"\nDuration: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
