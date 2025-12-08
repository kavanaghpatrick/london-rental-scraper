#!/usr/bin/env python3
"""
Pricing Diagnostic Script - Verify suspicious listings with Playwright

This script:
1. Queries listings with suspicious price per sqft
2. Uses Playwright to load actual pages
3. Extracts real price and size from the page
4. Compares to database values
5. Reports discrepancies and likely issues

Usage:
    python scripts/diagnose_pricing.py --source rightmove --limit 10
    python scripts/diagnose_pricing.py --issue low --limit 20
    python scripts/diagnose_pricing.py --all
"""

import sqlite3
import asyncio
import argparse
import re
import json
from pathlib import Path
from datetime import datetime

# Check playwright availability
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("WARNING: playwright not available. Install with: pip install playwright && playwright install chromium")

DB_PATH = Path(__file__).parent.parent / 'output' / 'rentals.db'


def get_suspicious_listings(source=None, issue=None, limit=50):
    """Get listings with suspicious price per sqft."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Base query
    query = '''
        SELECT id, source, property_id, price_pcm, size_sqft, bedrooms, address, url,
               ROUND(price_pcm * 1.0 / size_sqft, 2) as ppsf
        FROM listings
        WHERE is_active = 1 AND size_sqft > 0 AND price_pcm > 0
    '''
    params = []

    # Filter by issue type
    if issue == 'low':
        query += ' AND price_pcm * 1.0 / size_sqft < 2.5'
    elif issue == 'high':
        query += ' AND price_pcm * 1.0 / size_sqft > 25'
    else:
        query += ' AND (price_pcm * 1.0 / size_sqft < 2.5 OR price_pcm * 1.0 / size_sqft > 25)'

    # Filter by source
    if source:
        query += ' AND source = ?'
        params.append(source)

    query += ' ORDER BY ppsf'

    if limit:
        query += f' LIMIT {int(limit)}'

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


async def extract_page_data(page, url, source):
    """Extract price and size from page based on source."""
    result = {
        'url': url,
        'source': source,
        'page_price': None,
        'page_price_period': None,
        'page_size': None,
        'page_title': None,
        'raw_price_text': None,
        'raw_size_text': None,
        'error': None,
    }

    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=30000)
        await page.wait_for_timeout(3000)  # Extra time for JS rendering

        # Get page title
        result['page_title'] = await page.title()

        # Source-specific extraction
        if source == 'rightmove':
            await extract_rightmove(page, result)
        elif source == 'savills':
            await extract_savills(page, result)
        elif source == 'knightfrank':
            await extract_knightfrank(page, result)
        elif source == 'chestertons':
            await extract_chestertons(page, result)
        elif source == 'foxtons':
            await extract_foxtons(page, result)
        elif source == 'johndwood':
            await extract_johndwood(page, result)
        else:
            result['error'] = f'Unknown source: {source}'

    except Exception as e:
        result['error'] = str(e)[:200]

    return result


async def extract_rightmove(page, result):
    """Extract from Rightmove page."""
    # Try to get NEXT_DATA first
    try:
        next_data = await page.evaluate('''() => {
            const el = document.getElementById('__NEXT_DATA__');
            return el ? el.textContent : null;
        }''')
        if next_data:
            data = json.loads(next_data)
            props = data.get('props', {}).get('pageProps', {}).get('propertyData', {})

            # Price
            prices = props.get('prices', {})
            primary = prices.get('primaryPrice', '')
            result['raw_price_text'] = primary

            # Parse price and period
            match = re.search(r'£([\d,]+)\s*(pcm|pw|per\s*month|per\s*week)?', primary, re.I)
            if match:
                result['page_price'] = int(match.group(1).replace(',', ''))
                period = match.group(2) or ''
                result['page_price_period'] = 'pw' if 'week' in period.lower() or period.lower() == 'pw' else 'pcm'

            # Size
            size_data = props.get('sizings', [])
            for s in size_data:
                if s.get('unit') == 'sqft':
                    result['page_size'] = s.get('minimumSize')
                    result['raw_size_text'] = f"{s.get('minimumSize')} sqft"
                    break
            return
    except:
        pass

    # Fallback: parse from HTML
    try:
        price_text = await page.inner_text('p._1gfnqJ3Vtd1z40MlC0MzXu')
        result['raw_price_text'] = price_text
    except:
        pass


async def extract_savills(page, result):
    """Extract from Savills page."""
    try:
        # Wait for content
        await page.wait_for_selector('.sv-property-detail', timeout=10000)

        # Price
        price_el = await page.query_selector('.sv-property-price')
        if price_el:
            price_text = await price_el.inner_text()
            result['raw_price_text'] = price_text.strip()

            # Parse: "£5,594 Per Week" or "£5,594 Per Month"
            match = re.search(r'£([\d,]+)\s*(Per\s*Week|Per\s*Month|pw|pcm)?', price_text, re.I)
            if match:
                result['page_price'] = int(match.group(1).replace(',', ''))
                period = match.group(2) or ''
                result['page_price_period'] = 'pw' if 'week' in period.lower() else 'pcm'

        # Size - usually in key details
        size_text = await page.evaluate('''() => {
            const els = document.querySelectorAll('.sv-key-detail, .sv-property-detail-feature');
            for (const el of els) {
                const text = el.textContent;
                if (text.match(/sq\\s*ft/i)) {
                    return text;
                }
            }
            return null;
        }''')
        if size_text:
            result['raw_size_text'] = size_text.strip()
            match = re.search(r'([\d,]+)\s*sq\s*ft', size_text, re.I)
            if match:
                result['page_size'] = int(match.group(1).replace(',', ''))
    except Exception as e:
        result['error'] = str(e)[:100]


async def extract_knightfrank(page, result):
    """Extract from Knight Frank page."""
    try:
        await page.wait_for_selector('.kf-property-overview', timeout=10000)

        # Price
        price_el = await page.query_selector('.kf-price, .kf-property-price')
        if price_el:
            price_text = await price_el.inner_text()
            result['raw_price_text'] = price_text.strip()

            match = re.search(r'£([\d,]+)\s*(pw|pcm|per\s*week|per\s*month)?', price_text, re.I)
            if match:
                result['page_price'] = int(match.group(1).replace(',', ''))
                period = match.group(2) or ''
                result['page_price_period'] = 'pw' if 'week' in period.lower() or period.lower() == 'pw' else 'pcm'

        # Size
        size_text = await page.evaluate('''() => {
            const els = document.querySelectorAll('.kf-property-detail, .kf-key-info');
            for (const el of els) {
                const text = el.textContent;
                if (text.match(/sq\\s*ft/i)) {
                    return text;
                }
            }
            return null;
        }''')
        if size_text:
            result['raw_size_text'] = size_text.strip()
            match = re.search(r'([\d,]+)\s*sq\s*ft', size_text, re.I)
            if match:
                result['page_size'] = int(match.group(1).replace(',', ''))
    except Exception as e:
        result['error'] = str(e)[:100]


async def extract_chestertons(page, result):
    """Extract from Chestertons page."""
    try:
        await page.wait_for_selector('.property-detail', timeout=10000)

        # Price
        price_el = await page.query_selector('.property-price, .price')
        if price_el:
            price_text = await price_el.inner_text()
            result['raw_price_text'] = price_text.strip()

            match = re.search(r'£([\d,]+)\s*(pw|pcm|per\s*week|per\s*month)?', price_text, re.I)
            if match:
                result['page_price'] = int(match.group(1).replace(',', ''))
                period = match.group(2) or ''
                result['page_price_period'] = 'pw' if 'week' in period.lower() or period.lower() == 'pw' else 'pcm'

        # Size
        size_text = await page.evaluate('''() => {
            const text = document.body.textContent;
            const match = text.match(/([\d,]+)\s*sq\s*ft/i);
            return match ? match[0] : null;
        }''')
        if size_text:
            result['raw_size_text'] = size_text.strip()
            match = re.search(r'([\d,]+)', size_text)
            if match:
                result['page_size'] = int(match.group(1).replace(',', ''))
    except Exception as e:
        result['error'] = str(e)[:100]


async def extract_foxtons(page, result):
    """Extract from Foxtons page."""
    try:
        # Foxtons uses __NEXT_DATA__
        next_data = await page.evaluate('''() => {
            const el = document.getElementById('__NEXT_DATA__');
            return el ? el.textContent : null;
        }''')
        if next_data:
            data = json.loads(next_data)
            props = data.get('props', {}).get('pageProps', {}).get('property', {})

            # Price
            rent = props.get('rent', {})
            price = rent.get('price', 0)
            period = rent.get('period', 'pcm')
            result['page_price'] = price
            result['page_price_period'] = period
            result['raw_price_text'] = f"£{price} {period}"

            # Size
            size = props.get('size', {})
            sqft = size.get('sqft', 0)
            result['page_size'] = sqft
            result['raw_size_text'] = f"{sqft} sqft"
    except Exception as e:
        result['error'] = str(e)[:100]


async def extract_johndwood(page, result):
    """Extract from John D Wood page."""
    try:
        await page.wait_for_selector('.property-detail, .property-info', timeout=10000)

        # Get all text content
        body_text = await page.evaluate('() => document.body.textContent')

        # Price
        price_match = re.search(r'£([\d,]+)\s*(pw|pcm|per\s*week|per\s*month|weekly|monthly)?', body_text, re.I)
        if price_match:
            result['page_price'] = int(price_match.group(1).replace(',', ''))
            period = price_match.group(2) or ''
            # John D Wood defaults to weekly
            if 'month' in period.lower() or period.lower() == 'pcm':
                result['page_price_period'] = 'pcm'
            else:
                result['page_price_period'] = 'pw'
            result['raw_price_text'] = price_match.group(0)

        # Size
        size_match = re.search(r'([\d,]+)\s*sq\s*ft', body_text, re.I)
        if size_match:
            result['page_size'] = int(size_match.group(1).replace(',', ''))
            result['raw_size_text'] = size_match.group(0)
    except Exception as e:
        result['error'] = str(e)[:100]


async def diagnose_listings(listings, headless=True):
    """Run diagnostics on listings using Playwright."""
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = await context.new_page()

        for i, listing in enumerate(listings):
            print(f"[{i+1}/{len(listings)}] Checking {listing['source']}: {listing['property_id']}...")

            result = await extract_page_data(page, listing['url'], listing['source'])
            result['db_price'] = listing['price_pcm']
            result['db_size'] = listing['size_sqft']
            result['db_ppsf'] = listing['ppsf']
            result['property_id'] = listing['property_id']
            result['address'] = listing['address']

            # Analyze discrepancy
            if result['page_price'] and result['page_price_period']:
                expected_pcm = result['page_price']
                if result['page_price_period'] == 'pw':
                    expected_pcm = int(result['page_price'] * 52 / 12)
                    result['diagnosis'] = 'WEEKLY_AS_MONTHLY' if abs(listing['price_pcm'] - result['page_price']) < 100 else 'PRICE_CORRECT'
                else:
                    result['diagnosis'] = 'PRICE_CORRECT' if abs(listing['price_pcm'] - expected_pcm) < 100 else 'PRICE_MISMATCH'
                result['expected_pcm'] = expected_pcm
            else:
                result['diagnosis'] = 'EXTRACTION_FAILED'
                result['expected_pcm'] = None

            # Check size
            if result['page_size']:
                if abs(listing['size_sqft'] - result['page_size']) > 50:
                    result['size_diagnosis'] = 'SIZE_MISMATCH'
                else:
                    result['size_diagnosis'] = 'SIZE_CORRECT'
            else:
                result['size_diagnosis'] = 'SIZE_NOT_FOUND'

            results.append(result)

            # Rate limiting
            await page.wait_for_timeout(1000)

        await browser.close()

    return results


def print_report(results):
    """Print diagnostic report."""
    print("\n" + "=" * 80)
    print("PRICING DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"Total listings checked: {len(results)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Group by diagnosis
    by_diagnosis = {}
    for r in results:
        diag = r.get('diagnosis', 'UNKNOWN')
        if diag not in by_diagnosis:
            by_diagnosis[diag] = []
        by_diagnosis[diag].append(r)

    print("SUMMARY BY DIAGNOSIS:")
    print("-" * 40)
    for diag, items in sorted(by_diagnosis.items()):
        print(f"  {diag}: {len(items)}")
    print()

    # Detail each issue type
    for diag in ['WEEKLY_AS_MONTHLY', 'PRICE_MISMATCH', 'SIZE_MISMATCH']:
        items = by_diagnosis.get(diag, [])
        if items:
            print(f"\n{diag} ({len(items)} listings):")
            print("-" * 60)
            for r in items[:10]:  # Show first 10
                print(f"  {r['source']} | {r['property_id']}")
                print(f"    DB: £{r['db_price']:,} pcm, {r['db_size']:,} sqft = £{r['db_ppsf']}/sqft")
                if r.get('page_price'):
                    print(f"    Page: £{r['page_price']:,} {r.get('page_price_period', '?')}")
                    if r.get('expected_pcm'):
                        print(f"    Expected PCM: £{r['expected_pcm']:,}")
                if r.get('page_size'):
                    print(f"    Page Size: {r['page_size']:,} sqft")
                print(f"    URL: {r['url']}")
                print()

    # Show extraction failures
    failures = by_diagnosis.get('EXTRACTION_FAILED', [])
    if failures:
        print(f"\nEXTRACTION FAILED ({len(failures)} listings):")
        print("-" * 60)
        for r in failures[:5]:
            print(f"  {r['source']} | {r['property_id']}: {r.get('error', 'Unknown error')[:80]}")
            print(f"    URL: {r['url']}")


async def main():
    parser = argparse.ArgumentParser(description='Diagnose pricing issues using Playwright')
    parser.add_argument('--source', '-s', help='Specific source to check')
    parser.add_argument('--issue', '-i', choices=['low', 'high'], help='Issue type: low ppsf or high ppsf')
    parser.add_argument('--limit', '-l', type=int, default=20, help='Max listings to check')
    parser.add_argument('--no-headless', action='store_true', help='Show browser window')
    parser.add_argument('--all', action='store_true', help='Check all sources')
    args = parser.parse_args()

    if not PLAYWRIGHT_AVAILABLE:
        print("ERROR: Playwright not available")
        return

    print("=" * 80)
    print("PRICING DIAGNOSTIC TOOL")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.source:
        print(f"Source: {args.source}")
    if args.issue:
        print(f"Issue type: {args.issue}")
    print(f"Limit: {args.limit}")
    print()

    # Get suspicious listings
    listings = get_suspicious_listings(args.source, args.issue, args.limit)
    print(f"Found {len(listings)} suspicious listings to check")

    if not listings:
        print("No suspicious listings found!")
        return

    # Show breakdown
    by_source = {}
    for l in listings:
        by_source[l['source']] = by_source.get(l['source'], 0) + 1
    print("By source:", dict(sorted(by_source.items(), key=lambda x: -x[1])))
    print()

    # Run diagnostics
    results = await diagnose_listings(listings, headless=not args.no_headless)

    # Print report
    print_report(results)

    # Save results to JSON
    output_path = Path(__file__).parent.parent / 'output' / 'pricing_diagnostic.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    asyncio.run(main())
