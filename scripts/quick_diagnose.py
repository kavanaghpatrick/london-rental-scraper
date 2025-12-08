#!/usr/bin/env python3
"""Quick diagnostic - check more URLs to understand pricing issues."""

import asyncio
import re
from playwright.async_api import async_playwright

URLS = [
    # Batch 2 - More LOW PPSF samples
    # Chestertons
    ("chestertons", "https://www.chestertons.co.uk/properties/21236397/lettings/PUL190024", 3750, 1993, 1.88),
    ("chestertons", "https://www.chestertons.co.uk/properties/20441946/lettings/SVL100020", 17500, 7964, 2.20),

    # Foxtons LOW
    ("foxtons", "https://www.foxtons.co.uk/properties/chpk2536664", 2709, 1106, 2.45),
    ("foxtons", "https://www.foxtons.co.uk/properties/nhgt0051461", 2500, 1012, 2.47),

    # More Savills
    ("savills", "https://search.savills.com/property-detail/gbnnrelhl250830l", 1330, 722, 1.84),

    # More Knight Frank
    ("knightfrank", "https://www.knightfrank.co.uk/properties/residential/to-let/shalstone-road-london-sw14/rmq012016560", 3250, 1926, 1.69),

    # HIGH PPSF - Foxtons
    ("foxtons", "https://www.foxtons.co.uk/properties/chpk5265532", 19760, 100, 197.6),

    # Savills HIGH
    ("savills", "https://search.savills.com/property-detail/gbpirephl250047l", 19002, 169, 112.44),
]

async def check_url(page, source, url, db_price, db_sqft, db_ppsf):
    """Check a single URL and extract price/size from page."""
    print(f"\n{'='*60}")
    print(f"Source: {source}")
    print(f"URL: {url}")
    print(f"DB: £{db_price:,} pcm, {db_sqft:,} sqft = £{db_ppsf}/sqft")

    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=30000)
        await page.wait_for_timeout(3000)

        # Get page text
        body_text = await page.evaluate('() => document.body.textContent')

        # Find all price patterns
        price_matches = re.findall(r'£([\d,]+)\s*(pw|pcm|per\s*week|per\s*month|weekly|monthly)?', body_text, re.I)

        # Find size patterns
        size_matches = re.findall(r'([\d,]+)\s*(?:sq\.?\s*ft\.?|square\s*feet?)', body_text, re.I)

        print(f"\nPrice patterns found on page:")
        for match in price_matches[:8]:
            price = int(match[0].replace(',', ''))
            period = match[1] or 'unknown'
            if price >= 500 and price <= 200000:
                print(f"  £{price:,} {period}")

        print(f"\nSize patterns found on page:")
        for match in size_matches[:5]:
            sqft = int(match.replace(',', ''))
            if sqft >= 100 and sqft <= 50000:
                print(f"  {sqft:,} sq ft")

        # Check for weekly indicator
        if 'per week' in body_text.lower() or ' pw' in body_text.lower() or 'weekly' in body_text.lower():
            print("\n⚠️  PAGE SHOWS WEEKLY PRICING")

        # Diagnosis
        if size_matches:
            page_sqft = int(size_matches[0].replace(',', ''))
            if page_sqft != db_sqft:
                pct_diff = abs(page_sqft - db_sqft) / max(page_sqft, db_sqft) * 100
                if pct_diff > 10:
                    print(f"\n🔴 SIZE MISMATCH: DB has {db_sqft:,}, page shows {page_sqft:,} ({pct_diff:.0f}% diff)")

    except Exception as e:
        print(f"Error: {e}")

async def main():
    print("QUICK PRICING DIAGNOSTIC - BATCH 2")
    print("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        )
        page = await context.new_page()

        for source, url, db_price, db_sqft, db_ppsf in URLS:
            await check_url(page, source, url, db_price, db_sqft, db_ppsf)
            await page.wait_for_timeout(1000)

        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
