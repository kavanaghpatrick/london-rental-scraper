#!/usr/bin/env python3
"""
Verify outlier listings using Playwright to check actual prices on websites.
"""

import asyncio
import sqlite3
import json
from playwright.async_api import async_playwright

# Most suspicious listings to verify
OUTLIERS = [
    # Price issues - suspiciously low
    {"id": "knightfrank_w1k_0", "url": "https://www.knightfrank.co.uk/properties/residential/to-let/park-lane-mayfair-london-w1k/maq7700160619", "reason": "£0/m - broken data?", "expected": "Should have real price"},
    {"id": "rightmove_sw7_250", "url": "https://www.rightmove.co.uk/properties/168224336#/?channel=RES_LET", "reason": "£250/m SW7 - weekly price?", "expected": "Check if weekly"},
    {"id": "rightmove_sw10_498", "url": "https://www.rightmove.co.uk/properties/169748237#/?channel=RES_LET", "reason": "£498/m SW10 - weekly price?", "expected": "Check if weekly"},
    {"id": "savills_sw1v_850", "url": "https://search.savills.com/property-detail/gbwprewsl160058l", "reason": "£850/m 3BR SW1V - way too cheap", "expected": "Verify price"},
    {"id": "chestertons_sw7_1400", "url": "https://www.chestertons.co.uk/properties/21094456/lettings/TYS230006", "reason": "£1400/m 1BR SW7 - weekly price?", "expected": "Check if weekly"},

    # Sqft issues - impossibly small
    {"id": "savills_47sqft", "url": "https://search.savills.com/property-detail/uk006511537", "reason": "47 sqft for 4BR - impossible", "expected": "Check actual sqft"},
    {"id": "rightmove_134sqft", "url": "https://www.rightmove.co.uk/properties/161860916#/?channel=RES_LET", "reason": "134 sqft for 4BR - impossible", "expected": "Check actual sqft"},
    {"id": "rightmove_200sqft_5br", "url": "https://www.rightmove.co.uk/properties/169974521#/?channel=RES_LET", "reason": "200 sqft for 5BR - impossible", "expected": "Check actual sqft"},
    {"id": "rightmove_231sqft_3br", "url": "https://www.rightmove.co.uk/properties/167098589#/?channel=RES_LET", "reason": "231 sqft for 3BR - impossible", "expected": "Check actual sqft"},

    # Sqft issues - suspiciously large
    {"id": "savills_11105sqft", "url": "https://search.savills.com/property-detail/gbnellwfl240044l", "reason": "11,105 sqft - verify if correct", "expected": "Check if sqm converted wrong"},

    # High price per sqft
    {"id": "rightmove_809sqft_32k", "url": "https://www.rightmove.co.uk/properties/167461922#/?channel=RES_LET", "reason": "£32.5k/m for 809sqft - verify", "expected": "Check price and sqft"},
]


async def verify_listing(page, outlier):
    """Verify a single listing."""
    url = outlier['url']
    result = {
        'id': outlier['id'],
        'url': url,
        'reason': outlier['reason'],
        'status': 'unknown',
        'findings': {}
    }

    try:
        print(f"\n{'='*60}")
        print(f"Checking: {outlier['id']}")
        print(f"URL: {url}")
        print(f"Reason: {outlier['reason']}")

        await page.goto(url, wait_until='domcontentloaded', timeout=30000)
        await page.wait_for_timeout(3000)  # Wait for JS to load

        # Extract price based on site
        if 'rightmove' in url:
            # Rightmove price extraction
            price_el = await page.query_selector('[data-testid="price"]')
            if not price_el:
                price_el = await page.query_selector('.propertyHeaderPrice')
            if not price_el:
                price_el = await page.query_selector('[class*="price"]')

            price_text = await price_el.inner_text() if price_el else "NOT FOUND"

            # Get sqft
            sqft_text = "NOT FOUND"
            content = await page.content()
            import re
            sqft_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*sq\s*ft', content, re.I)
            if sqft_match:
                sqft_text = sqft_match.group(0)

            # Get bedrooms
            beds_el = await page.query_selector('[data-testid="beds-label"]')
            if not beds_el:
                beds_match = re.search(r'(\d+)\s*bed', content, re.I)
                beds_text = beds_match.group(0) if beds_match else "NOT FOUND"
            else:
                beds_text = await beds_el.inner_text()

            result['findings'] = {
                'price': price_text,
                'sqft': sqft_text,
                'beds': beds_text
            }

        elif 'knightfrank' in url:
            price_el = await page.query_selector('.kf-detail__price')
            if not price_el:
                price_el = await page.query_selector('[class*="price"]')
            price_text = await price_el.inner_text() if price_el else "NOT FOUND"

            size_el = await page.query_selector('.kf-detail__size')
            size_text = await size_el.inner_text() if size_el else "NOT FOUND"

            result['findings'] = {
                'price': price_text,
                'sqft': size_text
            }

        elif 'savills' in url:
            # Savills
            price_el = await page.query_selector('.sv-price')
            if not price_el:
                price_el = await page.query_selector('[class*="price"]')
            price_text = await price_el.inner_text() if price_el else "NOT FOUND"

            content = await page.content()
            import re
            sqft_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*sq\s*ft', content, re.I)
            sqft_text = sqft_match.group(0) if sqft_match else "NOT FOUND"

            result['findings'] = {
                'price': price_text,
                'sqft': sqft_text
            }

        elif 'chestertons' in url:
            price_el = await page.query_selector('.property-price')
            if not price_el:
                price_el = await page.query_selector('[class*="price"]')
            price_text = await price_el.inner_text() if price_el else "NOT FOUND"

            content = await page.content()
            import re
            sqft_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*sq\s*ft', content, re.I)
            sqft_text = sqft_match.group(0) if sqft_match else "NOT FOUND"

            result['findings'] = {
                'price': price_text,
                'sqft': sqft_text
            }

        result['status'] = 'verified'
        print(f"FINDINGS: {json.dumps(result['findings'], indent=2)}")

        # Take screenshot
        screenshot_path = f"output/outlier_screenshots/{outlier['id']}.png"
        await page.screenshot(path=screenshot_path, full_page=False)
        result['screenshot'] = screenshot_path
        print(f"Screenshot saved: {screenshot_path}")

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"ERROR: {e}")

    return result


async def main():
    import os
    os.makedirs('output/outlier_screenshots', exist_ok=True)

    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Headed for debugging
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        )
        page = await context.new_page()

        for outlier in OUTLIERS:
            result = await verify_listing(page, outlier)
            results.append(result)
            await page.wait_for_timeout(2000)  # Pause between requests

        await browser.close()

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    for r in results:
        status_icon = "✓" if r['status'] == 'verified' else "✗"
        print(f"\n{status_icon} {r['id']}")
        print(f"   Reason: {r['reason']}")
        if r.get('findings'):
            for k, v in r['findings'].items():
                print(f"   {k}: {v}")
        if r.get('error'):
            print(f"   Error: {r['error']}")

    # Save results
    with open('output/outlier_verification.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to output/outlier_verification.json")


if __name__ == '__main__':
    asyncio.run(main())
