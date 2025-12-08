#!/usr/bin/env python3
"""Test script to find correct CSS selectors for Chestertons property pages."""

import asyncio
import sqlite3
from playwright.async_api import async_playwright

async def main():
    # Get a sample Chestertons URL from database
    conn = sqlite3.connect('output/rentals.db')
    cursor = conn.cursor()
    cursor.execute("SELECT url FROM listings WHERE source = 'chestertons' AND url IS NOT NULL LIMIT 5")
    urls = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"Testing {len(urls)} Chestertons URLs...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Headed to see the page
        page = await browser.new_page()

        for url in urls[:3]:  # Test first 3
            print(f"\n{'='*70}")
            print(f"Testing: {url}")
            print('='*70)

            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                await page.wait_for_timeout(5000)  # Wait for JS to load

                # Try to find description with various selectors
                selectors_to_try = [
                    # Original selectors
                    '.property-description',
                    '.description',
                    '[class*="Description"]',
                    '.pegasus-property-details',
                    '.property-details',

                    # New attempts
                    '[class*="description"]',  # case-insensitive via *=
                    '.property-summary',
                    '.property-text',
                    '.listing-description',
                    '.property-info',
                    'article p',
                    'main p',
                    '.content p',
                    '[data-testid*="description"]',
                    '.details-content',
                    '.property-content',

                    # Pegasus-specific (Chestertons uses Pegasus platform)
                    '.pegasus-description',
                    '.pegasus-text',
                    '.pegasus-content',
                    '[class*="pegasus"]',

                    # Generic text containers
                    '.text-content',
                    '.body-text',
                    '.main-content',
                ]

                found_any = False
                for selector in selectors_to_try:
                    try:
                        elements = await page.query_selector_all(selector)
                        if elements:
                            print(f"\n[FOUND] {selector}: {len(elements)} elements")
                            for i, el in enumerate(elements[:2]):  # Show first 2
                                text = await el.inner_text()
                                text = text.strip()[:200] if text else "(empty)"
                                print(f"  [{i}] {text}...")
                            found_any = True
                    except Exception as e:
                        pass

                if not found_any:
                    print("\n[WARN] No selectors matched!")

                # Try to find ALL large text blocks
                print("\n--- Searching for large text blocks ---")
                large_text = await page.evaluate('''() => {
                    const results = [];
                    const elements = document.querySelectorAll('p, div, section, article');
                    for (const el of elements) {
                        const text = el.innerText || '';
                        const className = el.className || '';
                        if (text.length > 200 && text.length < 5000) {
                            results.push({
                                tag: el.tagName,
                                class: className.substring(0, 100),
                                textPreview: text.substring(0, 150)
                            });
                        }
                    }
                    return results.slice(0, 5);
                }''')

                for i, block in enumerate(large_text):
                    print(f"\n[{i}] <{block['tag']}> class='{block['class']}'")
                    print(f"    Text: {block['textPreview']}...")

            except Exception as e:
                print(f"[ERROR] {e}")

        print("\n\nPress Enter to close browser...")
        await asyncio.sleep(30)  # Keep browser open for inspection
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
