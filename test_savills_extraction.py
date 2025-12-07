#!/usr/bin/env python3
"""
Savills Floorplan Extraction Test - DRY RUN
Tests extraction logic without writing to database.
"""

import asyncio
from playwright.async_api import async_playwright
import json

# Test URLs from database
TEST_URLS = [
    ("savills_236264", "https://search.savills.com/property-detail/gbnnrenhl250073l", "London, W2 4JY"),
    ("savills_258100", "https://search.savills.com/property-detail/gbwarewfl250019l", "London, SW18 4RG"),
    ("savills_534501", "https://search.savills.com/property-detail/gbwprewsl250097l", "3 Abbey Orchard Street"),
    ("savills_539465", "https://search.savills.com/property-detail/gbwpresll050073l", "8 Dean Ryle Street"),
    ("savills_579040", "https://search.savills.com/property-detail/gbnnrelhl250830l", "149 Queensway"),
]

async def extract_floorplan(page, url, prop_id, address):
    """Test floorplan extraction on a Savills detail page."""
    print(f"\n{'='*60}")
    print(f"Testing: {prop_id} - {address}")
    print(f"URL: {url}")
    print('='*60)

    try:
        # Navigate to page
        response = await page.goto(url, wait_until='networkidle', timeout=30000)
        print(f"[STATUS] HTTP {response.status}")

        if response.status != 200:
            print(f"[ERROR] Non-200 status")
            return None

        await page.wait_for_timeout(2000)

        # Check page structure
        page_info = await page.evaluate('''() => {
            return {
                title: document.title,
                tabs: Array.from(document.querySelectorAll('[role="tab"], button, a')).map(t =>
                    (t.innerText || t.textContent || '').trim().substring(0, 30)
                ).filter(t => t.length > 0 && t.length < 30),
                imgCount: document.querySelectorAll('img').length
            };
        }''')
        print(f"[PAGE] Title: {page_info['title'][:50]}...")
        print(f"[PAGE] Found {page_info['imgCount']} images")
        print(f"[PAGE] Tabs/buttons: {page_info['tabs'][:10]}")

        # Click the PLANS button (it's a button, not a tab)
        clicked = await page.evaluate('''() => {
            const buttons = document.querySelectorAll('button');
            for (const btn of buttons) {
                const text = (btn.innerText || btn.textContent || '').trim().toLowerCase();
                // Exact match for "plans"
                if (text === 'plans') {
                    btn.click();
                    return 'plans';
                }
            }
            return null;
        }''')

        if clicked:
            print(f"[CLICK] Clicked button: '{clicked}'")
            await page.wait_for_timeout(3000)  # Wait longer for floorplan to load
        else:
            print(f"[CLICK] No 'plans' button found")

        # Search for floorplan images - Savills uses GIF files and alt text
        floorplan_data = await page.evaluate('''() => {
            const results = {
                strategy1_alt_floorplan: [],
                strategy2_gif_files: [],
                strategy3_html_gif: null,
                strategy4_all_property: []
            };

            // Strategy 1: Images with "Floorplan" in alt text (primary method)
            document.querySelectorAll('img').forEach(img => {
                const src = img.src || '';
                const alt = img.alt || '';
                if (src.includes('assets.savills.com/properties/') &&
                    alt.toLowerCase().includes('floorplan')) {
                    results.strategy1_alt_floorplan.push({
                        src: src,
                        alt: alt.substring(0, 80)
                    });
                }
            });

            // Strategy 2: GIF files from properties (floorplans are GIFs, photos are JPGs)
            document.querySelectorAll('img').forEach(img => {
                const src = img.src || '';
                if (src.includes('assets.savills.com/properties/') &&
                    src.toLowerCase().endsWith('.gif')) {
                    results.strategy2_gif_files.push(src);
                }
            });

            // Strategy 3: Regex for GIF URLs in HTML
            const html = document.documentElement.innerHTML;
            const match = html.match(/https:\/\/assets\.savills\.com\/properties\/[A-Z0-9]+\/[^"'\s]+\.gif/i);
            results.strategy3_html_gif = match ? match[0] : null;

            // Strategy 4: All property images for debugging
            document.querySelectorAll('img').forEach(img => {
                const src = img.src || '';
                const alt = img.alt || '';
                if (src.includes('assets.savills.com/properties/')) {
                    results.strategy4_all_property.push({
                        src: src.substring(0, 80),
                        alt: alt.substring(0, 50),
                        isGif: src.toLowerCase().endsWith('.gif')
                    });
                }
            });

            return results;
        }''')

        print(f"\n[RESULTS]")
        print(f"  Strategy 1 (alt=Floorplan): {len(floorplan_data['strategy1_alt_floorplan'])} found")
        for item in floorplan_data['strategy1_alt_floorplan'][:2]:
            print(f"    -> {item['src'][:70]}... alt='{item['alt'][:30]}'")

        print(f"  Strategy 2 (GIF files): {len(floorplan_data['strategy2_gif_files'])} found")
        for url in floorplan_data['strategy2_gif_files'][:3]:
            print(f"    -> {url[:80]}...")

        print(f"  Strategy 3 (HTML regex GIF): {'FOUND' if floorplan_data['strategy3_html_gif'] else 'NOT FOUND'}")
        if floorplan_data['strategy3_html_gif']:
            print(f"    -> {floorplan_data['strategy3_html_gif'][:80]}...")

        print(f"  Strategy 4 (all property imgs): {len(floorplan_data['strategy4_all_property'])} found")
        for item in floorplan_data['strategy4_all_property'][:3]:
            print(f"    -> {item['src'][:60]}... isGif={item['isGif']}")

        # Priority: Alt text > GIF files > HTML regex
        best_url = None
        if floorplan_data['strategy1_alt_floorplan']:
            best_url = floorplan_data['strategy1_alt_floorplan'][0]['src']
        elif floorplan_data['strategy2_gif_files']:
            best_url = floorplan_data['strategy2_gif_files'][0]
        elif floorplan_data['strategy3_html_gif']:
            best_url = floorplan_data['strategy3_html_gif']
        # NOTE: strategy4 is for debugging only - shows ALL property images

        if best_url:
            print(f"\n[SUCCESS] True floorplan URL found!")
            print(f"  {best_url[:100]}...")
            return best_url
        else:
            print(f"\n[FAIL] No floorplan-specific URL found")
            print(f"[DEBUG] Property images exist but no floorplan naming pattern detected")

            # Deep investigation: look at page structure after clicking Plans tab
            plans_content = await page.evaluate('''() => {
                // Look for visible images in the current view
                const results = {
                    visible_imgs: [],
                    data_attrs: [],
                    img_containers: []
                };

                // Get images that might be in a plans/floorplan section
                document.querySelectorAll('img').forEach(img => {
                    const rect = img.getBoundingClientRect();
                    const src = img.src || '';
                    const alt = img.alt || '';
                    const classes = img.className || '';
                    // Check if image is visible
                    if (rect.width > 100 && rect.height > 100 && src.includes('savills')) {
                        results.visible_imgs.push({
                            src: src.substring(0, 80),
                            alt: alt,
                            width: rect.width,
                            height: rect.height
                        });
                    }
                });

                // Look for any data attributes that might contain floorplan info
                document.querySelectorAll('[data-floorplan], [data-plan], [data-image-type]').forEach(el => {
                    results.data_attrs.push({
                        tag: el.tagName,
                        attrs: Object.fromEntries(
                            Array.from(el.attributes).filter(a => a.name.includes('data')).map(a => [a.name, a.value.substring(0, 50)])
                        )
                    });
                });

                return results;
            }''')
            print(f"[DEBUG] Visible images: {len(plans_content['visible_imgs'])}")
            for img in plans_content['visible_imgs'][:3]:
                print(f"    {img}")
            if plans_content['data_attrs']:
                print(f"[DEBUG] Data attributes: {plans_content['data_attrs']}")
            return None

    except Exception as e:
        print(f"[ERROR] {e}")
        return None


async def main():
    print("="*60)
    print("SAVILLS FLOORPLAN EXTRACTION TEST - DRY RUN")
    print("="*60)
    print("Testing extraction logic WITHOUT writing to database")

    results = {
        'total': len(TEST_URLS),
        'success': 0,
        'failed': 0,
        'found_urls': []
    }

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        )
        page = await context.new_page()

        for prop_id, url, address in TEST_URLS:
            floorplan_url = await extract_floorplan(page, url, prop_id, address)
            if floorplan_url:
                results['success'] += 1
                results['found_urls'].append((prop_id, floorplan_url[:80]))
            else:
                results['failed'] += 1

            await asyncio.sleep(1)  # Be polite

        await browser.close()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tested: {results['total']}")
    print(f"Floorplans found: {results['success']} ({results['success']/results['total']*100:.0f}%)")
    print(f"Failed: {results['failed']}")

    if results['found_urls']:
        print(f"\nFound floorplan URLs:")
        for prop_id, url in results['found_urls']:
            print(f"  {prop_id}: {url}...")

    return results

if __name__ == '__main__':
    asyncio.run(main())
