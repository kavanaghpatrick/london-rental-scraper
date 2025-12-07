#!/usr/bin/env python3
"""
Deep exploration of Savills property page structure using Playwright.
Goal: Understand exactly how floorplans are organized on the site.
"""

import asyncio
from playwright.async_api import async_playwright
import json

TEST_URL = "https://search.savills.com/property-detail/gbnnrenhl250073l"

async def main():
    print("=" * 70)
    print("SAVILLS WEBSITE DEEP EXPLORATION")
    print("=" * 70)

    async with async_playwright() as p:
        # Launch with headless for automation, but save screenshots
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        )
        page = await context.new_page()

        print(f"\n[1] Loading page: {TEST_URL}")
        await page.goto(TEST_URL, wait_until='networkidle', timeout=30000)
        await page.wait_for_timeout(3000)

        # Screenshot initial state
        await page.screenshot(path='/tmp/savills_1_initial.png', full_page=True)
        print("[1] Screenshot saved: /tmp/savills_1_initial.png")

        # Explore page structure
        print("\n[2] Analyzing page structure...")
        structure = await page.evaluate('''() => {
            const result = {
                title: document.title,
                url: window.location.href,
                tabs: [],
                images: [],
                sections: [],
                buttons: []
            };

            // Find all tab-like elements
            document.querySelectorAll('[role="tab"], [data-tab], .tab, [class*="tab"]').forEach(el => {
                result.tabs.push({
                    text: (el.innerText || el.textContent || '').trim().substring(0, 50),
                    class: el.className,
                    id: el.id,
                    tagName: el.tagName
                });
            });

            // Find all images
            document.querySelectorAll('img').forEach(img => {
                if (img.src && img.src.includes('savills')) {
                    result.images.push({
                        src: img.src.substring(0, 100),
                        alt: (img.alt || '').substring(0, 50),
                        class: img.className
                    });
                }
            });

            // Find section headers
            document.querySelectorAll('h1, h2, h3, h4').forEach(h => {
                const text = (h.innerText || h.textContent || '').trim();
                if (text.length > 0 && text.length < 100) {
                    result.sections.push(text);
                }
            });

            // Find buttons
            document.querySelectorAll('button').forEach(btn => {
                const text = (btn.innerText || btn.textContent || '').trim();
                if (text.length > 0 && text.length < 50) {
                    result.buttons.push({
                        text: text,
                        class: btn.className
                    });
                }
            });

            return result;
        }''')

        print(f"  Title: {structure['title']}")
        print(f"\n  Tabs found ({len(structure['tabs'])}):")
        for tab in structure['tabs'][:10]:
            print(f"    - {tab}")
        print(f"\n  Buttons ({len(structure['buttons'])}):")
        for btn in structure['buttons'][:15]:
            print(f"    - {btn['text'][:40]}")
        print(f"\n  Section headers: {structure['sections'][:10]}")
        print(f"\n  Images ({len(structure['images'])}):")
        for img in structure['images'][:5]:
            print(f"    - {img['src']}")

        # Look specifically for floorplan/plans tab
        print("\n[3] Looking for Plans/Floorplan tabs...")
        tabs_info = await page.evaluate('''() => {
            const result = [];
            // Look for anything clickable that might be plans
            const selectors = [
                'a', 'button', '[role="tab"]', '[data-tab]', '.tab',
                '[class*="tab"]', '[class*="nav"]', '[class*="menu"]'
            ];

            selectors.forEach(sel => {
                document.querySelectorAll(sel).forEach(el => {
                    const text = (el.innerText || el.textContent || '').toLowerCase().trim();
                    if (text.includes('plan') || text.includes('floor') || text.includes('gallery')) {
                        result.push({
                            selector: sel,
                            text: text.substring(0, 50),
                            tagName: el.tagName,
                            className: el.className,
                            href: el.href || null
                        });
                    }
                });
            });
            return result;
        }''')

        print(f"  Found {len(tabs_info)} potential plan-related elements:")
        for t in tabs_info[:10]:
            print(f"    - {t}")

        # Try to find and click "Plans" or similar
        print("\n[4] Attempting to click Plans tab...")
        click_result = await page.evaluate('''() => {
            // Try multiple strategies to find the plans tab
            const strategies = [];

            // Strategy 1: Look for exact "Plans" text in various elements
            const allElements = document.querySelectorAll('a, button, [role="tab"], span, div');
            for (const el of allElements) {
                const text = (el.innerText || el.textContent || '').trim().toLowerCase();
                // Exact match for "plans" but not "planning"
                if (text === 'plans' || text === 'floor plans' || text === 'floorplans') {
                    el.click();
                    return {clicked: true, text: text, strategy: 'exact_match'};
                }
            }

            // Strategy 2: Look in navigation areas
            const navAreas = document.querySelectorAll('nav, [role="tablist"], .nav, .tabs, .menu');
            for (const nav of navAreas) {
                const items = nav.querySelectorAll('a, button, [role="tab"]');
                for (const item of items) {
                    const text = (item.innerText || item.textContent || '').trim().toLowerCase();
                    if (text.includes('plan') && !text.includes('planning')) {
                        item.click();
                        return {clicked: true, text: text, strategy: 'nav_search'};
                    }
                }
            }

            // Strategy 3: Look for any clickable with "floor" in it
            for (const el of allElements) {
                const text = (el.innerText || el.textContent || '').trim().toLowerCase();
                if (text.includes('floor') && text.includes('plan')) {
                    el.click();
                    return {clicked: true, text: text, strategy: 'floor_plan'};
                }
            }

            return {clicked: false, strategy: 'none_found'};
        }''')

        print(f"  Click result: {click_result}")

        if click_result.get('clicked'):
            await page.wait_for_timeout(3000)
            await page.screenshot(path='/tmp/savills_2_after_click.png', full_page=True)
            print("  Screenshot saved: /tmp/savills_2_after_click.png")

            # Now look for floorplan images
            print("\n[5] Looking for floorplan images after clicking...")
            images_after = await page.evaluate('''() => {
                const images = [];
                document.querySelectorAll('img').forEach(img => {
                    const src = img.src || '';
                    const dataSrc = img.getAttribute('data-src') || '';
                    const style = window.getComputedStyle(img);
                    const visible = style.display !== 'none' && style.visibility !== 'hidden';

                    if ((src.includes('savills') || dataSrc.includes('savills')) && visible) {
                        images.push({
                            src: src.substring(0, 120),
                            dataSrc: dataSrc.substring(0, 120),
                            alt: (img.alt || '').substring(0, 80),
                            width: img.width,
                            height: img.height,
                            naturalW: img.naturalWidth,
                            naturalH: img.naturalHeight
                        });
                    }
                });
                return images;
            }''')

            print(f"  Found {len(images_after)} visible images:")
            for img in images_after:
                print(f"    - {img}")

        # Explore the gallery/slideshow structure
        print("\n[6] Exploring image gallery structure...")
        gallery_info = await page.evaluate('''() => {
            const result = {
                galleries: [],
                sliders: [],
                carousels: []
            };

            // Look for gallery containers
            document.querySelectorAll('[class*="gallery"], [class*="slider"], [class*="carousel"], [class*="slideshow"]').forEach(el => {
                const images = el.querySelectorAll('img');
                result.galleries.push({
                    className: el.className,
                    imageCount: images.length,
                    firstImage: images[0] ? images[0].src.substring(0, 80) : null
                });
            });

            // Look for swiper or similar
            document.querySelectorAll('.swiper, .swiper-container, .slick-slider').forEach(el => {
                result.sliders.push({
                    className: el.className,
                    slides: el.querySelectorAll('.swiper-slide, .slick-slide').length
                });
            });

            return result;
        }''')

        print(f"  Galleries: {gallery_info['galleries'][:5]}")
        print(f"  Sliders: {gallery_info['sliders'][:5]}")

        # Try to find the actual floorplan by examining ALL src and data-src
        print("\n[7] Deep scan for ANY floorplan references in page...")
        deep_scan = await page.evaluate('''() => {
            const html = document.documentElement.innerHTML;
            const results = {
                floorplan_urls: [],
                plan_urls: [],
                fp_urls: []
            };

            // Find all URLs that might be floorplans
            const patterns = [
                /https?:\/\/[^\s"']+floorplan[^\s"']*/gi,
                /https?:\/\/[^\s"']+_flp_[^\s"']*/gi,
                /https?:\/\/[^\s"']+_fp_[^\s"']*/gi,
                /https?:\/\/[^\s"']+floor[-_]?plan[^\s"']*/gi
            ];

            patterns.forEach((pattern, i) => {
                const matches = html.match(pattern);
                if (matches) {
                    results.floorplan_urls.push(...matches.map(m => m.substring(0, 100)));
                }
            });

            // Also check for JSON data that might contain floorplan info
            const scripts = document.querySelectorAll('script[type="application/json"], script:not([src])');
            scripts.forEach(script => {
                const content = script.textContent || '';
                if (content.includes('floorplan') || content.includes('floor_plan')) {
                    results.plan_urls.push('JSON with floorplan found');
                }
            });

            return results;
        }''')

        print(f"  Floorplan URLs in HTML: {deep_scan['floorplan_urls']}")
        print(f"  Plan URLs: {deep_scan['plan_urls']}")
        print(f"  FP URLs: {deep_scan['fp_urls']}")

        # Final screenshot
        await page.screenshot(path='/tmp/savills_3_final.png', full_page=True)
        print("\n[8] Final screenshot saved: /tmp/savills_3_final.png")

        # Get full HTML for offline analysis
        html_content = await page.content()
        with open('/tmp/savills_page.html', 'w') as f:
            f.write(html_content)
        print("[8] Full HTML saved: /tmp/savills_page.html")

        await browser.close()

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)
    print("Screenshots saved to /tmp/savills_*.png")
    print("HTML saved to /tmp/savills_page.html")
    print("\nNext: Open screenshots or HTML to understand the actual page structure")

if __name__ == '__main__':
    asyncio.run(main())
