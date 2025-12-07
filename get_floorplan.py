"""
Click FLOORPLANS tab and download the floorplan image
"""
import asyncio
from playwright.async_api import async_playwright
import os
import requests

async def get_floorplan():
    print("Getting floorplan from John D Wood...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()
        
        # Go to property page
        await page.goto('https://www.johndwood.co.uk/properties/20639639/lettings/SHL250035', 
                       wait_until='load', timeout=60000)
        await page.wait_for_timeout(3000)
        
        # Accept cookies if present
        try:
            accept_btn = await page.query_selector('text="ACCEPT ALL"')
            if accept_btn:
                await accept_btn.click()
                await page.wait_for_timeout(1000)
        except:
            pass
        
        print("Page loaded. Looking for FLOORPLANS tab...")
        
        # Click on FLOORPLANS tab
        floorplans_tab = await page.query_selector('text="FLOORPLANS"')
        if floorplans_tab:
            print("Found FLOORPLANS tab, clicking...")
            await floorplans_tab.click()
            await page.wait_for_timeout(3000)
            
            # Take screenshot after clicking
            await page.screenshot(path='/tmp/jdw_floorplan_view.png')
            print("Screenshot saved: /tmp/jdw_floorplan_view.png")
            
            # Find the floorplan image
            floorplan_info = await page.evaluate('''() => {
                // Get all visible images in the main content area
                const imgs = Array.from(document.querySelectorAll('img'));
                const visible = imgs.filter(img => {
                    const rect = img.getBoundingClientRect();
                    return rect.width > 200 && rect.height > 200;
                });
                
                return visible.map(img => ({
                    src: img.src,
                    alt: img.alt,
                    width: img.naturalWidth,
                    height: img.naturalHeight,
                    displayWidth: img.getBoundingClientRect().width,
                    displayHeight: img.getBoundingClientRect().height
                }));
            }''')
            
            print(f"\nVisible images after clicking FLOORPLANS: {len(floorplan_info)}")
            for img in floorplan_info[:5]:
                print(f"  {img['src'][:80]}...")
                print(f"    Size: {img['width']}x{img['height']}, Display: {img['displayWidth']:.0f}x{img['displayHeight']:.0f}")
            
            # Try to get the largest image (likely the floorplan)
            largest = max(floorplan_info, key=lambda x: x['displayWidth'] * x['displayHeight']) if floorplan_info else None
            if largest:
                print(f"\n*** LARGEST IMAGE (likely floorplan) ***")
                print(f"    URL: {largest['src']}")
                print(f"    Size: {largest['width']}x{largest['height']}")
                
                # Download the floorplan image
                print("\nDownloading floorplan image...")
                response = requests.get(largest['src'])
                if response.status_code == 200:
                    with open('/tmp/jdw_floorplan.jpg', 'wb') as f:
                        f.write(response.content)
                    print(f"Saved to /tmp/jdw_floorplan.jpg ({len(response.content)/1024:.1f} KB)")
        else:
            print("FLOORPLANS tab not found")
            
            # Check what tabs are available
            tabs = await page.evaluate('''() => {
                const tabEls = document.querySelectorAll('button, a, [role="tab"]');
                return Array.from(tabEls)
                    .map(t => t.innerText.trim())
                    .filter(t => t.length > 0 && t.length < 20);
            }''')
            print(f"Available tabs: {tabs}")
        
        await browser.close()


if __name__ == '__main__':
    asyncio.run(get_floorplan())
