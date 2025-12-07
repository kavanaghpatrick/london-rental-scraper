#!/usr/bin/env python3
"""
Unified Site Explorer for Rental Property Websites

Consolidates 6 explore scripts into one modular tool.
See docs/PRD_site_explorer.md for full specification.

Usage:
    python site_explorer.py savills
    python site_explorer.py dexters --floorplans
    python site_explorer.py johndwood --network --deep
    python site_explorer.py custom --url "https://example.com" --detail-pattern "/property/"
"""

import argparse
import asyncio
import json
import re
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Page, Request

# Optional rich for better console output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# =============================================================================
# Security: Network Sanitization
# =============================================================================

REDACTED_HEADERS = ['cookie', 'authorization', 'x-api-key', 'set-cookie', 'x-csrf-token']


def sanitize_request(request: Request) -> dict:
    """Redact sensitive headers from captured requests."""
    headers = {}
    for k, v in request.headers.items():
        if k.lower() in REDACTED_HEADERS:
            headers[k] = '[REDACTED]'
        else:
            headers[k] = v

    return {
        'url': request.url,
        'method': request.method,
        'resource_type': request.resource_type,
        'headers': headers
    }


# =============================================================================
# Block Detection
# =============================================================================

BLOCK_INDICATORS = [
    ('title', 'Just a moment'),           # Cloudflare
    ('title', 'Access Denied'),           # Generic block
    ('title', 'Security Check'),          # CAPTCHA
    ('title', 'Attention Required'),      # Cloudflare
    ('text', 'Please verify you are human'),
    ('text', 'Enable JavaScript and cookies'),
    ('text', 'checking your browser'),
    ('selector', '.cf-challenge'),        # Cloudflare challenge
    ('selector', '#challenge-running'),
    ('selector', '.captcha'),
]


async def detect_blocking(page: Page) -> dict:
    """Check if page is blocked by anti-bot measures."""
    title = await page.title()
    text = await page.evaluate('() => document.body?.innerText?.substring(0, 2000) || ""')

    for indicator_type, indicator_value in BLOCK_INDICATORS:
        if indicator_type == 'title' and indicator_value.lower() in title.lower():
            return {'blocked': True, 'reason': f'Title contains: {indicator_value}'}
        elif indicator_type == 'text' and indicator_value.lower() in text.lower():
            return {'blocked': True, 'reason': f'Page contains: {indicator_value}'}
        elif indicator_type == 'selector':
            if await page.query_selector(indicator_value):
                return {'blocked': True, 'reason': f'Found element: {indicator_value}'}

    return {'blocked': False, 'reason': None}


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class FrameworkInfo:
    detected: str = "Unknown"
    has_next_data: bool = False
    has_react_root: bool = False


@dataclass
class PaginationInfo:
    type: str = "unknown"  # numbered, infinite, load_more, none
    has_next: bool = False
    next_selector: Optional[str] = None
    total_pages: Optional[int] = None


@dataclass
class SearchPageInfo:
    url: str = ""
    total_listings_text: Optional[str] = None
    listing_selectors: list = field(default_factory=list)
    listing_count: int = 0
    pagination: PaginationInfo = field(default_factory=PaginationInfo)
    blocking_status: dict = field(default_factory=dict)


@dataclass
class SqftInfo:
    found: bool = False
    value: Optional[str] = None
    location: str = "unknown"  # text, json, attribute


@dataclass
class FloorplanInfo:
    found: bool = False
    url: Optional[str] = None
    discovery_method: str = "unknown"  # visible, tab_click, network


@dataclass
class DetailPageInfo:
    url: str = ""
    sqft: SqftInfo = field(default_factory=SqftInfo)
    floorplan: FloorplanInfo = field(default_factory=FloorplanInfo)
    price_text: Optional[str] = None
    key_selectors: dict = field(default_factory=dict)
    blocking_status: dict = field(default_factory=dict)


@dataclass
class ExplorationResults:
    meta: dict = field(default_factory=dict)
    framework: FrameworkInfo = field(default_factory=FrameworkInfo)
    search_page: SearchPageInfo = field(default_factory=SearchPageInfo)
    detail_page: DetailPageInfo = field(default_factory=DetailPageInfo)
    network_requests: list = field(default_factory=list)
    recommendations: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / 'report.json'
        with open(report_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return report_path


# =============================================================================
# Site Strategy Base Class
# =============================================================================

class BaseSiteStrategy(ABC):
    """Base class for site-specific exploration logic."""

    name: str = "base"
    search_url: str = ""
    detail_pattern: str = r"/property/"
    alternate_urls: list = []

    async def handle_cookie_consent(self, page: Page) -> bool:
        """Override for custom cookie handling. Returns True if handled."""
        # Try common cookie consent selectors
        selectors = [
            'button:has-text("Accept")',
            'button:has-text("Accept All")',
            'button:has-text("I Accept")',
            '#onetrust-accept-btn-handler',
            '.cookie-accept',
        ]
        for selector in selectors:
            try:
                btn = await page.query_selector(selector)
                if btn:
                    await btn.click()
                    await page.wait_for_timeout(1000)
                    return True
            except:
                pass
        return False

    async def get_next_page_element(self, page: Page):
        """Override for custom pagination logic."""
        selectors = [
            'a[rel="next"]',
            '.pagination-next',
            'a:has-text("Next")',
            'button:has-text("Next")',
            '[aria-label="Next page"]',
        ]
        for selector in selectors:
            el = await page.query_selector(selector)
            if el:
                return el
        return None

    async def wait_for_listings(self, page: Page):
        """Override for custom wait conditions."""
        await page.wait_for_load_state('networkidle')

    def get_listing_selectors(self) -> list[str]:
        """Return selectors to try for listing cards."""
        return [
            'article',
            '.property-card',
            '[data-testid="property-card"]',
            '.listing-card',
            '.property-item',
            '[class*="property"]',
            '[class*="listing"]',
            'li[class*="result"]',
        ]

    def get_sqft_patterns(self) -> list[str]:
        """Regex patterns to find sqft in text."""
        return [
            r'(\d{2,5}(?:,\d{3})?)\s*sq\s*\.?\s*ft',
            r'(\d{2,5}(?:,\d{3})?)\s*sqft',
            r'(\d{2,5}(?:,\d{3})?)\s*square\s*feet',
        ]


# =============================================================================
# Site Strategies
# =============================================================================

class SavillsStrategy(BaseSiteStrategy):
    name = 'savills'
    search_url = 'https://search.savills.com/list/property-to-rent/uk'
    detail_pattern = r'/property-detail/'

    async def handle_cookie_consent(self, page: Page) -> bool:
        try:
            btn = await page.query_selector('button:has-text("Accept")')
            if btn:
                await btn.click()
                await page.wait_for_timeout(2000)
                return True
        except:
            pass
        return False

    def get_listing_selectors(self) -> list[str]:
        return ['li.sv-results-listing__item', 'article', '.property-card']


class DextersStrategy(BaseSiteStrategy):
    name = 'dexters'
    search_url = 'https://www.dexters.co.uk/property-lettings/properties-to-rent-in-london'
    detail_pattern = r'/property-to-rent/[a-z]'
    alternate_urls = [
        'https://www.dexters.co.uk/property-search/properties-to-rent-in-london',
        'https://www.dexters.co.uk/flats-to-rent-in-london',
    ]


class JohnDWoodStrategy(BaseSiteStrategy):
    name = 'johndwood'
    search_url = 'https://www.johndwood.co.uk/properties/lettings/'
    detail_pattern = r'/properties/\d+/lettings'


class KnightFrankStrategy(BaseSiteStrategy):
    name = 'knightfrank'
    search_url = 'https://www.knightfrank.co.uk/properties/residential/to-let/london'
    detail_pattern = r'/properties/residential/to-let/'


class FoxtonsStrategy(BaseSiteStrategy):
    name = 'foxtons'
    search_url = 'https://www.foxtons.co.uk/properties-to-rent-in-london'
    detail_pattern = r'foxtons\.co\.uk/[a-z]+-\d+'  # e.g., /belgravia-123456

    def get_listing_selectors(self) -> list[str]:
        return ['[data-testid="property-card"]', 'article', '.property-card', 'a[href*="-to-rent-"]']


class ChestertonsStrategy(BaseSiteStrategy):
    name = 'chestertons'
    search_url = 'https://www.chestertons.com/en-gb/property-to-rent/london'
    detail_pattern = r'/property/'


class RightmoveStrategy(BaseSiteStrategy):
    name = 'rightmove'
    search_url = 'https://www.rightmove.co.uk/property-to-rent/find.html?locationIdentifier=REGION%5E87490'
    detail_pattern = r'rightmove\.co\.uk/properties/\d+'

    def get_listing_selectors(self) -> list[str]:
        return ['[data-testid="propertyCard"]', '.propertyCard', 'article', '[class*="propertyCard"]']


class CustomStrategy(BaseSiteStrategy):
    """Strategy for custom sites configured via CLI."""

    def __init__(self, name: str, search_url: str, detail_pattern: str,
                 listing_selectors: Optional[list] = None,
                 cookie_selector: Optional[str] = None,
                 next_selector: Optional[str] = None):
        self.name = name
        self.search_url = search_url
        self.detail_pattern = detail_pattern
        self._listing_selectors = listing_selectors or []
        self._cookie_selector = cookie_selector
        self._next_selector = next_selector

    async def handle_cookie_consent(self, page: Page) -> bool:
        if self._cookie_selector:
            try:
                btn = await page.query_selector(self._cookie_selector)
                if btn:
                    await btn.click()
                    await page.wait_for_timeout(1000)
                    return True
            except:
                pass
        return await super().handle_cookie_consent(page)

    def get_listing_selectors(self) -> list[str]:
        if self._listing_selectors:
            return self._listing_selectors
        return super().get_listing_selectors()


# Strategy Registry
SITE_STRATEGIES = {
    'savills': SavillsStrategy,
    'dexters': DextersStrategy,
    'johndwood': JohnDWoodStrategy,
    'knightfrank': KnightFrankStrategy,
    'foxtons': FoxtonsStrategy,
    'chestertons': ChestertonsStrategy,
    'rightmove': RightmoveStrategy,
}


# =============================================================================
# Main Explorer Class
# =============================================================================

@dataclass
class ExploreOptions:
    headful: bool = False
    floorplans: bool = False
    network: bool = False
    deep: bool = False
    screenshots: bool = True
    timeout: int = 60000
    output_dir: Optional[Path] = None


class SiteExplorer:
    """Main explorer class."""

    def __init__(self, strategy: BaseSiteStrategy, options: ExploreOptions):
        self.strategy = strategy
        self.options = options
        self.results = ExplorationResults()
        self.network_requests: list[dict] = []

        # Set up output directory
        if options.output_dir:
            self.output_dir = options.output_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = Path(f'exploration/{strategy.name}_{timestamp}')

    def _log(self, message: str, style: str = ""):
        """Log with rich if available, else print."""
        if RICH_AVAILABLE and console:
            console.print(message, style=style)
        else:
            print(message)

    def _setup_network_capture(self, page: Page):
        """Set up network request interception."""
        def handle_request(request: Request):
            url = request.url.lower()
            if any(ext in url for ext in ['.jpg', '.jpeg', '.png', '.webp', 'image']):
                self.network_requests.append(sanitize_request(request))

        page.on('request', handle_request)

    async def run(self) -> ExplorationResults:
        """Run the full exploration."""
        self._log(f"\n{'='*70}", style="bold")
        self._log(f"EXPLORING: {self.strategy.name.upper()}", style="bold blue")
        self._log(f"{'='*70}\n", style="bold")

        self.results.meta = {
            'site': self.strategy.name,
            'url': self.strategy.search_url,
            'explored_at': datetime.now().isoformat(),
            'options': {
                'floorplans': self.options.floorplans,
                'network': self.options.network,
                'deep': self.options.deep,
            }
        }

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=not self.options.headful)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()

            if self.options.network:
                self._setup_network_capture(page)
                self._log("Network capture enabled", style="dim")

            # Explore search page
            await self._explore_search_page(page)

            # Explore detail page
            await self._explore_detail_page(page)

            # Deep floorplan analysis if requested
            if self.options.floorplans:
                await self._deep_floorplan_analysis(page)

            # Deep URL discovery if requested
            if self.options.deep and self.strategy.alternate_urls:
                await self._try_alternate_urls(page)

            await browser.close()

        # Generate recommendations
        self._generate_recommendations()

        # Save results
        if self.options.network:
            self.results.network_requests = self.network_requests

        report_path = self.results.save(self.output_dir)
        self._log(f"\nReport saved: {report_path}", style="green")

        return self.results

    async def _explore_search_page(self, page: Page):
        """Explore the search/listing page."""
        self._log("[1/4] Loading search page...", style="bold")

        try:
            await page.goto(self.strategy.search_url, timeout=self.options.timeout)
            await self.strategy.wait_for_listings(page)
        except Exception as e:
            self._log(f"  Error loading page: {e}", style="red")
            return

        self.results.search_page.url = page.url

        # Check for blocking
        blocking = await detect_blocking(page)
        self.results.search_page.blocking_status = blocking
        if blocking['blocked']:
            self._log(f"  BLOCKED: {blocking['reason']}", style="red bold")
            return

        # Handle cookies
        cookie_handled = await self.strategy.handle_cookie_consent(page)
        if cookie_handled:
            self._log("  Cookie consent handled", style="dim")
            await page.wait_for_timeout(1000)

        # Screenshot
        if self.options.screenshots:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=self.output_dir / '01_search_page.png')

        # Detect framework
        self.results.framework = await self._detect_framework(page)
        self._log(f"  Framework: {self.results.framework.detected}", style="cyan")

        # Find listing selectors
        for selector in self.strategy.get_listing_selectors():
            try:
                elements = await page.query_selector_all(selector)
                count = len(elements)
                if count > 0:
                    self.results.search_page.listing_selectors.append({
                        'selector': selector,
                        'count': count
                    })
                    if count > self.results.search_page.listing_count:
                        self.results.search_page.listing_count = count
            except:
                pass

        self._log(f"  Listings found: {self.results.search_page.listing_count}", style="green")

        # Analyze pagination
        self.results.search_page.pagination = await self._analyze_pagination(page)
        self._log(f"  Pagination: {self.results.search_page.pagination.type}", style="cyan")

        # Get total results text
        total_text = await page.evaluate('''() => {
            const text = document.body.innerText;
            const match = text.match(/(\\d+(?:,\\d+)?)\\s*(?:properties|results|homes|listings)/i);
            return match ? match[0] : null;
        }''')
        self.results.search_page.total_listings_text = total_text

    async def _explore_detail_page(self, page: Page):
        """Find and explore a property detail page."""
        self._log("\n[2/4] Finding detail page...", style="bold")

        # Find a detail page link - escape pattern for JavaScript
        js_pattern = self.strategy.detail_pattern.replace('\\', '\\\\')
        detail_url = await page.evaluate(f'''() => {{
            const links = Array.from(document.querySelectorAll('a[href]'));
            const pattern = new RegExp('{js_pattern}');
            for (const a of links) {{
                // Skip anchor-only links but allow URLs with # params
                if (a.href === '#' || a.href.endsWith('#')) continue;
                if (pattern.test(a.href)) {{
                    // Return URL without fragment for cleaner navigation
                    return a.href.split('#')[0];
                }}
            }}
            return null;
        }}''')

        if not detail_url:
            self._log("  No detail page found", style="yellow")
            return

        self._log(f"  Navigating to: {detail_url[:60]}...", style="dim")

        try:
            await page.goto(detail_url, timeout=self.options.timeout)
            await page.wait_for_load_state('networkidle')
        except Exception as e:
            self._log(f"  Error loading detail page: {e}", style="red")
            return

        self.results.detail_page.url = detail_url

        # Check for blocking
        blocking = await detect_blocking(page)
        self.results.detail_page.blocking_status = blocking
        if blocking['blocked']:
            self._log(f"  BLOCKED: {blocking['reason']}", style="red bold")
            return

        # Screenshot
        if self.options.screenshots:
            await page.screenshot(path=self.output_dir / '02_detail_page.png')

        # Extract sqft
        page_text = await page.evaluate('() => document.body.innerText')
        for pattern in self.strategy.get_sqft_patterns():
            match = re.search(pattern, page_text, re.I)
            if match:
                self.results.detail_page.sqft = SqftInfo(
                    found=True,
                    value=match.group(0),
                    location='text'
                )
                self._log(f"  Sqft found: {match.group(0)}", style="green")
                break

        if not self.results.detail_page.sqft.found:
            self._log("  Sqft: NOT FOUND", style="yellow")

        # Find floorplan images
        floorplan = await self._find_floorplan(page)
        self.results.detail_page.floorplan = floorplan
        if floorplan.found:
            self._log(f"  Floorplan: FOUND ({floorplan.discovery_method})", style="green")
        else:
            self._log("  Floorplan: NOT FOUND", style="yellow")

        # Extract price
        price_match = re.search(r'£[\d,]+\s*(?:pcm|pw|per\s*(?:month|week))', page_text, re.I)
        if price_match:
            self.results.detail_page.price_text = price_match.group(0)
            self._log(f"  Price: {price_match.group(0)}", style="green")

    async def _find_floorplan(self, page: Page) -> FloorplanInfo:
        """Find floorplan images on the page."""
        # Check visible images first
        floorplan_img = await page.evaluate('''() => {
            const imgs = Array.from(document.querySelectorAll('img'));
            for (const img of imgs) {
                const src = (img.src || '').toLowerCase();
                const alt = (img.alt || '').toLowerCase();
                if (src.includes('floor') || alt.includes('floor') ||
                    src.includes('plan') || alt.includes('plan')) {
                    return {src: img.src, alt: img.alt};
                }
            }
            return null;
        }''')

        if floorplan_img:
            return FloorplanInfo(
                found=True,
                url=floorplan_img['src'],
                discovery_method='visible'
            )

        return FloorplanInfo(found=False)

    async def _deep_floorplan_analysis(self, page: Page):
        """Try clicking tabs to find hidden floorplans."""
        self._log("\n[3/4] Deep floorplan analysis...", style="bold")

        # Look for floorplan tabs
        tab_texts = ['Floorplan', 'Floor Plan', 'Floor plan', 'floorplan', 'Plans']
        for tab_text in tab_texts:
            try:
                tab = await page.query_selector(f'text="{tab_text}"')
                if tab:
                    await tab.click()
                    self._log(f"  Clicked tab: {tab_text}", style="dim")
                    await page.wait_for_timeout(2000)

                    # Check for newly visible floorplan
                    floorplan = await self._find_floorplan(page)
                    if floorplan.found:
                        floorplan.discovery_method = 'tab_click'
                        self.results.detail_page.floorplan = floorplan
                        self._log(f"  Floorplan found after tab click!", style="green")

                        if self.options.screenshots:
                            await page.screenshot(path=self.output_dir / '03_floorplan.png')
                        break
            except:
                pass

    async def _try_alternate_urls(self, page: Page):
        """Try alternate URLs in deep mode."""
        self._log("\n[4/4] Trying alternate URLs...", style="bold")

        for url in self.strategy.alternate_urls:
            try:
                self._log(f"  Trying: {url[:60]}...", style="dim")
                response = await page.goto(url, timeout=30000)
                if response and response.status == 200:
                    await page.wait_for_timeout(2000)
                    count = 0
                    for selector in self.strategy.get_listing_selectors():
                        elements = await page.query_selector_all(selector)
                        if len(elements) > count:
                            count = len(elements)
                    self._log(f"    Status: {response.status}, Listings: {count}", style="green")
            except Exception as e:
                self._log(f"    Error: {e}", style="red")

    async def _detect_framework(self, page: Page) -> FrameworkInfo:
        """Detect JavaScript framework used by the site."""
        info = await page.evaluate('''() => {
            return {
                has_next_data: !!document.getElementById('__NEXT_DATA__'),
                has_react_root: !!document.querySelector('[data-reactroot]') || !!document.querySelector('#__next'),
                has_angular: !!window.angular || !!document.querySelector('[ng-app]'),
                has_nuxt: !!window.__NUXT__,
            };
        }''')

        framework = FrameworkInfo(
            has_next_data=info['has_next_data'],
            has_react_root=info['has_react_root']
        )

        if info['has_next_data']:
            framework.detected = 'Next.js'
        elif info['has_react_root']:
            framework.detected = 'React'
        elif info['has_angular']:
            framework.detected = 'Angular'
        elif info['has_nuxt']:
            framework.detected = 'Nuxt.js'
        else:
            framework.detected = 'Traditional HTML'

        return framework

    async def _analyze_pagination(self, page: Page) -> PaginationInfo:
        """Analyze pagination mechanism."""
        info = PaginationInfo()

        # Check for next button
        next_el = await self.strategy.get_next_page_element(page)
        if next_el:
            info.has_next = True
            info.type = 'numbered'

        # Check for load more button
        load_more = await page.query_selector('button:has-text("Load More"), button:has-text("Show More")')
        if load_more:
            info.type = 'load_more'

        # Check for infinite scroll indicators
        infinite_scroll = await page.evaluate('''() => {
            return document.querySelector('[data-infinite-scroll]') ||
                   document.querySelector('.infinite-scroll') ||
                   document.querySelector('[class*="infinite"]');
        }''')
        if infinite_scroll:
            info.type = 'infinite'

        if not info.has_next and info.type == 'unknown':
            info.type = 'none'

        return info

    def _generate_recommendations(self):
        """Generate recommendations based on findings."""
        recs = {
            'spider_type': 'standard',
            'pagination_strategy': 'unknown',
            'sqft_source': 'unknown',
            'notes': []
        }

        # Spider type recommendation
        if self.results.framework.detected in ['Next.js', 'React']:
            recs['spider_type'] = 'playwright'
            recs['notes'].append(f'{self.results.framework.detected} detected - use Playwright spider')

        if self.results.framework.has_next_data:
            recs['notes'].append('Has __NEXT_DATA__ JSON - can extract data without rendering')

        # Pagination
        pag_type = self.results.search_page.pagination.type
        if pag_type == 'numbered':
            recs['pagination_strategy'] = 'url_params or click'
        elif pag_type == 'load_more':
            recs['pagination_strategy'] = 'click'
            recs['notes'].append('Load More button - needs Playwright')
        elif pag_type == 'infinite':
            recs['pagination_strategy'] = 'scroll'
            recs['notes'].append('Infinite scroll - needs Playwright with scroll handling')

        # Sqft source
        if self.results.detail_page.sqft.found:
            recs['sqft_source'] = self.results.detail_page.sqft.location
        elif self.results.detail_page.floorplan.found:
            recs['sqft_source'] = 'floorplan_ocr'
            recs['notes'].append('Sqft not in text - may need floorplan OCR')

        # Blocking warnings
        if self.results.search_page.blocking_status.get('blocked'):
            recs['notes'].append(f"BLOCKED on search: {self.results.search_page.blocking_status['reason']}")
        if self.results.detail_page.blocking_status.get('blocked'):
            recs['notes'].append(f"BLOCKED on detail: {self.results.detail_page.blocking_status['reason']}")

        self.results.recommendations = recs


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified Site Explorer for Rental Property Websites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python site_explorer.py savills
  python site_explorer.py dexters --floorplans
  python site_explorer.py johndwood --network --deep
  python site_explorer.py custom --url "https://example.com" --detail-pattern "/property/"
        '''
    )

    parser.add_argument('site', choices=list(SITE_STRATEGIES.keys()) + ['custom'],
                        help='Site to explore (or "custom" for custom URL)')

    # Options
    parser.add_argument('--headful', action='store_true',
                        help='Show browser window')
    parser.add_argument('--floorplans', action='store_true',
                        help='Deep floorplan analysis with tab clicking')
    parser.add_argument('--network', action='store_true',
                        help='Capture network requests for images')
    parser.add_argument('--deep', action='store_true',
                        help='Try alternate URL patterns')
    parser.add_argument('--no-screenshots', action='store_true',
                        help='Disable screenshots')
    parser.add_argument('--timeout', type=int, default=60000,
                        help='Page load timeout in ms (default: 60000)')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for results')

    # Custom site options
    parser.add_argument('--url', type=str,
                        help='Search page URL (required for custom)')
    parser.add_argument('--detail-pattern', type=str,
                        help='Regex pattern for detail page URLs (required for custom)')
    parser.add_argument('--name', type=str, default='custom',
                        help='Name for the custom site')
    parser.add_argument('--listing-selectors', type=str,
                        help='Comma-separated CSS selectors for listing cards')
    parser.add_argument('--cookie-selector', type=str,
                        help='CSS selector for cookie consent button')
    parser.add_argument('--next-selector', type=str,
                        help='CSS selector for next page button')

    return parser.parse_args()


async def main():
    args = parse_args()

    # Build strategy
    if args.site == 'custom':
        if not args.url or not args.detail_pattern:
            print("Error: --url and --detail-pattern required for custom site")
            return 1

        listing_selectors = args.listing_selectors.split(',') if args.listing_selectors else None
        strategy = CustomStrategy(
            name=args.name,
            search_url=args.url,
            detail_pattern=args.detail_pattern,
            listing_selectors=listing_selectors,
            cookie_selector=args.cookie_selector,
            next_selector=args.next_selector
        )
    else:
        strategy = SITE_STRATEGIES[args.site]()

    # Build options
    options = ExploreOptions(
        headful=args.headful,
        floorplans=args.floorplans,
        network=args.network,
        deep=args.deep,
        screenshots=not args.no_screenshots,
        timeout=args.timeout,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )

    # Run exploration
    explorer = SiteExplorer(strategy, options)
    results = await explorer.run()

    # Print summary
    print("\n" + "="*70)
    print("EXPLORATION SUMMARY")
    print("="*70)
    print(f"Site: {strategy.name}")
    print(f"Framework: {results.framework.detected}")
    print(f"Listings found: {results.search_page.listing_count}")
    print(f"Sqft found: {results.detail_page.sqft.found}")
    print(f"Floorplan found: {results.detail_page.floorplan.found}")
    print(f"Spider recommendation: {results.recommendations.get('spider_type', 'unknown')}")
    if results.recommendations.get('notes'):
        print("Notes:")
        for note in results.recommendations['notes']:
            print(f"  - {note}")

    return 0


if __name__ == '__main__':
    exit(asyncio.run(main()))
