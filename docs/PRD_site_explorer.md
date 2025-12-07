# PRD: Unified Site Explorer

## Overview

Consolidate 6 separate explore scripts into a single, modular `site_explorer.py` that can explore any rental property website to understand its structure before building a Scrapy spider.

## Problem Statement

Currently we have 6 explore scripts with 80%+ code overlap:
- `explore_savills.py` (380 lines) - Savills-specific, structured output
- `explore_sites.py` (328 lines) - Dexters/JDW baseline
- `explore_sites_v2.py` (193 lines) - Parameterized refactor
- `explore_sites_v3.py` (220 lines) - Deep URL discovery
- `explore_floorplans.py` (248 lines) - Floorplan-focused
- `explore_network.py` (83 lines) - Network request analysis

**Pain points:**
1. Adding a new site requires copy-pasting and modifying an existing script
2. Improvements to one script don't propagate to others
3. Hard to remember which script does what
4. Inconsistent output formats

## Goals

1. **Single entry point** for all site exploration
2. **Modular capabilities** activated via flags
3. **Consistent output** format (JSON + screenshots)
4. **Easy to extend** with new sites via config
5. **Reduce total code** from ~1,450 lines to ~400 lines

## Non-Goals

- Building actual Scrapy spiders (that's a separate step)
- Automated spider generation from exploration results
- GUI interface

## User Stories

1. **As a developer**, I want to explore a new rental site with one command so I can understand its structure quickly.
2. **As a developer**, I want to enable specific exploration modes (floorplans, network) so I can focus on what I need.
3. **As a developer**, I want consistent JSON output so I can compare sites easily.

## Technical Design

### CLI Interface

```bash
# Explore a known site
python site_explorer.py savills
python site_explorer.py dexters --floorplans
python site_explorer.py johndwood --network --deep

# Explore a custom site
python site_explorer.py custom \
  --url "https://example.com/rentals" \
  --detail-pattern "/property/" \
  --name "example_agency"

# Options
--output-dir ./exploration    # Where to save results (default: ./exploration/{site}/)
--headful                     # Show browser window
--floorplans                  # Deep floorplan analysis with tab clicking
--network                     # Capture network requests for images
--deep                        # Try multiple URL patterns to find listings
--timeout 60000               # Page load timeout in ms
--screenshots                 # Save screenshots at each step (default: true)
```

### Site Configuration

```python
SITE_CONFIGS = {
    'savills': {
        'name': 'Savills',
        'search_url': 'https://search.savills.com/list/property-to-rent/uk',
        'detail_pattern': r'/property/',
        'needs_cookie_consent': True,
        'wait_strategy': 'networkidle',
    },
    'dexters': {
        'name': 'Dexters',
        'search_url': 'https://www.dexters.co.uk/property-lettings/properties-to-rent-in-london',
        'detail_pattern': r'/property-to-rent/[a-z]',
        'alternate_urls': [  # For --deep mode
            'https://www.dexters.co.uk/property-search/properties-to-rent-in-london',
            'https://www.dexters.co.uk/flats-to-rent-in-london',
        ],
    },
    'johndwood': {
        'name': 'John D Wood',
        'search_url': 'https://www.johndwood.co.uk/properties/lettings/',
        'detail_pattern': r'/properties/\d+/lettings',
    },
    'knightfrank': {
        'name': 'Knight Frank',
        'search_url': 'https://www.knightfrank.co.uk/properties/to-let',
        'detail_pattern': r'/properties/residential/to-let/',
    },
    'foxtons': {
        'name': 'Foxtons',
        'search_url': 'https://www.foxtons.co.uk/properties-to-rent-in-london',
        'detail_pattern': r'/property-to-rent/',
    },
    'chestertons': {
        'name': 'Chestertons',
        'search_url': 'https://www.chestertons.com/en-gb/property-to-rent/london',
        'detail_pattern': r'/property/',
    },
    'rightmove': {
        'name': 'Rightmove',
        'search_url': 'https://www.rightmove.co.uk/property-to-rent/find.html?locationIdentifier=REGION%5E87490',
        'detail_pattern': r'/properties/\d+',
    },
}
```

### Output Structure

```
exploration/
└── {site_name}_{timestamp}/
    ├── report.json           # Main exploration results
    ├── 01_search_page.png    # Search page screenshot
    ├── 02_detail_page.png    # Detail page screenshot
    ├── 03_floorplan.png      # (if --floorplans)
    ├── network_requests.json # (if --network)
    └── raw_html/             # (optional, for debugging)
        ├── search.html
        └── detail.html
```

### report.json Schema

```json
{
  "meta": {
    "site": "savills",
    "url": "https://search.savills.com/...",
    "explored_at": "2024-12-06T11:30:00Z",
    "options": {"floorplans": true, "network": false}
  },
  "framework": {
    "detected": "React",
    "has_next_data": false,
    "has_react_root": true
  },
  "search_page": {
    "total_listings_text": "1,234 properties",
    "listing_selectors": [
      {"selector": "article", "count": 24},
      {"selector": ".property-card", "count": 24}
    ],
    "pagination": {
      "type": "numbered|infinite|load_more|none",
      "has_next": true,
      "next_selector": ".pagination-next"
    },
    "filters": {
      "inputs": [...],
      "selects": [...],
      "buttons": [...]
    }
  },
  "detail_page": {
    "url": "https://...",
    "sqft": {
      "found": true,
      "value": "1,250 sq ft",
      "location": "text|json|attribute"
    },
    "floorplan": {
      "found": true,
      "url": "https://...",
      "discovery_method": "visible|tab_click|network"
    },
    "price": {
      "found": true,
      "value": "£2,500 pcm",
      "format": "pcm|pw"
    },
    "key_selectors": {
      "address": ".property-address",
      "price": ".property-price",
      "bedrooms": ".beds-count"
    }
  },
  "recommendations": {
    "spider_type": "playwright|standard",
    "pagination_strategy": "click|url_params|scroll",
    "sqft_source": "search_card|detail_page|floorplan_ocr",
    "notes": ["Uses __NEXT_DATA__ JSON", "Requires cookie consent"]
  }
}
```

### Core Classes

```python
class SiteExplorer:
    """Main explorer class."""

    def __init__(self, config: dict, options: ExploreOptions):
        self.config = config
        self.options = options
        self.results = ExplorationResults()

    async def run(self) -> ExplorationResults:
        async with async_playwright() as p:
            browser = await self._launch_browser(p)
            page = await self._create_page(browser)

            if self.options.network:
                self._setup_network_capture(page)

            await self._explore_search_page(page)
            await self._explore_detail_page(page)

            if self.options.floorplans:
                await self._deep_floorplan_analysis(page)

            await self._save_results()
            return self.results


class ExplorationResults:
    """Structured results container."""
    meta: dict
    framework: FrameworkInfo
    search_page: SearchPageInfo
    detail_page: DetailPageInfo
    network_requests: list[str]  # if captured
    recommendations: dict

    def to_json(self) -> str: ...
    def save(self, output_dir: Path): ...
```

### Key Functions (migrated from existing scripts)

| Function | Source | Purpose |
|----------|--------|---------|
| `detect_framework()` | explore_sites_v2.py:46-52 | Detect Next.js/React/Angular |
| `find_listing_selectors()` | explore_savills.py:137-159 | Find property card CSS selectors |
| `analyze_pagination()` | explore_savills.py:275-311 | Determine pagination type |
| `extract_sqft()` | explore_sites_v2.py:95-106 | Find sqft in text with regex |
| `find_floorplan_images()` | explore_floorplans.py:56-83 | Locate floorplan images |
| `click_floorplan_tab()` | explore_floorplans.py:89-122 | Click tabs to reveal floorplans |
| `capture_network_requests()` | explore_network.py:19-26 | Hook request events |
| `handle_cookie_consent()` | explore_savills.py:44-54 | Dismiss cookie popups |

## Implementation Plan

### Phase 1: Core Structure (MVP)
1. Create `site_explorer.py` with CLI argument parsing
2. Implement `SiteExplorer` class with basic flow
3. Implement search page exploration
4. Implement detail page exploration
5. JSON output generation

### Phase 2: Advanced Features
6. Add `--network` mode (request interception)
7. Add `--floorplans` mode (tab clicking)
8. Add `--deep` mode (alternate URL discovery)

### Phase 3: Polish
9. Add all site configs
10. Comprehensive error handling
11. Update CLAUDE.md with new usage

## Success Metrics

1. **Line count**: < 500 lines total (vs 1,450 current)
2. **Single command**: Any site explorable with one command
3. **Consistent output**: All explorations produce same JSON schema
4. **Zero regression**: Can still discover all info the old scripts found

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Over-engineering | MVP first, add features only when needed |
| Breaking existing workflows | Keep old scripts until new one validated |
| Site-specific quirks | Config-driven behavior, escape hatches for custom logic |

## Open Questions

1. Should we support parallel exploration of multiple sites?
2. Should we add OCR for floorplan sqft extraction? (Future scope)
3. Should the output integrate directly with spider templates?

---

## Revision History

### Rev 2 (Post AI Review - Grok, Gemini, Codex)

**Critical fixes based on 3-way AI review:**

#### 1. Security: Network Capture Sanitization (Grok + Codex)

Network capture must redact sensitive data before writing to disk:

```python
REDACTED_HEADERS = ['cookie', 'authorization', 'x-api-key', 'set-cookie']

def sanitize_request(request) -> dict:
    """Redact sensitive headers from captured requests."""
    return {
        'url': request.url,
        'method': request.method,
        'resource_type': request.resource_type,
        # Headers with sensitive values redacted
        'headers': {k: '[REDACTED]' if k.lower() in REDACTED_HEADERS else v
                    for k, v in request.headers.items()}
    }
```

#### 2. Architecture: Strategy Pattern Instead of Dict (Gemini + Grok)

Replace static `SITE_CONFIGS` dict with Strategy classes:

```python
class BaseSiteStrategy:
    """Base class for site-specific exploration logic."""
    name: str
    search_url: str
    detail_pattern: str

    async def handle_cookie_consent(self, page) -> bool:
        """Override for custom cookie handling. Returns True if handled."""
        return False

    async def get_next_page_element(self, page):
        """Override for custom pagination logic."""
        return await page.query_selector('[rel="next"], .pagination-next')

    async def wait_for_listings(self, page):
        """Override for custom wait conditions."""
        await page.wait_for_load_state('networkidle')

    def get_listing_selectors(self) -> list[str]:
        """Return selectors to try for listing cards."""
        return ['article', '.property-card', '[data-testid="property-card"]']


class SavillsStrategy(BaseSiteStrategy):
    name = 'savills'
    search_url = 'https://search.savills.com/list/property-to-rent/uk'
    detail_pattern = r'/property/'

    async def handle_cookie_consent(self, page) -> bool:
        btn = await page.query_selector('button:has-text("Accept")')
        if btn:
            await btn.click()
            await page.wait_for_timeout(2000)
            return True
        return False


# Registry
SITE_STRATEGIES = {
    'savills': SavillsStrategy,
    'dexters': DextersStrategy,
    # ... etc
}
```

#### 3. Block Detection (Gemini + Grok)

Add explicit detection for anti-bot measures:

```python
BLOCK_INDICATORS = [
    ('title', 'Just a moment'),           # Cloudflare
    ('title', 'Access Denied'),           # Generic block
    ('title', 'Security Check'),          # CAPTCHA
    ('text', 'Please verify you are human'),
    ('text', 'Enable JavaScript'),
    ('selector', '.cf-challenge'),        # Cloudflare challenge
    ('selector', '#challenge-running'),
]

async def detect_blocking(page) -> dict:
    """Check if page is blocked by anti-bot measures."""
    title = await page.title()
    text = await page.evaluate('() => document.body.innerText.substring(0, 1000)')

    for indicator_type, indicator_value in BLOCK_INDICATORS:
        if indicator_type == 'title' and indicator_value.lower() in title.lower():
            return {'blocked': True, 'reason': f'Title contains: {indicator_value}'}
        elif indicator_type == 'text' and indicator_value.lower() in text.lower():
            return {'blocked': True, 'reason': f'Page contains: {indicator_value}'}
        elif indicator_type == 'selector':
            if await page.query_selector(indicator_value):
                return {'blocked': True, 'reason': f'Found element: {indicator_value}'}

    return {'blocked': False, 'reason': None}
```

Output report now includes:
```json
{
  "blocking_status": {
    "search_page": {"blocked": false, "reason": null},
    "detail_page": {"blocked": true, "reason": "Title contains: Just a moment"}
  }
}
```

#### 4. Custom Site Config Completeness (Codex)

Expand custom site CLI to include all necessary parameters:

```bash
python site_explorer.py custom \
  --url "https://example.com/rentals" \
  --detail-pattern "/property/" \
  --name "example_agency" \
  --listing-selectors ".property-card,article" \
  --pagination-type "numbered|scroll|load_more" \
  --next-selector ".pagination-next" \
  --wait-strategy "networkidle|load|domcontentloaded" \
  --cookie-selector "button.accept-cookies"
```

Or via JSON config file:
```bash
python site_explorer.py custom --config ./configs/new_site.json
```

#### 5. Real-time Console Feedback (Gemini)

Use `rich` library for live progress:

```python
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

async def run(self):
    with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
        task = progress.add_task("Loading search page...", total=None)
        await self._explore_search_page(page)
        console.print("✅ Search page analyzed", style="green")

        progress.update(task, description="Exploring detail page...")
        await self._explore_detail_page(page)
        console.print("✅ Detail page analyzed", style="green")
```

### Updated Implementation Plan

| Phase | Items | Status |
|-------|-------|--------|
| **1: Core** | CLI, Strategy base class, search/detail exploration, JSON output | MVP |
| **2: Security** | Network sanitization, block detection | Required |
| **3: UX** | Rich console output, custom site JSON config | Nice-to-have |
| **4: Polish** | All site strategies, error handling, CLAUDE.md update | Final |
