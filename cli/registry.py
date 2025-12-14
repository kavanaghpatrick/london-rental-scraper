"""Spider registry for CLI orchestration (PRD-004).

Issue #19 FIX: Centralized area configuration for all spiders.
"""

from dataclasses import dataclass


# =============================================================================
# Issue #19 FIX: Centralized Target Areas Configuration
# =============================================================================

# Target areas for scraping (used in URL building)
# Format: Display name -> URL slug mapping
TARGET_AREAS = {
    'Belgravia': 'belgravia',
    'Chelsea': 'chelsea',
    'Kensington': 'kensington',
    'South-Kensington': 'south-kensington',
    'Knightsbridge': 'knightsbridge',
    'Notting-Hill': 'notting-hill',
    'Earls-Court': 'earls-court',
    'Fulham': 'fulham',
    'Hampstead': 'hampstead',
    'St-Johns-Wood': 'st-johns-wood',
    'Mayfair': 'mayfair',
    'Marylebone': 'marylebone',
}

# Target postcodes for filtering (Prime Central London)
TARGET_POSTCODES = [
    'SW1', 'SW3', 'SW5', 'SW7', 'SW10',  # Chelsea, South Ken, Knightsbridge
    'W8', 'W11', 'W2', 'W1',  # Kensington, Notting Hill, Mayfair
    'NW1', 'NW3', 'NW8',  # St John's Wood, Hampstead
    'SW6', 'SW11',  # Fulham
]

# Postcode to area name mapping
POSTCODE_TO_AREA = {
    'SW1': 'Belgravia',
    'SW3': 'Chelsea',
    'SW5': 'Earls Court',
    'SW6': 'Fulham',
    'SW7': 'South Kensington',
    'SW10': 'Chelsea',
    'SW11': 'Battersea',
    'W1': 'Mayfair',
    'W2': 'Bayswater',
    'W8': 'Kensington',
    'W11': 'Notting Hill',
    'W14': 'West Kensington',
    'NW1': "St John's Wood",
    'NW3': 'Hampstead',
    'NW8': "St John's Wood",
}


def get_area_list() -> list[str]:
    """Get list of target area display names."""
    return list(TARGET_AREAS.keys())


def get_area_slug(area: str) -> str:
    """Get URL slug for an area name."""
    return TARGET_AREAS.get(area, area.lower().replace(' ', '-'))


def postcode_to_area(postcode: str) -> str:
    """Convert postcode prefix to area name."""
    if not postcode:
        return ''
    # Extract district (e.g., SW3 from SW3 4PL)
    district = postcode.upper().split()[0] if ' ' in postcode else postcode.upper()
    # Try exact match first
    if district in POSTCODE_TO_AREA:
        return POSTCODE_TO_AREA[district]
    # Try prefix match (SW1 matches SW1A, SW1E, etc.)
    for prefix, area in POSTCODE_TO_AREA.items():
        if district.startswith(prefix):
            return area
    return postcode


def is_target_postcode(postcode: str) -> bool:
    """Check if postcode is in target area."""
    if not postcode:
        return True  # Include if unknown
    district = postcode.upper().split()[0] if ' ' in postcode else postcode.upper()
    for target in TARGET_POSTCODES:
        if district.startswith(target):
            return True
    return False


# =============================================================================
# Spider Configuration
# =============================================================================

@dataclass
class SpiderConfig:
    """Configuration for a spider."""
    name: str
    spider_class: str
    requires_playwright: bool
    default_max_pages: int
    priority: int  # Lower = run first
    description: str
    supports_detail_fetch: bool = True  # If False, --full mode skips detail pages for this spider


# Spider registry - all available spiders
SPIDERS = {
    'savills': SpiderConfig(
        name='savills',
        spider_class='property_scraper.spiders.savills_spider.SavillsSpider',
        requires_playwright=True,
        default_max_pages=None,  # Unlimited - scrape all available
        priority=1,
        description='Savills - Premium agent, best sqft coverage (99.9%)',
    ),
    'knightfrank': SpiderConfig(
        name='knightfrank',
        spider_class='property_scraper.spiders.knightfrank_spider.KnightFrankSpider',
        requires_playwright=True,
        default_max_pages=None,  # Unlimited
        priority=2,
        description='Knight Frank - Premium agent, excellent sqft (93%)',
    ),
    'chestertons': SpiderConfig(
        name='chestertons',
        spider_class='property_scraper.spiders.chestertons_spider.ChestertonsSpider',
        requires_playwright=True,
        default_max_pages=None,  # Unlimited
        priority=3,
        description='Chestertons - Premium agent with Cloudflare',
        supports_detail_fetch=False,  # Too slow with 1400+ Playwright detail pages; use enrich-floorplans instead
    ),
    'foxtons': SpiderConfig(
        name='foxtons',
        spider_class='property_scraper.spiders.foxtons_spider.FoxtonsSpider',
        requires_playwright=False,
        default_max_pages=None,  # Unlimited
        priority=4,
        description='Foxtons - Fast HTTP spider, excellent sqft (98%)',
    ),
    'rightmove': SpiderConfig(
        name='rightmove',
        spider_class='property_scraper.spiders.rightmove_spider.RightmoveSpider',
        requires_playwright=False,
        default_max_pages=None,  # Unlimited - scrape all available
        priority=5,
        description='Rightmove - Aggregator, needs enrichment for sqft',
    ),
}


def get_spider(name: str) -> SpiderConfig | None:
    """Get spider config by name."""
    return SPIDERS.get(name.lower())


def get_all_spiders() -> list[SpiderConfig]:
    """Get all spiders sorted by priority."""
    return sorted(SPIDERS.values(), key=lambda s: s.priority)


def get_playwright_spiders() -> list[SpiderConfig]:
    """Get spiders that require Playwright."""
    return [s for s in get_all_spiders() if s.requires_playwright]


def get_http_spiders() -> list[SpiderConfig]:
    """Get spiders that use standard HTTP."""
    return [s for s in get_all_spiders() if not s.requires_playwright]
