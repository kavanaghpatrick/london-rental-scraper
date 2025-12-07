"""Spider registry for CLI orchestration (PRD-004)."""

from dataclasses import dataclass


@dataclass
class SpiderConfig:
    """Configuration for a spider."""
    name: str
    spider_class: str
    requires_playwright: bool
    default_max_pages: int
    priority: int  # Lower = run first
    description: str


# Spider registry - all available spiders
SPIDERS = {
    'savills': SpiderConfig(
        name='savills',
        spider_class='property_scraper.spiders.savills_spider.SavillsSpider',
        requires_playwright=True,
        default_max_pages=84,
        priority=1,
        description='Savills - Premium agent, best sqft coverage (99.9%)',
    ),
    'knightfrank': SpiderConfig(
        name='knightfrank',
        spider_class='property_scraper.spiders.knightfrank_spider.KnightFrankSpider',
        requires_playwright=True,
        default_max_pages=50,
        priority=2,
        description='Knight Frank - Premium agent, excellent sqft (93%)',
    ),
    'chestertons': SpiderConfig(
        name='chestertons',
        spider_class='property_scraper.spiders.chestertons_spider.ChestertonsSpider',
        requires_playwright=True,
        default_max_pages=50,
        priority=3,
        description='Chestertons - Premium agent with Cloudflare',
    ),
    'foxtons': SpiderConfig(
        name='foxtons',
        spider_class='property_scraper.spiders.foxtons_spider.FoxtonsSpider',
        requires_playwright=False,
        default_max_pages=100,
        priority=4,
        description='Foxtons - Fast HTTP spider, excellent sqft (98%)',
    ),
    'rightmove': SpiderConfig(
        name='rightmove',
        spider_class='property_scraper.spiders.rightmove_spider.RightmoveSpider',
        requires_playwright=False,
        default_max_pages=100,
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
