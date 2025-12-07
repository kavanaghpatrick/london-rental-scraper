# Scrapy settings for property_scraper project
#
# Optimized for scraping UK property listings at scale
# with respectful rate limiting and anti-detection measures.

BOT_NAME = 'property_scraper'

SPIDER_MODULES = ['property_scraper.spiders']
NEWSPIDER_MODULE = 'property_scraper.spiders'

# Identify yourself (good practice)
USER_AGENT = 'PropertyResearchBot/1.0 (Academic Research; +https://example.edu)'

# Obey robots.txt - set to False only for academic research with justification
ROBOTSTXT_OBEY = False  # Rightmove blocks bots in robots.txt

# =============================================================================
# CONCURRENCY SETTINGS
# =============================================================================

# Global concurrent requests
CONCURRENT_REQUESTS = 8

# Per-domain limits (CRITICAL for not getting blocked)
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# Download delay between requests (seconds)
DOWNLOAD_DELAY = 2

# Randomize delay (0.5x to 1.5x of DOWNLOAD_DELAY)
RANDOMIZE_DOWNLOAD_DELAY = True

# =============================================================================
# AUTOTHROTTLE (Recommended for respectful scraping)
# =============================================================================

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 30
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
AUTOTHROTTLE_DEBUG = False

# =============================================================================
# RETRY SETTINGS
# =============================================================================

RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Custom retry for 429 (rate limit)
RETRY_PRIORITY_ADJUST = -1

# =============================================================================
# CACHING (speeds up development)
# =============================================================================

HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # 24 hours
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [429, 500, 502, 503]
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# =============================================================================
# MIDDLEWARES
# =============================================================================

DOWNLOADER_MIDDLEWARES = {
    # Rotate user agents
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'property_scraper.middlewares.RotateUserAgentMiddleware': 400,

    # Custom rate limit handling
    'property_scraper.middlewares.RateLimitMiddleware': 543,

    # Uncomment for proxy rotation (requires proxy list)
    # 'property_scraper.middlewares.ProxyMiddleware': 350,
}

# =============================================================================
# ITEM PIPELINES
# =============================================================================

ITEM_PIPELINES = {
    'property_scraper.pipelines.CleanDataPipeline': 100,
    'property_scraper.pipelines.DuplicateFilterPipeline': 200,
    'property_scraper.pipelines.JsonWriterPipeline': 300,
    'property_scraper.pipelines.SQLitePipeline': 400,  # SQLite database
}

# =============================================================================
# PLAYWRIGHT INTEGRATION (for JS-heavy pages like OpenRent)
# =============================================================================

# Enable Playwright for OpenRent spider (requires: pip install scrapy-playwright && playwright install chromium)
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}
PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_LAUNCH_OPTIONS = {"headless": True}
PLAYWRIGHT_MAX_CONTEXTS = 3

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

FEED_EXPORT_ENCODING = 'utf-8'

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
LOG_DATEFORMAT = '%H:%M:%S'
LOG_SHORT_NAMES = True  # Use short logger names

# Log stats every N seconds
STATS_DUMP = True
LOGSTATS_INTERVAL = 30.0

# =============================================================================
# MISC
# =============================================================================

# Use new request fingerprinter
REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'

# Asyncio reactor for better async support
TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'

# Telnet console (disable in production)
TELNETCONSOLE_ENABLED = False

# =============================================================================
# CUSTOM SETTINGS
# =============================================================================

# List of user agents to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

# Target areas
TARGET_AREAS = [
    'Belgravia', 'Chelsea', 'Kensington', 'South-Kensington',
    'Knightsbridge', 'Mayfair', 'Notting-Hill', 'Holland-Park',
]

# Output directory
OUTPUT_DIR = 'output'
