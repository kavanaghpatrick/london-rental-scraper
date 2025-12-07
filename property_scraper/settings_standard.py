# Standard settings without Playwright (for Rightmove, Foxtons)

BOT_NAME = 'property_scraper'
SPIDER_MODULES = ['property_scraper.spiders']
NEWSPIDER_MODULE = 'property_scraper.spiders'

USER_AGENT = 'PropertyResearchBot/1.0 (Academic Research)'
ROBOTSTXT_OBEY = False

# Concurrency
CONCURRENT_REQUESTS = 8
CONCURRENT_REQUESTS_PER_DOMAIN = 2
DOWNLOAD_DELAY = 2
RANDOMIZE_DOWNLOAD_DELAY = True

# Autothrottle
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 30
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0

# Retry
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Caching
HTTPCACHE_ENABLED = False

# Middlewares
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'property_scraper.middlewares.RotateUserAgentMiddleware': 400,
    'property_scraper.middlewares.RateLimitMiddleware': 543,
}

# Pipelines
ITEM_PIPELINES = {
    'property_scraper.pipelines.CleanDataPipeline': 100,
    'property_scraper.pipelines.DuplicateFilterPipeline': 200,
    'property_scraper.pipelines.JsonWriterPipeline': 300,
    'property_scraper.pipelines.SQLitePipeline': 400,
}

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
LOG_DATEFORMAT = '%H:%M:%S'

# Misc
REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'
TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'
TELNETCONSOLE_ENABLED = False
FEED_EXPORT_ENCODING = 'utf-8'

OUTPUT_DIR = 'output'

USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
]
