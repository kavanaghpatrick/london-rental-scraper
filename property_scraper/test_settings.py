# Test settings without Playwright
from property_scraper.settings import *

# Use default HTTP handlers instead of Playwright
DOWNLOAD_HANDLERS = {
    "http": "scrapy.core.downloader.handlers.http.HTTPDownloadHandler",
    "https": "scrapy.core.downloader.handlers.http.HTTPDownloadHandler",
}
HTTPCACHE_ENABLED = False
