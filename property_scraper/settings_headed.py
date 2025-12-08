# Headed mode settings for manual captcha solving
# Usage: SCRAPY_SETTINGS_MODULE=property_scraper.settings_headed scrapy crawl floorplan_enricher -a source=chestertons

from property_scraper.settings import *

# Override to show browser window
PLAYWRIGHT_LAUNCH_OPTIONS = {"headless": False}

# DISABLE HTTP CACHE - we need fresh requests, not cached Cloudflare challenges
HTTPCACHE_ENABLED = False

# Slower settings to allow manual intervention
DOWNLOAD_DELAY = 10
CONCURRENT_REQUESTS = 1
CONCURRENT_REQUESTS_PER_DOMAIN = 1
AUTOTHROTTLE_START_DELAY = 10
AUTOTHROTTLE_MAX_DELAY = 120

# Longer timeout to allow time for captcha solving
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 120000  # 2 minutes
