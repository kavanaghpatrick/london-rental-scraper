
import sys
sys.path.insert(0, '.')
from property_scraper.settings import *

# Disable Playwright for this test
DOWNLOAD_HANDLERS = {}
