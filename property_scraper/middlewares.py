"""
Scrapy Middlewares for property scraping.

Includes:
- User-agent rotation
- Rate limit handling
- Proxy rotation (optional)
"""

import random
import time
import logging
from scrapy import signals
from scrapy.exceptions import IgnoreRequest


logger = logging.getLogger(__name__)


class RotateUserAgentMiddleware:
    """Rotate user agents for each request."""

    def __init__(self, user_agents):
        self.user_agents = user_agents

    @classmethod
    def from_crawler(cls, crawler):
        user_agents = crawler.settings.getlist('USER_AGENTS')
        if not user_agents:
            user_agents = [crawler.settings.get('USER_AGENT')]
        return cls(user_agents)

    def process_request(self, request, spider):
        request.headers['User-Agent'] = random.choice(self.user_agents)


class RateLimitMiddleware:
    """Handle 429 rate limit responses with NON-BLOCKING exponential backoff.

    Uses Scrapy's request scheduling instead of blocking sleep() which would
    freeze the entire Twisted event loop and all concurrent requests.

    Strategy:
    - On 429: Schedule retry with lower priority and increased download_delay
    - Track consecutive 429s to escalate backoff
    - Clean up retry tracking on successful responses to prevent memory leak
    """

    def __init__(self):
        self.retry_times = {}
        self.consecutive_429s = 0
        self.base_backoff = 30  # Base delay in seconds

    def process_response(self, request, response, spider):
        if response.status == 429:
            self.consecutive_429s += 1

            # Calculate backoff delay: 30s, 60s, 120s, 240s (max 4 min)
            backoff_delay = min(self.base_backoff * (2 ** min(self.consecutive_429s - 1, 3)), 240)

            # Get per-URL retry count
            retries = self.retry_times.get(request.url, 0)

            if retries < 3:
                self.retry_times[request.url] = retries + 1

                logger.warning(
                    f"[RATE-LIMIT] 429 on {request.url[:60]}... | "
                    f"Retry {retries + 1}/3 | "
                    f"Backoff: {backoff_delay}s | "
                    f"Consecutive 429s: {self.consecutive_429s}"
                )

                # NON-BLOCKING: Use Scrapy's download_delay mechanism
                # Schedule retry with lower priority so other requests proceed
                new_request = request.copy()
                new_request.dont_filter = True
                new_request.priority = request.priority - 10  # Lower priority
                new_request.meta['download_delay'] = backoff_delay

                # Also update spider's download delay if possible
                if hasattr(spider, 'download_delay'):
                    spider.download_delay = max(spider.download_delay, backoff_delay / 2)

                return new_request
            else:
                logger.error(f"[RATE-LIMIT] Max retries (3) exceeded for {request.url}")
                # Clean up retry tracking
                self.retry_times.pop(request.url, None)
        else:
            # Successful response - clean up and decay counters
            self.retry_times.pop(request.url, None)  # Prevent memory leak
            if self.consecutive_429s > 0:
                self.consecutive_429s = max(0, self.consecutive_429s - 1)

        return response


class ProxyMiddleware:
    """
    Rotate proxies for each request.

    Requires PROXY_LIST setting with list of proxy URLs.
    Format: http://user:pass@host:port or http://host:port
    """

    def __init__(self, proxy_list):
        self.proxy_list = proxy_list
        self.bad_proxies = set()

    @classmethod
    def from_crawler(cls, crawler):
        proxy_list = crawler.settings.getlist('PROXY_LIST', [])
        if not proxy_list:
            # Try loading from file
            proxy_file = crawler.settings.get('PROXY_FILE')
            if proxy_file:
                try:
                    with open(proxy_file) as f:
                        proxy_list = [line.strip() for line in f if line.strip()]
                except FileNotFoundError:
                    logger.warning(f"Proxy file not found: {proxy_file}")

        return cls(proxy_list)

    def process_request(self, request, spider):
        if not self.proxy_list:
            return

        # Get available proxies (excluding bad ones)
        available = [p for p in self.proxy_list if p not in self.bad_proxies]

        if available:
            proxy = random.choice(available)
            request.meta['proxy'] = proxy

    def process_exception(self, request, exception, spider):
        """Mark proxy as bad on connection errors."""
        proxy = request.meta.get('proxy')
        if proxy:
            self.bad_proxies.add(proxy)
            logger.warning(f"Marked proxy as bad: {proxy}")

            # Retry with different proxy
            if len(self.bad_proxies) < len(self.proxy_list):
                new_request = request.copy()
                new_request.dont_filter = True
                return new_request

        return None


class PropertyScraperSpiderMiddleware:
    """Spider middleware for property scraper."""

    @classmethod
    def from_crawler(cls, crawler):
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        return None

    def process_spider_output(self, response, result, spider):
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        pass

    def process_start_requests(self, start_requests, spider):
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info(f'Spider opened: {spider.name}')
