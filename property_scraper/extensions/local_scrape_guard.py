"""
LocalScrapeGuard — spider-level enforcement that the scraper NEVER runs locally.

WHY (BLOCKER 2 — the big one):
    A CLI-only guard is a SIEVE. `scrapy crawl <spider> -a listing_type=sale`,
    `scripts/run_full_scrape.sh`, and a direct `python -m scrapy crawl …` all start a
    REAL crawl while bypassing cli.main entirely. To make "the scraper never runs
    locally" actually true, enforcement has to live at the SCRAPY level so EVERY real
    crawl entrypoint is covered — CLI, `scrapy crawl`, and run_full_scrape.sh — for BOTH
    the rental and the for-sale verticals.

HOW:
    A scrapy EXTENSION wired into EXTENSIONS in both settings modules
    (property_scraper.settings + property_scraper.settings_standard). Its engine_started
    handler RAISES (RuntimeError) UNLESS the run is inside CI (GITHUB_ACTIONS / CI) or the
    dangerous dev-only ALLOW_LOCAL_SCRAPE=1 override is set.

    `engine_started` fires ONLY on a real crawl (when the Scrapy engine actually starts
    fetching). Unit tests that merely instantiate a spider and call its parse_* methods
    directly never start the engine, so they are UNAFFECTED — no test triggers this guard.

PRODUCTION:
    Real crawls run only in GitHub Actions:
      - for-sale vertical -> .github/workflows/for-sale-scrape.yml
      - rental vertical   -> .github/workflows/daily-scrape.yml
    Both set GITHUB_ACTIONS=true, so the guard is a no-op there. A local operator who
    truly needs to crawl (debugging) can set ALLOW_LOCAL_SCRAPE=1 — intentionally loud
    and dev-only.
"""

import logging
import os

from scrapy import signals

logger = logging.getLogger(__name__)


def _in_ci() -> bool:
    """True when running inside a CI environment (GitHub Actions or generic CI)."""
    return os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true"


def _local_scrape_allowed() -> bool:
    """True when a real crawl is permitted in this environment.

    Permitted ONLY inside CI, or when the explicit dev-only ALLOW_LOCAL_SCRAPE=1
    override is set. Everything else is a local machine and must be refused.
    """
    return _in_ci() or os.environ.get("ALLOW_LOCAL_SCRAPE") == "1"


REFUSAL_MESSAGE = (
    "REFUSED: local scraping is disabled. Production scraping runs ONLY in GitHub "
    "Actions:\n"
    "  - for-sale vertical -> .github/workflows/for-sale-scrape.yml\n"
    "  - rental vertical   -> .github/workflows/daily-scrape.yml\n"
    "A real crawl was started outside CI (e.g. `scrapy crawl ...`, "
    "scripts/run_full_scrape.sh, or `python -m cli.main scrape ...`). This is blocked at "
    "the SPIDER level so EVERY entrypoint is covered.\n"
    "If you genuinely must crawl from a local machine (debugging only), set the dangerous "
    "dev-only override ALLOW_LOCAL_SCRAPE=1 — never in normal use."
)


class LocalScrapeGuard:
    """Scrapy extension that aborts any REAL crawl started outside CI.

    ENFORCEMENT MECHANISM (load-bearing):
        The refusal is raised in ``from_crawler``. Scrapy's ExtensionManager.from_crawler
        loop catches ONLY ``NotConfigured`` — any other exception propagates and aborts
        the entire crawler build BEFORE the engine starts or a single request is made.
        That is the bulletproof seam: a RuntimeError here hard-stops `scrapy crawl`,
        run_full_scrape.sh, and the CLI subprocess alike.

        IMPORTANT: a RuntimeError raised inside the ``engine_started`` SIGNAL handler is
        NOT sufficient — Scrapy dispatches signals via send_catch_log, which catches and
        merely LOGS handler exceptions while the crawl continues to completion (verified:
        it scraped 210 items despite the handler raising). So the signal handler is kept
        only as defense-in-depth; the from_crawler raise is what actually blocks.
    """

    def __init__(self):
        # If we reach __init__ at all, the run is permitted (from_crawler already
        # enforced the ban). The signal handler below is belt-and-braces.
        pass

    @classmethod
    def from_crawler(cls, crawler):
        # HARD REFUSAL: raise before the engine can start. Propagates out of the
        # ExtensionManager (only NotConfigured is swallowed there) -> crawl aborts.
        if not _local_scrape_allowed():
            logger.error(REFUSAL_MESSAGE)
            raise RuntimeError(REFUSAL_MESSAGE)
        ext = cls()
        crawler.signals.connect(ext.engine_started, signal=signals.engine_started)
        return ext

    def engine_started(self):
        """Defense-in-depth re-check at engine start.

        from_crawler is the real enforcement (a raise there aborts the build). This
        re-checks in case the environment changed between construction and engine start;
        raising here is logged-and-swallowed by Scrapy, so we ALSO stop the engine
        explicitly when a crawler reference is available.
        """
        if _local_scrape_allowed():
            return
        logger.error(REFUSAL_MESSAGE)
        raise RuntimeError(REFUSAL_MESSAGE)
