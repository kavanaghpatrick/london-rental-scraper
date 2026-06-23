"""Guard the Scrapy version contract the spiders depend on.

WHY THIS EXISTS (a real, silent prod outage on 2026-06-23):
    requirements.txt pinned only `scrapy>=2.11.0`. Every spider in this project defines
    the LEGACY synchronous `start_requests()` and relies on the base Spider bridging it
    into the new (2.13+) `async def start()` entry point. Scrapy 2.16.0 REMOVED that
    bridge: the default `start()` now only reads `start_urls` (which these spiders leave
    empty — they build per-area URLs dynamically in `start_requests()`), so the engine
    emitted ZERO start requests and every spider "finished" with 0 items. Both the
    for-sale-scrape and the rental daily-scrape went silently 0-items the day unbounded
    pip resolved 2.16.0. No exception, no failed step inside scrapy — just no data.

    PR CI cannot run a real crawl (the LocalScrapeGuard + the no-local-scrape rule), so a
    crawl-level test can't catch this. These pure-unit assertions can, and run in PR CI:

      1. the installed Scrapy stays under the ceiling pinned in requirements.txt, and
      2. the installed Scrapy's base `start()` STILL bridges to `start_requests()`
         (the actual mechanism the spiders depend on), and
      3. a representative production spider actually yields start requests in both the
         rental and the for-sale verticals.

    To raise the ceiling: first give every spider an `async def start()` (delegating to
    `start_requests()`, which works on all 2.11–2.16+ engines), THEN bump the pin here and
    in requirements.txt together.
"""

import inspect
import re

import scrapy

# The exclusive upper bound pinned in requirements.txt (`scrapy>=2.11.0,<2.15`).
# 2.14.1 is the validated local/CI version; the start()->start_requests() bridge is
# present through 2.14 and removed by 2.16. <2.15 keeps CI on the proven, bridging engine.
SUPPORTED_MAX_EXCLUSIVE = (2, 15)
SUPPORTED_MIN_INCLUSIVE = (2, 11)


def _version_tuple(v: str) -> tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", v)
    assert m, f"could not parse scrapy version {v!r}"
    return int(m.group(1)), int(m.group(2))


def test_scrapy_version_within_supported_ceiling():
    """Installed Scrapy must stay within the bridging range pinned in requirements.txt.

    This is the guard that would have caught 2026-06-23: an unbounded `scrapy>=2.11.0`
    let CI silently jump to 2.16.0, which broke every spider's start path.
    """
    ver = _version_tuple(scrapy.__version__)
    assert ver >= SUPPORTED_MIN_INCLUSIVE, (
        f"Scrapy {scrapy.__version__} is older than the supported floor "
        f"{SUPPORTED_MIN_INCLUSIVE}."
    )
    assert ver < SUPPORTED_MAX_EXCLUSIVE, (
        f"Scrapy {scrapy.__version__} is at/above the unsupported ceiling "
        f"{SUPPORTED_MAX_EXCLUSIVE}. Scrapy 2.15+/2.16 removed the base "
        f"Spider.start() -> start_requests() bridge that EVERY spider here depends on, so "
        f"the engine emits ZERO start requests and the scrape silently produces 0 items. "
        f"Do NOT loosen requirements.txt's `scrapy<2.15` pin until every spider defines an "
        f"`async def start()`. See this module's docstring."
    )


def test_base_spider_start_still_bridges_to_start_requests():
    """The installed engine's base `start()` must still invoke `start_requests()`.

    Mechanism-level guard: even if a version somehow slipped past the numeric ceiling,
    the thing that actually broke is the bridge. The base `Spider.start` source must still
    reference `start_requests` (it iterates `self.start_requests()` through 2.14). If a
    future Scrapy keeps the version low but drops the bridge, this fails LOUD instead of
    shipping a silent 0-item scrape.
    """
    start = getattr(scrapy.Spider, "start", None)
    assert start is not None, "scrapy.Spider has no start() — unexpected engine shape"
    src = inspect.getsource(start)
    assert "start_requests" in src, (
        "scrapy.Spider.start() no longer bridges to start_requests(). Every spider in "
        "this project defines only the legacy sync start_requests() and leaves start_urls "
        "empty, so without the bridge the engine emits ZERO start requests -> silent "
        "0-item scrape. Migrate the spiders to `async def start()` before upgrading."
    )


def test_production_spiders_emit_start_requests_in_both_modes():
    """Each production spider must yield >=1 start request in BOTH verticals.

    Pure instantiation + start_requests() iteration (no engine, no crawl, no network) —
    documents the contract and catches a spider that stops building URLs in either mode.
    """
    from property_scraper.spiders.rightmove_spider import RightmoveSpider

    for mode, section in (("rent", "property-to-rent"), ("sale", "property-for-sale")):
        spider = RightmoveSpider(listing_type=mode)
        reqs = list(spider.start_requests())
        assert reqs, f"rightmove yielded NO start requests in {mode} mode"
        assert all(section in r.url for r in reqs), (
            f"rightmove {mode}-mode start URLs do not target /{section}/: "
            f"{[r.url for r in reqs][:3]}"
        )
