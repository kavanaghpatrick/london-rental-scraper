# Scraper Brittleness Analysis - December 14, 2025

## Summary

Full scrape analysis revealed **5 critical** and **3 moderate** robustness issues. CLI reported "All 5 spider(s) succeeded" but 2 spiders (Foxtons, Rightmove) scraped **0 items** due to DNS failures.

## Critical Issues

### 1. CLI False Success Detection (CRITICAL)
**Location:** `cli/main.py:122`

**Problem:** CLI determines success solely by subprocess exit code. Scrapy returns exit code 0 even when:
- 0 items scraped
- All requests failed
- DNS resolution errors occurred

**Evidence:**
```
  OK foxtons: Completed. Log: foxtons_20251214_153137.log
  OK rightmove: Completed. Log: rightmove_20251214_153211.log
```
But both had `items_per_minute: 0.0` and 24 DNS errors each.

**Impact:** Silent failures - operators think scrape succeeded when it completely failed.

**Fix:** Parse scrapy stats from log or stdout, fail if `item_scraped_count == 0` for non-enricher spiders.

---

### 2. HTTP Spider DNS Failures (CRITICAL)
**Location:** Twisted/Scrapy DNS resolver

**Problem:** Twisted's DNS resolver fails to resolve domains that system DNS resolves fine.

**Evidence:**
```
DNSLookupError: DNS lookup failed: no results for hostname lookup: www.foxtons.co.uk
```
But `ping www.foxtons.co.uk` works immediately.

**Pattern:**
| Spider | Type | DNS Result |
|--------|------|------------|
| savills | Playwright | OK |
| knightfrank | Playwright | OK |
| chestertons | Playwright | OK |
| foxtons | HTTP/Twisted | FAILED |
| rightmove | HTTP/Twisted | FAILED |

Playwright spiders use Chromium's DNS resolver, HTTP spiders use Twisted's.

**Impact:** HTTP spiders (Foxtons, Rightmove) completely fail while Playwright spiders work.

**Fix Options:**
1. Use system DNS resolver: `SCRAPY_DNS_RESOLVER_CLASS=scrapy.resolver.CachingHostnameResolver`
2. Add DNS retry logic with fallback to system resolver
3. Convert HTTP spiders to Playwright (heavyweight but reliable)

---

### 3. KnightFrank Over-Pagination (CRITICAL)
**Location:** `property_scraper/spiders/knightfrank_spider.py:146`

**Problem:** Spider hardcodes `pages_needed = 50` (2400 properties) but KnightFrank only has ~700-800 listings (~16 pages).

```python
if self.max_properties is None:
    pages_needed = 50  # ~2400 properties max  <-- BUG
```

**Evidence:**
```
15:22:17 [WARNING] Selector timeout on page 32 (offset=1488)
15:22:20 [WARNING] Selector timeout on page 33 (offset=1536)
...
```
Pages 32+ have no results, `.property-features` selector times out (30s each).

**Impact:**
- ~30 pages × 30s timeout = **15+ minutes** wasted on empty pages
- 79 warnings logged
- Response times showed 2000+ seconds (queued requests backing up)

**Fix:** Detect end-of-results:
1. Parse total results count from first page
2. Track consecutive empty pages, stop after N (e.g., 3)
3. Check if "No results" message appears

---

### 4. No Item Count Validation (CRITICAL)
**Location:** `cli/main.py`

**Problem:** No validation that spiders actually scraped data before reporting success.

**Impact:** Entire scrape can fail silently. Database `last_seen` dates show stale data:
```
rightmove   │ 2025-12-13  <-- Yesterday, not updated!
foxtons     │ 2025-12-13  <-- Yesterday, not updated!
```

**Fix:** After spider completes, query DB for items scraped in last N minutes from that source. Alert if count is 0 or significantly below expected.

---

### 5. Conda Environment Dependency (CRITICAL)
**Location:** CLI entry point

**Problem:** Running without proper conda environment causes SQLite import error:
```
ImportError: dlopen(_sqlite3.cpython-311-darwin.so): Symbol not found: _sqlite3_enable_load_extension
```

**Impact:** CLI appears broken without explanation.

**Fix:**
1. Add shebang with conda env: `#!/usr/bin/env -S conda run -n claude-code python`
2. Add environment check at CLI startup with helpful error message

---

## Moderate Issues

### 6. Chestertons Fast Mode Fallback
**Location:** `cli/main.py` run_spider logic

**Problem:** Chestertons automatically falls back to "fast mode" (search results only, no detail pages) when detail fetching is requested.

**Evidence:**
```
OK chestertons: Completed. Log: chestertons_20251214_152817.log (fast mode)
response_received_count: 1  # Only 1 page scraped
```

**Impact:** Missing sqft data from detail pages (Chestertons has 71% sqft vs 99% for Savills with full mode).

**Recommendation:** Investigate why Chestertons detail fetching fails, add logging.

---

### 7. KnightFrank Selector Timeout Cascade
**Location:** `property_scraper/spiders/knightfrank_spider.py:212-227`

**Problem:** When page has no results, waits full 30s timeout × 3 retries = 90s per empty page.

**Evidence:**
```
15:22:55 [WARNING] Timeout waiting for cards on page 32 after 3 attempts
15:22:59 [WARNING] Timeout waiting for cards on page 33 after 3 attempts
```

**Impact:** Combined with over-pagination, creates 15-20 minute delays.

**Fix:** Check for "No results" message or empty page indicator before waiting for results.

---

### 8. Playwright Page Close Delay
**Location:** All Playwright spiders

**Problem:** No explicit page close on timeout errors, potential memory leak.

**Evidence:** Stats showed `playwright/page_count: 706` with `max_concurrent: 8`.

**Impact:** Memory growth over long runs.

**Fix:** Ensure `playwright_page.close()` in all error paths.

---

## Scrape Results Summary

| Spider | Status | Items | Errors | Notes |
|--------|--------|-------|--------|-------|
| savills | OK | 370 | 0 | Full detail pages |
| knightfrank | OK | 653 | 2 | 79 warnings (over-pagination) |
| chestertons | OK | ? | 0 | Fast mode only |
| foxtons | FAILED | 0 | 24 | DNS errors |
| rightmove | FAILED | 0 | 24 | DNS errors |

## Recommended Priority

1. **P0 (Immediate):** CLI success validation - add item count check
2. **P0 (Immediate):** DNS resolver fix for HTTP spiders
3. **P1 (This week):** KnightFrank pagination fix
4. **P2 (Soon):** Environment check at startup
5. **P2 (Soon):** Chestertons detail fetch investigation

## Test Commands

```bash
# Re-run just HTTP spiders to test DNS fix
python -m cli.main scrape -s rightmove -s foxtons

# Test KnightFrank with limited pages
python -m cli.main scrape -s knightfrank --max-pages 20

# Check data freshness
python -m cli.main status
```
