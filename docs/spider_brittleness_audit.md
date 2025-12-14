# Spider Brittleness Audit - December 13, 2025

## Summary of Failures (Last 3 Days)

### Critical Issue 1: Chestertons PageMethod Bug (ROOT CAUSE OF TODAY'S STALL)

**Location**: `chestertons_spider.py` lines 150-156

**Bug**: Using dict format instead of PageMethod objects:
```python
# WRONG (current code):
'playwright_page_methods': [
    {'method': 'wait_for_selector', 'args': ['.pegasus-property-card'], 'kwargs': {'timeout': 30000}},
    {'method': 'wait_for_timeout', 'args': [3000]},
]

# CORRECT (Savills does this right):
'playwright_page_methods': [
    PageMethod('wait_for_selector', '.pegasus-property-card', timeout=30000),
    PageMethod('wait_for_timeout', 3000),
]
```

**Evidence**: Warning logs show:
```
[scrapy-playwright] WARNING: Ignoring {'method': 'wait_for_selector', ...}: expected PageMethod, got <class 'dict'>
```

**Impact**: Initial page wait is SKIPPED, spider proceeds without content loaded, then stalls.

---

### Critical Issue 2: "No Playwright page available" Error

**Affected**: All Playwright spiders (savills, knightfrank, chestertons)

**Error Pattern**:
```
[spider] ERROR: [ERROR] No Playwright page available
```

**Root Cause Analysis**:
1. HTTP cache (even disabled) or browser context issues
2. Playwright page not being passed correctly to callback
3. Request gets served from some cache layer without Playwright rendering

**Occurrences**:
- savills: 1 error per run
- knightfrank: 11 errors in one run
- chestertons: 1+ errors

**Current Handling**: Spider logs error and returns, losing that page entirely.

**Needed**: Retry mechanism when Playwright page is missing.

---

### Critical Issue 3: Knight Frank Selector Timeouts

**Pattern**:
```
[knightfrank] WARNING: Timeout waiting for cards on page 29: Page.wait_for_selector: Timeout 30000ms exceeded.
[knightfrank] WARNING: Timeout waiting for cards on page 30: ...
```

**Location**: `knightfrank_spider.py` lines 198-204

**Current Code**:
```python
try:
    await playwright_page.wait_for_selector('.property-features', timeout=30000)
except Exception as e:
    self.logger.warning(f"Timeout waiting for cards on page {page_num}: {e}")
    await playwright_page.close()
    return  # <-- GIVES UP ENTIRELY
```

**Impact**: Pages 29-37 skipped entirely, ~300 listings lost per run.

**Needed**: Retry with backoff for selector timeouts.

---

### Critical Issue 4: Detail Page Processing Stalls

**Pattern**: After loading search results, spider stalls during detail page fetches.

**Evidence**: Chestertons today - loaded 642 cards, then stuck at 7 items for 4+ minutes.

**Root Causes**:
1. Playwright detail page requests not timing out properly
2. Concurrency issues with 642 simultaneous detail requests
3. No circuit breaker for failing detail pages

**Needed**:
- Batch detail page requests (max 10 concurrent)
- Aggressive timeout on detail pages
- Skip and continue pattern instead of blocking

---

### Issue 5: No Global Spider Timeout

**Problem**: Spiders can run indefinitely if something gets stuck.

**Needed**: `CLOSESPIDER_TIMEOUT` setting or manual timeout watchdog.

---

## Current Code Analysis

### Savills Spider (MOST STABLE)
- Uses PageMethod correctly ✅
- Has safe_evaluate() with timeouts ✅
- Has DETAIL_PAGE_TIMEOUT ✅
- Click-based pagination (complex but works)
- Single browser session for pagination ✅

### Knight Frank Spider (MODERATE ISSUES)
- Uses playwright_include_page ✅
- Has safe_evaluate() with timeouts ✅
- Has DETAIL_PAGE_TIMEOUT ✅
- NO RETRY on selector timeout ❌
- All pages queued upfront (offset pagination)

### Chestertons Spider (MOST BRITTLE)
- **BROKEN**: PageMethod format is wrong ❌
- Has safe_evaluate() with timeouts ✅
- Has DETAIL_PAGE_TIMEOUT ✅
- Load More pagination within single session
- Queues 642 detail requests at once ❌

### Foxtons Spider (HTTP - STABLE)
- No Playwright needed ✅
- Simple JSON extraction from __NEXT_DATA__ ✅
- Basic error handling ✅

### Rightmove Spider (HTTP - STABLE)
- No Playwright needed ✅
- Similar to Foxtons ✅

---

## Recommended Fixes

### Fix 1: Chestertons PageMethod Format (CRITICAL)
```python
from scrapy_playwright.page import PageMethod

yield scrapy.Request(
    url,
    meta={
        'playwright': True,
        'playwright_include_page': True,
        'playwright_page_methods': [
            PageMethod('wait_for_selector', '.pegasus-property-card', timeout=30000),
            PageMethod('wait_for_timeout', 3000),
        ],
    },
)
```

### Fix 2: Retry Logic for Missing Playwright Page
```python
async def parse_search(self, response):
    playwright_page = response.meta.get('playwright_page')

    if not playwright_page:
        retry_count = response.meta.get('retry_count', 0)
        if retry_count < 3:
            self.logger.warning(f"No Playwright page, retrying ({retry_count + 1}/3)")
            yield response.request.replace(
                meta={**response.meta, 'retry_count': retry_count + 1},
                dont_filter=True
            )
            return
        self.logger.error("No Playwright page after 3 retries, skipping")
        return
```

### Fix 3: Selector Timeout Retry
```python
async def wait_for_content_with_retry(page, selector, timeout=30000, max_retries=3):
    for attempt in range(max_retries):
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Selector timeout, retry {attempt + 1}/{max_retries}")
                await page.reload()
                await page.wait_for_timeout(2000)
            else:
                logger.error(f"Selector timeout after {max_retries} retries")
                return False
    return False
```

### Fix 4: Batch Detail Page Processing
```python
# Instead of yielding all 642 detail requests at once:
MAX_CONCURRENT_DETAILS = 20

# In parse_search, collect items first, then batch process:
detail_items = []
for card_data in cards_data:
    item = self.parse_card_data(card_data)
    if item and self.is_target_area(item):
        detail_items.append(item)

# Process in batches
for i, item in enumerate(detail_items):
    yield scrapy.Request(
        item['url'],
        callback=self.parse_detail,
        meta={
            'item': dict(item),
            'playwright': True,
            'playwright_include_page': True,
            'playwright_page_goto_kwargs': {
                'timeout': 20000,  # Reduced from 30s
                'wait_until': 'domcontentloaded',
            },
        },
        priority=-i,  # Process sequentially by priority
        errback=self.handle_detail_error
    )
```

### Fix 5: Global Spider Timeout
```python
# In settings.py:
CLOSESPIDER_TIMEOUT = 3600  # 1 hour max per spider

# Or in spider:
custom_settings = {
    'CLOSESPIDER_TIMEOUT': 3600,
}
```

### Fix 6: Circuit Breaker for Detail Pages
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure = 0
        self.is_open = False

    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.threshold:
            self.is_open = True

    def record_success(self):
        self.failures = 0
        self.is_open = False

    def can_proceed(self):
        if not self.is_open:
            return True
        if time.time() - self.last_failure > self.reset_timeout:
            self.is_open = False
            self.failures = 0
            return True
        return False
```

---

## Testing Checklist

After implementing fixes:

1. [ ] Run each spider individually with `--max-pages 3`
2. [ ] Verify no "No Playwright page available" errors
3. [ ] Verify no selector timeout without retry
4. [ ] Verify detail pages don't stall
5. [ ] Run full workflow `python -m cli.main scrape --all --full`
6. [ ] Monitor for 30+ minutes without intervention
7. [ ] Verify all 5 spiders complete successfully

---

## Files to Modify

1. `property_scraper/spiders/chestertons_spider.py` - PageMethod fix
2. `property_scraper/spiders/knightfrank_spider.py` - Retry on selector timeout
3. `property_scraper/spiders/savills_spider.py` - Add retry for missing page
4. `property_scraper/settings.py` - Add CLOSESPIDER_TIMEOUT
5. Create `property_scraper/utils/resilience.py` - Shared retry/circuit breaker utilities
