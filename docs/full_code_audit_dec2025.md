# Full Codebase Audit - December 14, 2025

## Executive Summary

Comprehensive audit of the London rental property scraper codebase covering CLI, pipelines, and all spiders. The code is generally well-structured with good error handling. **6 critical issues** and **12 medium-priority issues** identified.

---

## CRITICAL Issues (P0 - Fix Immediately)

### 1. JSON File Append Race Condition
**File:** `property_scraper/pipelines.py:199`
**Severity:** CRITICAL - Data Loss Risk

```python
with open(filepath, 'a') as f:
    f.write(line)
```

**Problem:** Multiple spiders running concurrently can cause interleaved writes to the same JSONL file, producing invalid JSON. The `open(..., 'a')` is not atomic.

**Impact:** Corrupted JSONL files when running `--all` with multiple spiders.

**Fix:**
```python
import fcntl

with open(filepath, 'a') as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    f.write(line)
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

Or use unique files per spider: `{area}_{source}_listings.jsonl`

---

### 2. Unbounded Memory Growth in DuplicateFilterPipeline
**File:** `property_scraper/pipelines.py:129-130`
**Severity:** CRITICAL - Memory Leak

```python
self.seen_ids = set()  # Grows unbounded
self.seen_properties = set()  # Grows unbounded
```

**Problem:** Sets grow indefinitely across spider runs. With 10,000+ listings, memory can balloon significantly.

**Impact:** OOM on long scrapes or machines with limited RAM.

**Fix:** Use LRU cache with bounded size:
```python
from functools import lru_cache
# Or use a bounded dict like cachetools.LRUCache
```

---

### 3. SQL Injection via format string in fingerprint matching
**File:** `property_scraper/pipelines.py:354-358`
**Severity:** HIGH (not exploitable externally, but code smell)

```python
self.cursor.execute(
    '''SELECT id, price_pcm, first_seen, price_change_count, property_id
       FROM listings WHERE source=? AND address_fingerprint=? AND bedrooms=?
       ORDER BY last_seen DESC LIMIT 1''',
    (source, fingerprint, bedrooms)
)
```

**Status:** Properly parameterized - NOT vulnerable. Good practice throughout.

---

### 4. Foxtons Pagination Logic Bug
**File:** `property_scraper/spiders/foxtons_spider.py:175`
**Severity:** CRITICAL - Silent Data Loss

```python
should_continue = total_count >= 100 and (...)
```

**Problem:** If Foxtons returns exactly 99 results on page 1, pagination stops even if more pages exist. The assumption that `total_count >= 100` means "more pages" is fragile.

**Impact:** Missing listings when Foxtons changes their page size.

**Fix:** Check for explicit pagination metadata or use `has_more` flag if available in JSON.

---

### 5. Missing Error Handling for Schema Migration
**File:** `property_scraper/pipelines.py:285-295`
**Severity:** CRITICAL - Startup Crash

```python
if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}. Run migrate_schema_v2.py first."
    )
```

**Problem:** If migration hasn't run, the spider crashes with no recovery path.

**Impact:** New installs fail confusingly. CI/CD pipelines fail.

**Fix:** Auto-run migration or provide clearer instructions:
```python
if missing:
    logger.error("Database schema needs migration. Running auto-migration...")
    self._run_migration()
```

---

### 6. Playwright Page Leak in Retry Path
**File:** `property_scraper/spiders/knightfrank_spider.py:221-225`
**Severity:** HIGH - Resource Leak (FIXED in recent changes)

```python
yield response.request.replace(
    meta={**response.meta, 'retry_count': retry_count + 1},
    dont_filter=True
)
return  # Page not closed!
```

**Status:** FIXED in recent changes - but same pattern exists in other spiders.

**Check:** Savills (line 191), Chestertons (line 195) - same pattern but `playwright_page` is None in these cases, so OK.

---

## HIGH Priority Issues (P1 - Fix This Week)

### 7. No Connection Pool for SQLite
**File:** `property_scraper/pipelines.py:258`
**Severity:** HIGH - Performance

```python
self.conn = sqlite3.connect(self.db_path, timeout=60)
```

**Problem:** Single connection shared across all items. WAL mode helps, but under heavy load with multiple spiders, lock contention is possible.

**Recommendation:** Use connection pooling or ensure only one spider writes at a time.

---

### 8. Rightmove Rate Limit Handling Incomplete
**File:** `property_scraper/spiders/rightmove_spider.py:141-143`
**Severity:** HIGH - Silent Failure

```python
if response.status == 429:
    self.logger.warning(f"[RATE-LIMIT] Got 429 for {area} - backing off")
    return  # Just drops the page!
```

**Problem:** On 429, the page is silently skipped. No retry logic.

**Impact:** Missing data when rate limited.

**Fix:**
```python
if response.status == 429:
    wait_time = response.headers.get('Retry-After', 60)
    yield response.request.replace(
        meta={**response.meta, 'retry_count': response.meta.get('retry_count', 0) + 1},
        dont_filter=True
    )
```

---

### 9. Hardcoded Area Lists
**File:** Multiple spiders
**Severity:** MEDIUM - Maintainability

Each spider has its own `DEFAULT_AREAS` list that can drift out of sync.

**Fix:** Move to shared config in `registry.py` or a settings file.

---

### 10. No Validation of Extracted Data Before Yield
**File:** `property_scraper/spiders/rightmove_spider.py:365`
**Severity:** MEDIUM - Data Quality

Items are yielded even with missing critical fields (price=0, no address, etc.).

**Fix:** Add validation before yield:
```python
if not item.get('price_pcm') or not item.get('address'):
    self.logger.warning(f"Skipping incomplete item: {prop_id}")
    return None
```

---

## MEDIUM Priority Issues (P2)

### 11. Thread Pool Not Properly Cleaned
**File:** `property_scraper/spiders/knightfrank_spider.py:717`
**Severity:** MEDIUM - Resource Leak

```python
if self.executor:
    self.executor.shutdown(wait=False)
```

**Problem:** `wait=False` means threads may still be running when spider closes.

**Fix:** Use `wait=True` or ensure all futures are cancelled:
```python
self.executor.shutdown(wait=True, cancel_futures=True)  # Python 3.9+
```

---

### 12. Inconsistent Postcode Extraction Regex
**File:** Multiple files
**Severity:** MEDIUM - Data Quality

Different regex patterns used:
- `r'([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d?[A-Z]{0,2}$'` (strict, end-anchored)
- `r'([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d[A-Z]{2}'` (full postcode)
- `r'[A-Z]{1,2}\d{1,2}'` (loose)

**Fix:** Centralize in `utils/postcode.py`:
```python
POSTCODE_REGEX = re.compile(r'([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d[A-Z]{2}', re.I)
```

---

### 13. Missing Floorplan URL Validation
**File:** `property_scraper/spiders/rightmove_spider.py:481-523`
**Severity:** LOW - Data Quality

Floorplan URLs extracted but not validated (could be relative, malformed, or 404).

**Fix:** Add URL validation:
```python
from urllib.parse import urlparse
if floorplan_url and urlparse(floorplan_url).scheme in ('http', 'https'):
    item['floorplan_url'] = floorplan_url
```

---

### 14. Potential Division by Zero
**File:** `property_scraper/spiders/foxtons_spider.py:329`
**Severity:** LOW - Runtime Error

```python
sqft_pct = (self.stats['sqft_found'] / self.stats['total'] * 100) if self.stats['total'] else 0
```

**Status:** Already handled correctly. Good pattern.

---

### 15. CLI Doesn't Wait for All Subprocesses
**File:** `cli/main.py:198-205`
**Severity:** MEDIUM - Orphan Processes

```python
result = subprocess.run(cmd, ..., timeout=3600)
```

**Problem:** If CLI is killed (Ctrl+C), the scrapy subprocess may continue running.

**Fix:** Use signal handlers:
```python
import signal
def handle_sigint(sig, frame):
    process.terminate()
    sys.exit(1)
signal.signal(signal.SIGINT, handle_sigint)
```

---

### 16. Chestertons `parse_search` Single Point of Failure
**File:** `property_scraper/spiders/chestertons_spider.py:175-276`
**Severity:** MEDIUM - Fragile

The entire scrape happens in one browser session. If it crashes mid-way through 150 "Load More" clicks, all progress is lost.

**Fix:** Periodically yield items and checkpoint progress, or use pagination URLs if available.

---

### 17. No Graceful Degradation for OCR Failures
**File:** `property_scraper/spiders/knightfrank_spider.py:644-712`
**Severity:** LOW - Silent Failure

OCR errors are silently caught and logged at debug level. No metrics on OCR success rate.

**Fix:** Track OCR failures in stats:
```python
self.stats['ocr_failures'] = 0
# In except block:
self.stats['ocr_failures'] += 1
```

---

### 18. CLI `--dry-run` Still Creates Log Files
**File:** `cli/main.py:191-194`
**Severity:** LOW - Unexpected Behavior

Dry run creates log files in `logs/` directory even though no data is saved.

**Fix:** Skip log creation or use `/tmp` for dry runs.

---

## Code Quality Observations

### Positive Patterns
1. Consistent use of parameterized SQL queries (no SQL injection)
2. Good logging throughout with structured prefixes
3. SAVEPOINT usage for per-item rollback
4. Proper async/await patterns in Playwright spiders
5. `safe_evaluate()` and `safe_click()` helper functions with timeouts
6. COALESCE for sticky field updates (prevents data loss)
7. WAL mode and busy_timeout for SQLite concurrency

### Suggested Improvements
1. Add type hints to all functions (currently partial)
2. Add docstrings to all public methods
3. Create shared utility modules for common patterns
4. Add integration tests for full scrape workflow
5. Consider moving to async SQLite (aiosqlite) for better concurrency

---

## Summary Table

| Issue # | Severity | File | Description | Status |
|---------|----------|------|-------------|--------|
| 1 | CRITICAL | pipelines.py:199 | JSON append race condition | Open |
| 2 | CRITICAL | pipelines.py:129 | Unbounded memory in dedupe | Open |
| 3 | N/A | pipelines.py | SQL injection | Not vulnerable |
| 4 | CRITICAL | foxtons_spider.py:175 | Pagination assumption | Open |
| 5 | CRITICAL | pipelines.py:285 | Schema migration crash | Open |
| 6 | HIGH | knightfrank_spider.py | Page leak in retry | FIXED |
| 7 | HIGH | pipelines.py:258 | No connection pooling | Open |
| 8 | HIGH | rightmove_spider.py:141 | 429 not retried | Open |
| 9 | MEDIUM | Multiple | Hardcoded area lists | Open |
| 10 | MEDIUM | rightmove_spider.py | No item validation | Open |
| 11 | MEDIUM | knightfrank_spider.py:717 | Thread pool cleanup | Open |
| 12 | MEDIUM | Multiple | Inconsistent postcode regex | Open |
| 13 | LOW | rightmove_spider.py | No URL validation | Open |
| 14 | N/A | foxtons_spider.py | Division by zero | Handled |
| 15 | MEDIUM | cli/main.py | Subprocess not killed on Ctrl+C | Open |
| 16 | MEDIUM | chestertons_spider.py | Single session failure | Open |
| 17 | LOW | knightfrank_spider.py | OCR failure metrics | Open |
| 18 | LOW | cli/main.py | Dry run creates logs | Open |

---

## Recommended Priority Order

1. **P0 (Immediate):** #1 JSON race condition, #5 Schema migration UX
2. **P0 (This session):** #4 Foxtons pagination, #8 Rightmove 429 retry
3. **P1 (This week):** #2 Memory growth, #7 Connection handling
4. **P2 (Soon):** #9-18 Code quality improvements

---

*Audit performed by Claude Opus 4.5 - December 14, 2025*
