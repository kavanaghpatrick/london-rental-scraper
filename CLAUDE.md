# CLAUDE.md - London Rental Property Scraper

> Scrapy-based scraping framework for London rental listings with historical tracking and price prediction.

---

## Quick Reference

```bash
# === CLI (PREFERRED) ===
python -m cli.main scrape --all              # Run all spiders (fast, search results only)
python -m cli.main scrape --all --full       # Run all with full data (sqft + floorplans from detail pages)
python -m cli.main scrape --source savills   # Single spider
python -m cli.main scrape -s rightmove -d -f # Single spider with details + floorplans
python -m cli.main enrich-floorplans         # Fetch missing floorplan URLs (all sources)
python -m cli.main ocr-enrich                # OCR floorplans for sqft & floor data
python -m cli.main status                    # Show database stats
python -m cli.main mark-inactive --days 7    # Mark stale listings inactive

# === RECOMMENDED: FULL DATA SCRAPE ===
python -m cli.main scrape --all --full       # Slower but gets sqft + floorplans upfront
python -m cli.main dedupe --merge --execute  # Cross-source dedupe

# === ALTERNATIVE: FAST SCRAPE + ENRICHMENT ===
python -m cli.main scrape --all              # 1. Fast scrape (search results only)
python -m cli.main enrich-floorplans         # 2. Backfill floorplan URLs
python -m cli.main ocr-enrich                # 3. OCR: extract sqft & floors
python -m cli.main dedupe --merge --execute  # 4. Cross-source dedupe

# === FULL SCRAPE WITH VALIDATION ===
./scripts/run_full_scrape.sh                 # Full automated scrape
./scripts/run_full_scrape.sh --http          # HTTP spiders only (faster)
./scripts/run_full_scrape.sh --quick         # Limited pages (testing)

# === VALIDATION TESTS ===
pytest tests/test_scrape_validation.py -v -k "pre_scrape"   # Before scraping
pytest tests/test_scrape_validation.py -v -k "post_scrape"  # After scraping
python3 tests/test_scrape_validation.py --snapshot          # Save baseline
python3 tests/test_scrape_validation.py --report            # Compare changes

# === DIRECT SPIDER RUNS ===
SCRAPY_SETTINGS_MODULE=property_scraper.settings scrapy crawl savills
SCRAPY_SETTINGS_MODULE=property_scraper.settings_standard scrapy crawl rightmove

# === DATABASE ===
sqlite3 output/rentals.db "SELECT source, COUNT(*), SUM(is_active) FROM listings GROUP BY source;"
sqlite3 output/rentals.db "SELECT COUNT(*) FROM price_history WHERE recorded_at LIKE '$(date +%Y-%m-%d)%';"
```

---

## Project Structure

```
scrapy_project/
├── cli/                   # Typer CLI (PRD-001)
│   ├── main.py            # Commands: scrape, status, mark-inactive, dedupe
│   └── registry.py        # Spider configs (settings, type: playwright/http)
├── property_scraper/
│   ├── spiders/           # All spiders
│   ├── services/
│   │   └── fingerprint.py # Address fingerprinting (PRD-002)
│   ├── pipelines.py       # Smart upsert with history (PRD-003/004)
│   ├── items.py           # PropertyItem schema
│   ├── settings.py        # Playwright settings
│   └── settings_standard.py  # HTTP settings
├── tests/
│   ├── test_scrape_validation.py  # Pre/post scrape validation
│   ├── test_pipeline.py   # Pipeline unit tests
│   └── test_fingerprint.py # Fingerprint tests
├── scripts/
│   └── run_full_scrape.sh # Automated scrape orchestration
├── output/
│   └── rentals.db         # SQLite database
└── docs/
    └── RESTART_PROMPT.md  # Session restart instructions
```

---

## Spider Summaries

### savills_spider.py (Best for sqft: 99.9%)
Uses Playwright with **click-based pagination** because Savills is a React SPA where URL parameters are ignored. Navigates by clicking "Next >" button (case-insensitive match), waits for `.sv--selected` page indicator to update, then waits for `li.sv-results-listing__item` listings to load. Handles both Weekly (×52/12) and Monthly prices. Only yields listings with BOTH sqft AND price.

### knightfrank_spider.py (sqft: 93%)
Uses Playwright with **offset URL pagination** (`/all-beds;offset=48`). Each page has 48 listings with excellent sqft data in property cards. Extracts from CSS selectors - postcode from `.kf-search-result__address`, price from `.kf-search-result__price`, sqft from `.kf-search-result__size`. Premium agent with consistent data format.

### chestertons_spider.py (sqft: 71%)
Uses Playwright with **"Load More" button** pagination inside a single browser session. Bypasses Cloudflare protection using Playwright. Clicks `.load-more-btn` repeatedly within `parse_search` async method. Property cards use `.pegasus-property-card` selector. Lower sqft coverage because many listings don't show size in cards.

### foxtons_spider.py (sqft: 98%)
Standard HTTP spider extracting from **`__NEXT_DATA__` JSON** embedded in pages. No JavaScript rendering needed - parses the JSON directly from `<script id="__NEXT_DATA__">`. Returns up to 100 properties per page. Excellent sqft coverage because Foxtons consistently includes size data.

### rightmove_spider.py (sqft: 28% without enrichment)
Standard HTTP spider using **`__NEXT_DATA__` JSON**. Fastest spider but sqft data is often missing from search results - requires enrichment via detail page fetches. Use `rightmove_enricher` spider to backfill sqft.

### rightmove_enricher.py (enricher)
Reads existing Rightmove listings from SQLite that lack sqft, fetches their detail pages, extracts size from `nearestSize` in JSON data, and updates the database. Run with `SCRAPY_SETTINGS_MODULE=property_scraper.settings_standard`.

---

## Key Learnings & Gotchas

### Playwright Spiders (savills, knightfrank, chestertons)

| Issue | Solution |
|-------|----------|
| URL params ignored by React | Use click-based pagination within single session |
| Content not loading after click | Use `wait_for_function` for page indicator, then **`wait_for_load_state('networkidle')`**, then `wait_for_selector` for listings |
| "Next" button not found | Use **case-insensitive** text matching: `toLowerCase().includes('next')` |
| HTTP cache returning stale pages | Clear cache before runs: `rm -rf .scrapy/httpcache/<spider>` |
| Playwright page not available | Ensure `playwright_include_page: True` in meta |
| SVG className errors | Use `typeof className === 'string'` check |

### Price Handling

| Format | Conversion |
|--------|------------|
| £1,500 pcm | Direct use |
| £350 pw | `×52/12 = £1,517 pcm` |
| £369 Weekly | Same as pw |

### Sqft Extraction

```javascript
// Regex that works across all sites
const sqftMatch = text.match(/(\d+(?:,\d+)?)\s*sq\s*ft/i);
```

---

## Settings Configurations

| Setting | Playwright (settings.py) | Standard (settings_standard.py) |
|---------|--------------------------|----------------------------------|
| DOWNLOAD_HANDLERS | scrapy_playwright.handler | Default HTTP |
| CONCURRENT_REQUESTS | 8 | 8 |
| DOWNLOAD_DELAY | 2s | 2s |
| AUTOTHROTTLE | Enabled | Enabled |

**Run with non-Playwright settings:**
```bash
SCRAPY_SETTINGS_MODULE=property_scraper.settings_standard scrapy crawl rightmove_enricher
```

---

## Database Schema

```sql
-- Main listings table with historical tracking (PRD-003/004)
CREATE TABLE listings (
    id INTEGER PRIMARY KEY,
    source TEXT,               -- savills, knightfrank, rightmove, etc.
    property_id TEXT,          -- Unique per source
    address_fingerprint TEXT,  -- 16-char hash for cross-source dedupe (PRD-002)
    first_seen TEXT,           -- When listing first appeared
    last_seen TEXT,            -- When last scraped (updated each run)
    is_active INTEGER DEFAULT 1,  -- 0 = stale/removed
    price_change_count INTEGER DEFAULT 0,
    url TEXT,
    address TEXT,
    postcode TEXT,
    price_pcm INTEGER,
    size_sqft INTEGER,
    bedrooms INTEGER,
    -- ... other fields ...
    UNIQUE(source, property_id)
);

-- Price history table (PRD-004)
CREATE TABLE price_history (
    id INTEGER PRIMARY KEY,
    listing_id INTEGER,
    old_price INTEGER,
    new_price INTEGER,
    recorded_at TEXT,
    FOREIGN KEY (listing_id) REFERENCES listings(id)
);
```

**Key Behaviors**:
- `first_seen`: Set once on INSERT, never overwritten
- `last_seen`: Updated on every scrape (indicates listing still active)
- `price_history`: New row added when price changes
- `is_active`: Set to 0 by `mark-inactive` for listings not seen in N days

---

## Pipeline Order

1. **CleanDataPipeline** - Normalizes prices, cleans whitespace
2. **DuplicateFilterPipeline** - Filters by `source:property_id` within session
3. **JsonWriterPipeline** - Writes `{area}_listings.jsonl` files
4. **SQLitePipeline** - Smart upsert (PRD-003):
   - Generates `address_fingerprint` via fingerprint service
   - Checks if listing exists → UPDATE (preserve `first_seen`) or INSERT
   - Detects price changes → logs to `price_history`
   - Uses SAVEPOINT for per-item atomicity

---

## Cross-Source Deduplication

**Problem**: Rightmove is an aggregator - properties appear both on Rightmove AND on agent sites (Savills, Knight Frank, Foxtons). Same property = duplicate records.

**Solution**: `dedupe_cross_source.py`

```bash
# Analyze duplicates (dry-run)
python3 dedupe_cross_source.py --analyze

# Merge sqft from agents to Rightmove (dry-run)
python3 dedupe_cross_source.py --merge

# Actually execute merge
python3 dedupe_cross_source.py --merge --execute

# Mark duplicates with canonical_id
python3 dedupe_cross_source.py --mark --execute

# Remove duplicates (keep best record)
python3 dedupe_cross_source.py --remove --execute
```

**Duplicate Detection Criteria**:
- Same postcode district (SW1, W8, etc.)
- Price within 5%
- Same number of bedrooms
- Address similarity > 85%

**Source Priority** (for choosing canonical record):
1. savills (best sqft)
2. knightfrank
3. foxtons
4. chestertons
5. rightmove (aggregator, often missing data)

**Impact**: Cross-source merge added ~170 sqft values to Rightmove records by copying from matching agent listings.

---

## Current Data Status (Dec 2025)

| Source | Total | Active | With sqft |
|--------|-------|--------|-----------|
| rightmove | 1,223 | 1,223 | ~28% |
| knightfrank | 498 | 498 | ~93% |
| chestertons | 386 | 386 | ~71% |
| savills | 372 | 372 | ~99% |
| foxtons | 141 | 141 | ~98% |
| johndwood | 30 | 30 | varies |

**Total: 2,650 listings with historical tracking enabled**

---

## Price Prediction Models

Located in `rental_price_models_v7.py` (best) and `rental_price_models_v6.py`:

| Version | R² | MAE | MAPE | Median APE |
|---------|-----|-----|------|------------|
| V7 (Optuna XGBoost) | **0.908** | £860 | 10.5% | **4.5%** |
| V6 (Standardized) | 0.904 | £976 | 12.6% | 7.8% |
| V5 (Amenities) | 0.773 | £1,650 | 20.1% | ~15% |

**V7 Key Improvements**:
- Optuna hyperparameter tuning (30 trials)
- 65 features including price-per-sqft encodings
- 2,622 training samples with 97.8% postcode coverage

**Top Features**: luxury_size_interaction, size_postcode_interaction, size_area_interaction, premium_agent_size, postcode_district_ppsf_encoded

---

## Common Issues & Fixes

### "0 listings found on page 2+"
**Cause**: Content not loaded before extraction
**Fix**: Add proper waits:
```python
await page.wait_for_function(f'''() => {{
    const selected = document.querySelector('.pagination .selected');
    return selected && parseInt(selected.textContent) === {expected_page};
}}''', timeout=10000)
await page.wait_for_selector('.listing-item', timeout=10000)
await page.wait_for_timeout(2000)  # Extra render time
```

### "Has next: False" despite more pages
**Cause**: Case-sensitive text matching
**Fix**: Use `toLowerCase().includes('next')` instead of exact match

### Playwright page is None
**Cause**: HTTP cache hit returned cached response without Playwright
**Fix**: Clear cache or disable for Playwright spiders

### Enricher not finding sqft
**Cause**: Using wrong settings file
**Fix**: `SCRAPY_SETTINGS_MODULE=property_scraper.settings_standard scrapy crawl rightmove_enricher`

---

## Address Fingerprinting (PRD-002)

Located in `property_scraper/services/fingerprint.py`. Creates 16-char hashes for cross-source deduplication.

**Algorithm**:
1. Normalize address (lowercase, strip punctuation, expand abbreviations)
2. Extract components: flat/unit, street number (with letter suffix), street name, postcode
3. MD5 hash → first 16 chars

**Example**: `"Flat 2, 100A King's Road, SW3 4TX"` → `a1b2c3d4e5f6g7h8`

**Usage**: Listings with same fingerprint across sources are likely duplicates.

---

## Test Suite

```bash
# All unit tests (45 tests)
pytest tests/ -v

# Pre-scrape validation (run BEFORE scraping)
pytest tests/test_scrape_validation.py -v -k "pre_scrape"

# Post-scrape validation (run AFTER scraping)
pytest tests/test_scrape_validation.py -v -k "post_scrape"

# Data integrity checks (run anytime)
pytest tests/test_scrape_validation.py -v -k "integrity"
```

**Snapshot workflow** (for comparing before/after scrape):
```bash
python3 tests/test_scrape_validation.py --snapshot  # Save baseline
# ... run scrape ...
python3 tests/test_scrape_validation.py --report    # Compare changes
```

---

## Development Tips

1. **Use the CLI**: `python -m cli.main scrape --all` instead of manual scrapy commands
2. **Test with limits**: `python -m cli.main scrape --source rightmove --max-pages 3`
3. **Clear cache** before debugging: `rm -rf .scrapy/httpcache/<spider>`
4. **Check logs**: Look for `[DISCOVERY]` and `[UPSERT]` lines
5. **Playwright sites**: savills, knightfrank, chestertons (use `settings.py`)
6. **HTTP sites**: rightmove, foxtons (use `settings_standard.py`)
7. **403 errors = USE PLAYWRIGHT**: Sites like Chestertons block HTTP requests with 403. Use Playwright with headed mode for manual CAPTCHA solving:
   ```bash
   SCRAPY_SETTINGS_MODULE=property_scraper.settings_headed scrapy crawl floorplan_enricher -a source=chestertons
   ```
