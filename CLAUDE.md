# CLAUDE.md - London Rental Property Scraper

> Scrapy-based scraping framework for London rental listings with price prediction models.

---

## Quick Reference

```bash
# Explore a new site before building a spider
python site_explorer.py savills                    # Quick exploration
python site_explorer.py rightmove --floorplans     # Deep floorplan analysis
python site_explorer.py custom --url "https://example.com" --detail-pattern "/property/"

# Run any spider
scrapy crawl <spider_name> -a max_properties=500

# Common spiders
scrapy crawl savills -a max_pages=84      # Best sqft coverage (99.9%)
scrapy crawl knightfrank -a max_properties=500
scrapy crawl foxtons
scrapy crawl rightmove -a areas=Chelsea,Kensington

# Enrichers (add sqft to existing listings)
scrapy crawl rightmove_enricher -a limit=100

# Check database
sqlite3 output/rentals.db "SELECT source, COUNT(*), SUM(CASE WHEN size_sqft > 0 THEN 1 ELSE 0 END) FROM listings GROUP BY source;"
```

---

## Project Structure

```
scrapy_project/
├── site_explorer.py       # Unified site exploration tool (run before building spiders)
├── property_scraper/
│   ├── spiders/           # All spiders live here
│   │   ├── savills_spider.py      # Playwright, click pagination
│   │   ├── knightfrank_spider.py  # Playwright, offset pagination
│   │   ├── chestertons_spider.py  # Playwright, "Load More" button
│   │   ├── foxtons_spider.py      # Standard HTTP, __NEXT_DATA__
│   │   ├── rightmove_spider.py    # Standard HTTP, __NEXT_DATA__
│   │   └── rightmove_enricher.py  # Enriches existing DB records
│   ├── items.py           # PropertyItem schema
│   ├── pipelines.py       # Clean → Dedupe → JSON → SQLite
│   ├── middlewares.py     # User-agent rotation, rate limiting
│   ├── settings.py        # Playwright-enabled settings
│   └── settings_standard.py  # Non-Playwright settings
├── output/
│   └── rentals.db         # SQLite database
├── exploration/           # Output from site_explorer.py
├── docs/
│   └── PRD_site_explorer.md  # Design doc for site explorer
├── rental_price_models_v7.py  # Best model: Optuna-tuned XGBoost (R²=0.908)
└── rental_price_models_v6.py  # Previous best model
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
CREATE TABLE listings (
    id INTEGER PRIMARY KEY,
    source TEXT,           -- savills, knightfrank, rightmove, etc.
    property_id TEXT,      -- Unique per source
    url TEXT,
    address TEXT,
    postcode TEXT,         -- SW1, W8, NW3, etc.
    area TEXT,             -- Chelsea, Kensington, etc.
    price_pcm INTEGER,
    price_pw INTEGER,
    bedrooms INTEGER,
    bathrooms INTEGER,
    size_sqft INTEGER,     -- Critical for model training
    property_type TEXT,
    furnished TEXT,
    features TEXT,         -- JSON string
    summary TEXT,
    description TEXT,
    latitude REAL,
    longitude REAL,
    let_agreed BOOLEAN,
    agent_name TEXT,
    added_date TEXT,
    scraped_at TEXT,
    UNIQUE(source, property_id)
);
```

---

## Pipeline Order

1. **CleanDataPipeline** - Normalizes prices, cleans whitespace, ensures timestamps
2. **DuplicateFilterPipeline** - Filters by `source:property_id` within session
3. **JsonWriterPipeline** - Writes `{area}_listings.jsonl` files
4. **SQLitePipeline** - Upserts to SQLite with `INSERT OR REPLACE`

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

| Source | Total | With sqft | Coverage | Avg Price | Avg Sqft |
|--------|-------|-----------|----------|-----------|----------|
| savills | 712 | 711 | 99.9% | £7,284 | 1,413 |
| rightmove | 2,092 | 591 | 28.3% | £13,241 | 719 |
| knightfrank | 545 | 505 | 92.7% | £9,446 | 1,555 |
| chestertons | 451 | 320 | 71.0% | £8,302 | 1,277 |
| foxtons | 218 | 214 | 98.2% | £6,099 | 1,020 |

**Total: 2,341 properties with sqft data for model training**

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

## Development Tips

1. **Always clear cache** before debugging pagination: `rm -rf .scrapy/httpcache/<spider>`
2. **Test with small limits first**: `-a max_pages=3` or `-a max_properties=50`
3. **Check logs for extraction patterns**: Look for `[DISCOVERY]` lines
4. **For React sites**: Assume URL params don't work, use click navigation
5. **For sqft**: Premium agents (Savills, Knight Frank) have best coverage
