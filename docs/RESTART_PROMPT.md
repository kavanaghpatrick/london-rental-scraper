# London Rental Scraper - Session Restart Prompt

Copy this prompt into Claude to continue from where we left off:

---

## RESTART PROMPT (copy below):

```
I'm continuing work on the London Rental Scraper project. Here's where we are:

PROJECT: /Users/patrickkavanagh/rentnegotiation/scrapy_project

CONTEXT:
- PRD-001 through PRD-004 implemented (Typer CLI, fingerprinting, smart upsert, historical tracking)
- 6 critical bugs were found and fixed across multiple code reviews
- All 45 unit tests passing
- V3 code review completed - no remaining critical bugs

DATABASE STATE (Dec 6, 2025):
- 2,650 total listings across 6 sources
- 2,650 price history records
- Sources: rightmove (1,223), knightfrank (498), chestertons (386), savills (372), foxtons (141), johndwood (30)

TEST SUITE READY:
- tests/test_scrape_validation.py - Pre/post scrape validation tests
- scripts/run_full_scrape.sh - Full scrape orchestration script

MY TASK: Run today's full scrape and validate results.

STEP-BY-STEP COMMANDS TO RUN:

1. Navigate to project:
   cd /Users/patrickkavanagh/rentnegotiation/scrapy_project

2. Run pre-scrape validation:
   pytest tests/test_scrape_validation.py -v -k "pre_scrape"

3. Take baseline snapshot:
   python3 tests/test_scrape_validation.py --snapshot

4. Run all spiders (full scrape):
   # HTTP spiders (fast)
   SCRAPY_SETTINGS_MODULE=property_scraper.settings_standard scrapy crawl rightmove 2>&1 | tee logs/rightmove_$(date +%Y%m%d_%H%M%S).log
   SCRAPY_SETTINGS_MODULE=property_scraper.settings_standard scrapy crawl foxtons 2>&1 | tee logs/foxtons_$(date +%Y%m%d_%H%M%S).log

   # Playwright spiders (slower)
   SCRAPY_SETTINGS_MODULE=property_scraper.settings scrapy crawl savills 2>&1 | tee logs/savills_$(date +%Y%m%d_%H%M%S).log
   SCRAPY_SETTINGS_MODULE=property_scraper.settings scrapy crawl knightfrank 2>&1 | tee logs/knightfrank_$(date +%Y%m%d_%H%M%S).log
   SCRAPY_SETTINGS_MODULE=property_scraper.settings scrapy crawl chestertons 2>&1 | tee logs/chestertons_$(date +%Y%m%d_%H%M%S).log

5. Run post-scrape validation:
   pytest tests/test_scrape_validation.py -v -k "post_scrape"

6. Generate comparison report:
   python3 tests/test_scrape_validation.py --report

OR use the automated script:
   ./scripts/run_full_scrape.sh           # Full scrape
   ./scripts/run_full_scrape.sh --quick   # Quick test (limited pages)
   ./scripts/run_full_scrape.sh --http    # HTTP spiders only (faster)

Please run these commands and show me the results.
```

---

## QUICK REFERENCE

### Test Commands
```bash
# Pre-scrape tests (run BEFORE scraping)
pytest tests/test_scrape_validation.py -v -k "pre_scrape"

# Post-scrape tests (run AFTER scraping)
pytest tests/test_scrape_validation.py -v -k "post_scrape"

# Data integrity tests (run anytime)
pytest tests/test_scrape_validation.py -v -k "integrity"

# All tests
pytest tests/test_scrape_validation.py -v
```

### Snapshot Commands
```bash
# Save pre-scrape snapshot
python3 tests/test_scrape_validation.py --snapshot

# Generate comparison report
python3 tests/test_scrape_validation.py --report

# Show current stats
python3 tests/test_scrape_validation.py --stats
```

### Spider Commands
```bash
# HTTP spiders (use settings_standard)
SCRAPY_SETTINGS_MODULE=property_scraper.settings_standard scrapy crawl rightmove
SCRAPY_SETTINGS_MODULE=property_scraper.settings_standard scrapy crawl foxtons

# Playwright spiders (use settings)
SCRAPY_SETTINGS_MODULE=property_scraper.settings scrapy crawl savills
SCRAPY_SETTINGS_MODULE=property_scraper.settings scrapy crawl knightfrank
SCRAPY_SETTINGS_MODULE=property_scraper.settings scrapy crawl chestertons
```

### Database Queries
```bash
# Check counts by source
sqlite3 output/rentals.db "SELECT source, COUNT(*), SUM(CASE WHEN size_sqft > 0 THEN 1 ELSE 0 END) as with_sqft FROM listings GROUP BY source;"

# Check today's activity
sqlite3 output/rentals.db "SELECT COUNT(*) as new_today FROM listings WHERE first_seen LIKE '$(date +%Y-%m-%d)%';"
sqlite3 output/rentals.db "SELECT COUNT(*) as updated_today FROM listings WHERE last_seen LIKE '$(date +%Y-%m-%d)%';"

# Check price changes today
sqlite3 output/rentals.db "SELECT COUNT(*) as price_changes FROM price_history WHERE recorded_at LIKE '$(date +%Y-%m-%d)%';"
```

---

## FILES CREATED IN THIS SESSION

1. `tests/test_scrape_validation.py` - Comprehensive validation test suite
2. `scripts/run_full_scrape.sh` - Automated scrape orchestration script
3. `docs/RESTART_PROMPT.md` - This file

## EXPECTED SCRAPE BEHAVIOR

After running spiders:
- **Existing listings**: `last_seen` updated, `first_seen` preserved
- **New listings**: Both `first_seen` and `last_seen` set to now
- **Price changes**: Logged to `price_history` table
- **Removed listings**: Stay in DB but marked as `is_active=0` after mark-inactive

## SUCCESS CRITERIA

After scraping, you should see:
1. Pre-scrape tests: 9/9 passed
2. Post-scrape tests: 6/6 passed (or show explanation for any failures)
3. Data integrity tests: 7/7 passed
4. Comparison report showing new/updated listings
