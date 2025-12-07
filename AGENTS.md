# Repository Guidelines

## Project Structure & Module Organization
Core scraping logic lives in `property_scraper/`: spiders inside `property_scraper/spiders/`, items and pipelines at the top level, helper ML/rich-media routines in `property_scraper/utils/`. Reusable scripts for enrichment or modeling (e.g., `extract_amenities.py`, `rental_price_models_v*.py`) sit in the repo root, while exploratory notebooks or datasets belong in `exploration/` and `archive/`. Persisted artifacts land in `output/` (JSON/CSV) and `logs/`; scrub these before committing. Documentation drafts, including product briefs, are under `docs/`.

## Build, Test, and Development Commands
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt           # Installs Scrapy, Playwright, helpers
scrapy crawl rightmove -s SCRAPY_SETTINGS_MODULE=property_scraper.settings \
    -O output/rightmove.json                         # Main crawl with full settings
scrapy crawl savills -s SCRAPY_SETTINGS_MODULE=property_scraper.settings_nopw \
    -o output/savills.json                           # Lighter crawl without Playwright
python -m pytest property_scraper/utils/test_floorplan_extractor.py -v  # Floorplan tests
```
Use `SCRAPY_SETTINGS_MODULE=property_scraper.test_settings` when you want cheaper local runs without Playwright.

## Coding Style & Naming Conventions
Stick to Python 3.10+, PEP8, and 4-space indentation. Spiders are named `<vendor>_spider.py` with the Scrapy `name` matching the vendor (`rightmove`, `knightfrank`, etc.). Keep pipeline and middleware names descriptive (`CleanDataPipeline`, `RateLimitMiddleware`) and mirror existing logging tone (`[CONFIG]`, `[ERROR]`). Comments should capture “why” (rate limits, JSON parsing tricks) rather than “what”. Output files go under `output/<source>_<date>.json`; do not dump raw HTML in git.

## Testing Guidelines
Pytest backs the utilities; the most complete suite is `property_scraper/utils/test_floorplan_extractor.py`, which expects fixture images in `output/` or `archive/`. Name new tests `test_<behavior>` and colocate them with the module they cover. Run targeted pytest commands before pushing, and prefer `scrapy check <spider>` to ensure new spiders validate their contracts. There is no formal coverage gate, but aim to exercise every parsing helper and any new middleware branches.

## Commit & Pull Request Guidelines
The repo currently has no published history, so adopt Conventional Commit-style prefixes (`feat:`, `fix:`, `chore:`) to keep future history searchable. Each commit should explain the “why” (e.g., `feat: add OpenRent spider with playwright handler`). PRs need: a short summary of changes, reproduction steps or sample scrapy command, expected artifacts (e.g., a JSON snippet), and any scraping risks (rate limits, authentication). Link to Linear/GitHub issues when available; attach screenshots/log excerpts for crawler diffs.

## Security & Data Handling
Do not hardcode credentials, proxies, or cookies—load them from your shell environment before running Scrapy. Clean personally identifiable information from `output/` before sharing. When working with Playwright, run `playwright install chromium` locally instead of committing binaries. If you must store secrets for team members, use the shared password vault documented in `docs/` rather than `.env` files.
