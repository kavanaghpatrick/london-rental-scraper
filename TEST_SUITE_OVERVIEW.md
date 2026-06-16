# TEST SUITE OVERVIEW (#49)

**Author:** `automation`
**Date:** 2026-06-16
**Deliverable:** the consolidated CI (`.github/workflows/ci.yml`) + this map of what's
covered, how to run it, and the required-checks set.

---

## The CI: `.github/workflows/ci.yml`

Runs on every **push (main)** + **pull_request** + manual dispatch. Three independent
jobs (each a required check):

| Job | What it runs | Gates |
|-----|--------------|-------|
| `python-tests` | `pytest` (unit + self-contained) with coverage over `property_scraper`/`automation`/`cli`/`scripts` | `--cov-fail-under=5` (ratchet floor) |
| `node-parity` | regen golden from Python (`gen_feature_parity_golden.py`) → `node chrome-extension/_fixture_diff.mjs` | any JS↔Python feature/£ drift |
| `dashboard` | `npx tsc --noEmit` (type-check) + lint | TS type errors |

### Other workflows kept as their own required checks
- `extension-drift-guard.yml` — parity + vendored byte-identity (separate, path-scoped triggers).
- `neon-backup.yml` — scheduled prod backup (#38).
- `daily-scrape.yml` — runs the **data-validation** + post-scrape tests where a populated DB exists (NOT in PR CI).

---

## Test inventory (live, `tests/` — `testpaths=tests`)

| File | Tests | Kind | In PR CI? |
|------|------:|------|-----------|
| `tests/test_fingerprint.py` | 41 | pure-unit | ✅ always |
| `tests/test_pipeline.py` | 11 | self-contained (temp DB) | ✅ always |
| `tests/test_safe_delete.py` | 5 | pure-unit (destructive-op guard, #40) | ✅ always |
| `tests/test_scrape_validation.py` | 22 | **data-validation** (reads live `rentals.db`) | ⏭️ auto-skipped without DB → runs in daily-scrape |

**PR CI run today:** `92 passed, 22 skipped` (the 22 are data-validation, skipped because
CI has no `rentals.db`). Green.

### Why the unit/data split
CI has no `rentals.db` (gitignored). `tests/conftest.py` auto-skips data-validation
tests (marked `data` or in `test_scrape_validation.py`) when no populated DB is present,
so PR CI is green on **logic**, while the data-quality assertions run for real in
daily-scrape's post-scrape step (where the data owner can act on failures). Forcing
data-validation into PR CI would red the build on data state, which is wrong.

---

## Coverage matrix (current)

| Module | Coverage | Owner / gap |
|--------|---------:|-------------|
| `property_scraper/services/fingerprint.py` | **100%** | done |
| `scripts/_safe_delete.py` | **100%** | done (automation #40) |
| `property_scraper/pipelines.py` | 41% | scrapers (#45) |
| `automation/daily_pipeline.py` | 0% | **gap** — automation, follow-up |
| `cli/main.py` | 0% | **gap** — automation, follow-up |
| `property_scraper/pipelines_postgres.py` | 0% | data/sync (#46) |
| **TOTAL** | **~5%** | ratchet target below |

**The 5% is honest:** ~8,000 statements, most in spiders/pipelines/CLI not yet unit-
tested. The test-suite owners (#45 scrapers, #46 data/sync ✓, #47 model ✓, #48 API) are
filling these. The gate is set at the current floor so it catches a **regression** without
blocking PRs on a number the suite hasn't reached.

### Coverage ratchet plan (lead-set targets)
The FLOOR (`--cov-fail-under`) starts at the current baseline (5%) and ratchets UP as
tests land — never gate at the target today (it'd red every PR). Each owner bumps the
floor when their tests merge so it never slides back.

**Aspiration target ~70% on the CORE-LOGIC modules** (per the lead) — NOT on spider
network code (legitimately hard to unit-test; cover those via fixture-parse/selector
tests, not coverage %):
- `property_scraper/pipelines.py`, `property_scraper/services/fingerprint.py`
- `scripts/sync_sqlite_to_postgres.py`, `scripts/_safe_delete.py`, `merge_datasets.py`
- `canonical_predict.py`, `scripts/_canonical_features.py`, dashboard API routes

Floor trajectory: 5 → 15 (after #45 scrapers) → 30 (after #48 API) → ratchet toward 70%
on the core-logic set. Spider modules stay covered by parse-fixture tests, excluded from
the % target.

---

## How to run

```bash
# What PR CI runs (unit + self-contained; data tests skip without a DB):
pytest

# With coverage (as CI):
pytest --cov=property_scraper --cov=automation --cov=cli --cov=scripts --cov-report=term-missing

# Data-validation tests (need a populated output/rentals.db):
pytest tests/test_scrape_validation.py        # runs only if the DB exists

# Live/network tests (opt-in, excluded from CI):
pytest -m live

# JS↔Python parity (the node gate):
python3 gen_feature_parity_golden.py && node chrome-extension/_fixture_diff.mjs

# Dashboard type-check:
cd dashboard && npx tsc --noEmit
```

---

## Required-checks set (recommendation for branch protection)

**Required (block merge):** `python-tests`, `node-parity`, `dashboard` (tsc),
`extension-drift-guard / fixture-diff`, `extension-drift-guard / vendored-identity`.

**Not required (infra/data/scheduled):** `daily-scrape`, `neon-backup`,
`Generate Predictions Cache`, `Predict Fair Value` — these need prod secrets / a
populated DB / a schedule and shouldn't gate a code PR.

---

## Known gaps / handoffs
- **Lint:** Next.js 16 removed `next lint`; the dashboard has no ESLint config, so the
  lint step is non-blocking (`continue-on-error`) with a warning. **Serving** to add a
  flat ESLint config + `eslint` dep, then flip it to a hard gate.
- **automation coverage:** `daily_pipeline.py` / `cli/main.py` at 0% — follow-up unit
  tests (automation) to raise the floor.
- **#48 (API tests):** pending — once landed, add an API-test job (or fold into
  `python-tests`) + bump the coverage floor.
