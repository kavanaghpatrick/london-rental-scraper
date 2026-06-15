# CLEANUP MANIFEST — 2026-06-15

Repo hygiene pass by **janitor** (team service-rebuild, task #1).

**Policy:** ARCHIVE-NOT-DELETE. Files moved to `archive/_cleanup_2026-06-15/` (never deleted).
HARD-DELETE applied **only** to caches. No git commits made.

**Scope of action:** inner repo `/Users/patrickkavanagh/rentnegotiation/scrapy_project` only.
Outer-repo observations are listed under "Out of scope / flagged" — NOT acted on.

**Safety rule applied:** every `.py` was grepped for imports/references across the
repo (excluding caches, `archive/`, `.git`) before moving. Nothing still imported or
documented as an operational tool was moved. Post-move verification: `import shared_constants`
succeeds and `pytest tests/ --collect-only` collects 74 tests with no import errors.

---

## 1. HARD-DELETED (caches only — regenerable, never archived)

| Path | Size freed | Notes |
|------|-----------|-------|
| `.scrapy/` (httpcache) | ~247 MB | Scrapy HTTP cache; regenerated on next crawl |
| `catboost_info/` | ~180 KB | CatBoost training log dir; regenerated on training |
| `.pytest_cache/` | ~32 KB | Pytest cache; regenerated on next run |
| `__pycache__/` (11 dirs, all nested) | ~288 KB | Python bytecode; regenerated on import |

All four cache types were confirmed **not git-tracked** before deletion (0 tracked cache files).

---

## 2. ARCHIVED — moved to `archive/_cleanup_2026-06-15/`

### 2a. `dead_scripts/` — superseded / orphan one-offs (tracked → `git mv`)

| File | Reason | Verification |
|------|--------|-------------|
| `explore_savills.py` | Explicitly superseded by `site_explorer.py` per `docs/PRD_site_explorer.md` ("consolidate 6 separate explore scripts into a single `site_explorer.py`") | not imported anywhere |
| `test_savills_extraction.py` | Root-level manual DRY-RUN Playwright script (not a pytest test in `tests/`); orphan, zero references. Removing it also prevents bare `pytest` from trying to import playwright at root | not imported; real suite is `tests/` |
| `merge_and_delete_duplicates.py` | One-off dedupe script; zero doc/import/CLI references (superseded by `dedupe_cross_source.py` + `scripts/dedupe_same_source.py`) | not imported, not documented |
| `sw1_chestertons_analysis.py` | One-off ad-hoc area analysis; zero references | not imported, not documented |
| `chestertons_deep_dive.py` | One-off ad-hoc exploration; zero references | not imported, not documented |

### 2b. `loose_logs/` — stale logs & captured stdout (untracked, gitignored → plain `mv`)

| File | Type |
|------|------|
| `chestertons_run.log` | stale run log |
| `knightfrank_run.log` | stale run log |
| `knightfrank_run2.log` | stale run log |
| `knightfrank_run3.log` | stale run log |
| `knightfrank_run4.log` | stale run log |
| `enricher_log.txt` | stale enricher output |
| `model_output.txt` | captured stdout |
| `model_training_output.txt` | captured stdout |
| `ocr_enrichment_output.txt` | captured stdout |
| `scrape_full_output.txt` | captured stdout |

(All were already gitignored via `*.log` / `*.txt`, so they were untracked; moved with plain `mv`.)

### 2c. `empty_dbs/` — dead DB stub (untracked → plain `mv`)

| File | Reason |
|------|--------|
| `london_rentals.db` | **0-byte** empty stub; untracked; referenced by nothing. Canonical DB is `output/rentals.db` (dataeng-owned, untouched) |

---

## 3. `.gitignore` UPDATES (inner repo)

- Added `.pytest_cache/` under a new `# Pytest` section (was the only cache type not listed).
- Existing rules already cover: `__pycache__/`, `*.py[cod]`, `.scrapy/`, `*.log`, `logs/`,
  `catboost_info/`, `.env` / `.env.local` / `.env*.local`, `.DS_Store`, `archive/`, `*.pkl`,
  `exploration/`, `savills_exploration/`, `*.txt` (with `!requirements.txt`),
  `output/rentals.db` + `output/*_backup*.db`. No changes needed there.

Also deleted 3 untracked OS-cruft `.DS_Store` files (`./`, `output/`, `exploration/`) —
already gitignored; rules permit touching `.DS_Store`.

---

## 4. DELIBERATELY KEPT (NOT archived)

**Shared dependency:**
- `shared_constants.py` — imported by `rental_price_models_v15/v16/v20.py`. **Must stay.**

**Documented operational tools** (referenced in CLAUDE.md / AGENTS.md / PRD docs / CLI / automation):
- `dedupe.py`, `dedupe_cross_source.py` (CLAUDE.md cross-source dedupe workflow)
- `migrate_schema_v2.py` (referenced by `pipelines.py` error message + 3 PRD docs)
- `backfill_fingerprints.py` (PRD-002)
- `site_explorer.py` (current explorer tool, PRD_site_explorer.md)
- `predict_price.py`, `standardize_data.py`, `inherit_sqft.py`, `get_floorplan.py`,
  `floor_analysis.py`, `extract_amenities.py`, `market_duration_analysis.py`,
  `market_duration_by_agency.py`, `batch_floorplan_ocr.py`

> Note: `features_enricher.py`'s `self.extract_amenities(...)` is a **class method** (defined
> in that spider), NOT an import of root `extract_amenities.py` — verified before deciding.

---

## 5. MODEL-VERSION SCRIPTS — losers archived (canonical = v20, lead-confirmed)

**Canonical = v20**, confirmed with the user by the lead (the user reserved the call and had
leaned v15, but confirmed v20). Lead gave the GREEN LIGHT to archive the loser set.

> History: I archived these 5 once on modeler's decision, then REVERSED when the lead put it on
> hold for user confirmation, then RE-ARCHIVED on the lead's green light. Net result below is
> the final state. Imports were re-verified before each move.

### 5a. ARCHIVED → `archive/_cleanup_2026-06-15/old_model_scripts/`

All model-version losers AND the now-decommissioned v15 group live here, **flat** (the flat
layout matters — see the harness note below). 10 files total:

**5 bake-off losers** (archived on the green light):
| File | Reason | Moved via |
|------|--------|-----------|
| `rental_price_models_v16.py` | bake-off loser | `mv` (untracked) |
| `train_model_v16_demo.py` | throwaway demo | `mv` (untracked) |
| `train_v19_extend_v18.py` | v19 loser (brittle — loaded features from chrome-ext JSON) | `mv` (untracked) |
| `add_mews_features_v19.py` | v19 helper, loser | `mv` (untracked) |
| `train_model_postgres.py` | self-references only; not in any workflow | `git mv` (tracked) |

**v15 group** (archived after the serving cutover — see 5d):
| File | Moved via |
|------|-----------|
| `rental_price_models_v15.py` | `git mv` (tracked) |
| `rental_price_model.py` (symlink → `rental_price_models_v15.py`) | `git mv` (tracked); resolves inside the archive since its target moved alongside it |
| `rental_model_v15.pkl`, `rental_model_v15_features.pkl`, `rental_model_v15_info.pkl` | `mv` (untracked; were in `output/`) |

> **Harness-import note (resolved):** modeler's two KEPT reproducibility harnesses
> (`bakeoff_v15_v16_v19_v20.py`, `bakeoff_stability.py`) `import rental_price_models_v15/v16` +
> `train_v19_extend_v18`. modeler added a sys.path shim that appends **exactly**
> `archive/_cleanup_2026-06-15/old_model_scripts` — so the archived modules **must sit flat in
> that dir**, NOT in a subfolder. (I first put the v15 group in a `v15_group/` subdir, which
> broke the harness `import rental_price_models_v15`; I flattened it into `old_model_scripts/`
> and re-verified `import bakeoff_v15_v16_v19_v20` resolves all modules cleanly.) **No
> serving/runtime code is affected** — `predict_one.py`/`generate_predictions.py` load the
> canonical pkl. Did not edit modeler-owned files.

### 5b. KEPT — do NOT archive

| File | Why kept |
|------|----------|
| `rental_price_models_v20.py` | **CANONICAL** feature-engineering source; `canonical_predict.py` imports it (line 50), plus `retrain_canonical.py` + extension export depend on it |
| `canonical_predict.py` | canonical serving predictor (imports v20); modeler keeper |
| `generate_residuals.py` | still referenced; out of scope (note: `predict_price.py` was later archived per task #27 — see §5e) |
| `bakeoff_v15_v16_v19_v20.py`, `bakeoff_stability.py`, `retrain_canonical.py` | modeler's reproducibility scripts |

### 5c. `chrome-extension/api/` — RESOLVED by artifacts (not janitor's scope)

The 4 superseded api/ JSON (`model_v16/v19.json` + `features_v16/19.json`) were archived by
**artifacts**, which owns `chrome-extension/`, in commit `757b10e` "Archive stale v16/v19
extension model artifacts (canonical = v20)" → they now live at
`archive/_cleanup_2026-06-15/stale_extension_models/`. (I had attempted a `git mv` of these
into my archive earlier, but it was effectively a **no-op** — artifacts' 757b10e had already
moved them out of `chrome-extension/api/`. Verified by artifacts + me via `git ls-files`: the 4
files exist in **exactly one** place, artifacts' `stale_extension_models/`; no duplicates, and my
`old_model_scripts/chrome-extension-api/` subdir does **not** exist.)

Current `chrome-extension/api/` (verified): only `model.json` + `features.json` (the **canonical
v20** pair — modeler verified `model.json` = 1,500-tree XGBoost Booster, 135 features, and
`features.json` byte-identical to `output/rental_model_canonical_features.pkl` order) plus
`predictions.json` + `similar_listings.json`. **No janitor action — fully artifacts' domain, closed.**

### 5d. v15 group — ARCHIVED (serving cutover confirmed)

Done. The v15 group (`rental_price_models_v15.py` + symlink + the 3 `output/rental_model_v15*.pkl`)
was archived once modeler relayed the lead's "v15 group now safe" — automation (#22) repointed
`predict.yml`/`generate-predictions.yml` to `scripts/predict_one.py`/`generate_predictions.py`,
which load `output/rental_model_canonical.pkl`. Re-verified at archive time: **zero live v15
loads** (the only remaining `rental_model_v15` strings are migration comments/docstrings in
`predict.yml`, `predict_one.py`, `shared_constants.py` — harmless stale doc pointers; not mine
to edit). Post-move checks: canonical pkl intact, `shared_constants` imports, harnesses import
all modules via the shim, `pytest tests/` collects 74 tests.

**→ Model-version drift fully resolved: ONE canonical version (v20) everywhere.**

### 5e. ARCHIVED → `archive/_cleanup_2026-06-15/legacy_clis/` (2 orphaned standalone prediction CLIs) — task #27

Both are unreferenced standalone single-property prediction CLIs, **redundant** with the
canonical single-prediction path (`scripts/predict_one.py` / `canonical_predict.py` on v20).
Task #27 directs: archive if unreferenced/dead (preferred), repoint only if referenced-and-worth-keeping.
**Verified unreferenced** (grepped CLAUDE.md, AGENTS.md, `cli/`, `docs/`, `automation/`,
`.github/`, `scripts/`, and all Python imports — zero live references to either). → ARCHIVED.
(Subdir named `legacy_clis/` rather than `broken_v14_clis/` because only one of the two is
actually v14-broken; the other is obsolete-but-not-broken — see below.)

| File | Reason | Moved via |
|------|--------|-----------|
| `scripts/predict_rent.py` | **Broken from root** — loads `rental_model_v14.pkl`/`_info`/`_features`, now only in `archive/old_models/`. V14-era CLI (docstring: "V14 Model, MAPE ~24%"). Flagged in `MODEL_DECISION.md:127`. | `git mv` (tracked) |
| `predict_price.py` | **Obsolete, not broken** — a **V7-era** standalone CLI (docstring: "Uses V7 XGBoost model") that trains a V7 model in-process (`train_model()`→`XGBRegressor`); runs but is superseded by the canonical v20 path. Unreferenced. Lead ruled archive (obsolete legacy V7); modeler (model-side owner) blessed it. | `git mv` (tracked) |

> **Decision: archive over repoint** for both. Neither is referenced or worth keeping as a
> standalone, and `predict_one.py`/`canonical_predict.py` already cover single-property
> prediction on canonical v20. A repoint wasn't warranted (predict_rent is v14-schema, predict_price
> trains its own V7 — neither maps cleanly onto canonical_predict without a rewrite that duplicates
> predict_one). Safety: zero live imports of `predict_rent`/`predict_price`; post-archive
> `pytest tests/` collects 74, canonical pkl + keepers (`canonical_predict.py`,
> `rental_price_models_v20.py`, `retrain_canonical.py`, `predict_one.py`,
> `generate_predictions.py`) intact.
>
> **Note on automation's flag:** automation reported BOTH files load `rental_model_v14.pkl`. Only
> `predict_rent.py` actually does; `predict_price.py` loads no pkl (trains V7 in-process — its `def
> predict_rent()` at line 371 is an unrelated same-named function). Both are archived anyway, but
> for the right reason (dead + unreferenced + redundant), not "v14-broken", for `predict_price.py`.
> This supersedes the earlier "KEEP predict_price.py" note (that was before task #27's
> archive-redundant-CLIs directive + the confirmed-unreferenced check).

### 5f. ARCHIVED → `archive/_cleanup_2026-06-15/stale_model_pkls/` (4 orphaned `output/` pkls)

modeler's final model-surface sweep found 4 leftover model `.pkl` artifacts in `output/` from
before the canonical convergence. **Independently re-verified: zero live loaders** (no
`open()`/`load` of `rental_model_v16.pkl` or `rental_model_v20.pkl` in any `.py`/`.yml` outside
comments; `daily-scrape.yml` `git add -f`s `rental_model_canonical.pkl`, not v20). Archived:

| File | Reason | Moved via |
|------|--------|-----------|
| `output/rental_model_v16.pkl` + `_features.pkl` | v16 bake-off loser's leftover artifacts (script already archived) | `mv` (untracked, `.pkl` gitignored) |
| `output/rental_model_v20.pkl` + `_features.pkl` | OLD pre-canonical v20 artifacts, superseded by `rental_model_canonical.pkl` (the v20 retrained on the recency-independent set) | `git mv` (tracked) |

> **Caveat (modeler, non-blocking):** `rental_price_models_v20.py:1209-1214` still *writes*
> `output/rental_model_v20.pkl` if someone runs that trainer standalone as `__main__`. So the
> archived stale copy could only reappear via a manual standalone run — which shouldn't happen
> (`retrain_canonical.py` is the canonical path). Archiving the current stale copy is safe.

**KEPT (live canonical):** `output/rental_model_canonical.pkl` + `_features.pkl` + `_meta.json`.
Post-move: `output/` is **canonical-only** for model pkls; canonical serving artifact intact;
`pytest tests/` collects 74. → model surface fully clean.

`archive/old_models/` (pre-existing) already holds v2/v10/v14 `.py` + v11–v14 `.pkl`.
`archive/api-valuate/` (pre-existing) holds v15 + v20 `.pkl` bundles.

---

## 6. OUT OF SCOPE / FLAGGED (outer repo — not acted on)

The outer repo `/Users/patrickkavanagh/rentnegotiation` (separate `.git`, embedded — not a
submodule, no `.gitmodules`) has its own loose files. Per strict ownership these were **left
untouched**; flagged here for the lead / dataeng to decide:

- `/Users/patrickkavanagh/rentnegotiation/rentals.db` — 86 KB, **stale Dec-5 copy**, distinct
  from the canonical inner `scrapy_project/output/rentals.db` (14 MB, dataeng-owned). Candidate
  for archive, but it's the outer repo's DB and dataeng owns DB decisions — left as-is.
- Outer-root loose planning docs: `RENTAL_SCRAPING_PLAN.md`, `SCRAPING_PLAN.md`,
  `ASYNC_*.md`, plus `async_scraping_examples.py`, `custom_rate_limited_scraper.py`, `db.py`,
  `schema*.sql`, `sample_data.sql` — appear to be an older prototype superseded by the inner
  `scrapy_project`. Consolidation deferred to lead.
- Outer repo has **no `.gitignore`** and tracks `.DS_Store`. Recommend adding one if the outer
  repo is kept.

Inner-repo `docs/` already exists and is populated (18 `.md` + assets); the only loose root
`.md` files are `CLAUDE.md` and `AGENTS.md`, which are operational/agent-instruction files that
**belong at root** — so no `.md`-consolidation move was needed in the inner repo.
