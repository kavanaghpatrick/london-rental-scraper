# DATA LAYER CONTRACT

**Owner:** `dataeng` (Task #2) — sole writer of the canonical DB.
**Status:** PUBLISHED 2026-06-15. Foundation for Model (#6) + Serving (#8).
**Last verified:** 2026-06-15 against `scrapy_project/output/rentals.db`.

---

## 1. Canonical store

| | |
|---|---|
| **Canonical DB** | `scrapy_project/output/rentals.db` (SQLite, WAL mode, 14.2 MB) |
| **Listings** | **10,048** (verified) |
| **Active** | **7,763** (77.3%) — cycle-relative `is_active`, see §5. Reconstructable any time via the rule in §5.1. |
| **Sources** | 6: rightmove, chestertons, knightfrank, foxtons, savills (the 5 wired estate-agent spiders) + **johndwood** (15 rows, legacy, all inactive — historical data only, no active spider, not in `cli/registry.py`). |
| **Sole writer** | `dataeng`. Scrapers write here *during a live crawl only* via the SQLite pipeline. No agent edits this file out-of-band. |
| **Readers** | model trainers, dashboard/serving (via Postgres mirror), analysis scripts. **Read copies, not the live file.** |

> ## ⛔ `mark-inactive` on this frozen snapshot — read before touching `is_active`
> This DB has not been scraped since **2026-01-16**. A WALL-CLOCK `mark-inactive` (comparing `last_seen` to `datetime('now')`) on a 5-month-old snapshot marks **100% of listings inactive** (`SUM(is_active)=0`), which empties the dashboard and breaks the model (trains on `is_active=1`). **This happened once (2026-06-15, during pipeline validation) and was restored** to 7,763 active.
>
> **Now guarded at BOTH entry points** (automation, 2026-06-15): `daily_pipeline._mark_inactive_listings` AND `cli.main mark-inactive` each have a triple-layer guard — (1) frozen-snapshot detection (refuses/skips if `max(last_seen)` is older than the window), (2) **cycle-relative cutoff `max(last_seen)−days`**, NOT `datetime('now')`, and (3) abort if it would flip >50% of active rows. The CLI **refuses loudly** (`typer.Exit(1)`); the pipeline skips with 0. `--force` overrides the *refusal* but **keeps the cutoff cycle-relative** — even `--force --execute` cannot re-zero the snapshot. So the original footgun is closed in code, not just docs.
>
> Still: do not hand-write a raw `UPDATE ... is_active=0 WHERE last_seen < datetime('now')` against canonical — that bypasses the guarded commands. To set/repair `is_active`, use the §5.1 rule. `migrate_schema_v2.py`'s backfill is also wall-clock and is NOT guarded — don't re-run its backfill on this snapshot.

### Decoy / stale DB files — DO NOT USE

| Path | Size | Verdict |
|---|---|---|
| `scrapy_project/output/rentals.db` | 14.2 MB | **CANONICAL** |
| `rentals.db` (repo root) | 86 KB | **STALE DECOY** — sample-data schema (`properties`, `agents`, `rooms`). Written by root `db.py init` from `schema.sql`/`sample_data.sql`. **Unrelated to production.** Do not read or write. |
| `scrapy_project/london_rentals.db` | 0 bytes | **EMPTY** — never populated; touched 2026-01-15. Safe to delete (defer to repo-hygiene #1). |
| `scrapy_project/output/rentals_backup_*.db` | various | Historical backups. Read-only reference. |
| `scrapy_project/output/backups/rentals_20251209.db` | 11.4 MB | Dated backup. Read-only. |
| `scrapy_project/output/rentals.db.pre_dataeng_<ts>.bak` | 14.2 MB | Safety backup taken by dataeng before this audit. |

> **Root `db.py` / `schema.sql` / `schema_rentals.sql` / `sample_data.sql` describe a DIFFERENT, legacy schema** (`rental_listings`, `raw_listings`, `properties`, `agents`, `rooms`, `epc_certificates`). They are NOT the production schema. The production schema is defined by `property_scraper/pipelines.py` (`SQLitePipeline._create_full_schema`). Treat root schema files as dead code pending repo hygiene (#1).

---

## 2. Production schema (source of truth = `pipelines.py`)

The canonical write path is **`property_scraper/pipelines.py` → `SQLitePipeline`**. Its `_create_full_schema()` / `_ensure_schema()` define the live schema. The Postgres mirror (`pipelines_postgres.py`) and the dashboard `init-db/route.ts` must match it.

### Table `listings` (54 columns)

Identity / source:
`id` (PK, autoincr), `source`, `property_id`, `url`, `area`, `UNIQUE(source, property_id)`

Price: `price`, `price_pw`, `price_pcm`, `price_period`
Location: `address`, `postcode`, `latitude`, `longitude`, `postcode_normalized`, `postcode_inferred`
Structure: `bedrooms`, `bathrooms`, `reception_rooms`, `property_type`, `property_type_std`, `size_sqft`, `size_sqm`, `furnished`, `epc_rating`
Floors/levels: `floorplan_url`, `room_details`, `has_basement`, `has_lower_ground`, `has_ground`, `has_mezzanine`, `has_first_floor`, `has_second_floor`, `has_third_floor`, `has_fourth_plus`, `has_roof_terrace`, `floor_count`, `property_levels`
Letting: `let_agreed`, `let_type`, `is_short_let`
Agent: `agent_name`, `agent_phone`, `agent_brand`
Text: `summary`, `description`, `features`
Temporal/lifecycle: `added_date`, `scraped_at`, `first_seen`, `last_seen`, `is_active`, `price_change_count`
Dedupe: `address_fingerprint`, `canonical_id`

### Table `price_history`
`id` (PK), `listing_id` (FK→listings.id), `price_pcm` NOT NULL, `price_pw`, `recorded_at` NOT NULL.
> NOTE: the column is **`price_pcm` / `price_pw`** (one row per price snapshot), NOT `old_price`/`new_price`. The schema block in `scrapy_project/CLAUDE.md` is WRONG on this — trust this contract.

### Table `scrape_runs`
Run telemetry. Key: `run_id`, `spider_name`, `started_at`, `finished_at`, `status` ∈ {running, completed, failed}, `items_scraped`, `exit_reason`, `error_summary`. `UNIQUE(run_id, spider_name)`.

### Table `scrape_events`
Per-run event log: `run_id`, `spider_name`, `event_type`, `event_time`, `message`, `details`, `severity`.

### Views / snapshot tables (read-only, may be stale)
- `listings_clean` (VIEW) = `listings` minus `duplicate_groups.duplicate_id`. Since `duplicate_groups` is currently EMPTY, this returns all 10,048.
- `area_stats_clean` (VIEW) = per-area aggregates over `listings_clean`.
- `listings_deduped` (TABLE, 1,551 rows) = **STALE snapshot** from a Dec 2025 dedupe run. Do NOT use for current data.
- `duplicate_groups` (TABLE, 0 rows), `dedupe_stats` (TABLE, Dec 2025 stats).

---

## 3. Schema parity: SQLite ↔ Postgres ↔ Dashboard

Reconciled 2026-06-15. **Action items for serving (#8) are flagged.**

| Surface | Status |
|---|---|
| SQLite `listings` (canonical) | 54 cols incl. `canonical_id`, `is_short_let`, `property_type_std`, `let_type`, `postcode_normalized`, `postcode_inferred`, `agent_brand`, `size_sqm` (REAL) |
| `pipelines_postgres.py` `_create_schema` | **DRIFT**: base CREATE lacks the V15 columns (`is_short_let`, `property_type_std`, `let_type`, `postcode_normalized`, `postcode_inferred`, `agent_brand`) and `canonical_id`. Relies on dashboard `init-db` to ALTER them in. `size_sqm REAL` ✓. |
| dashboard `init-db/route.ts` | Adds V15 columns via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. **Still missing `canonical_id`.** Also `postcode_inferred` typed `TEXT` here but `INTEGER` in SQLite — harmless but inconsistent. |
| `price_history` | SQLite has `price_pw`; Postgres `init-db` and `pipelines_postgres` OMIT `price_pw`. Low-risk (pcm is primary) but a drift. |

**Drift summary for serving (#8) to fix in Postgres `init-db`:**
1. Add `canonical_id INTEGER` to `listings`.
2. (Optional) Add `price_pw INTEGER` to `price_history` for full parity.
3. Align `postcode_inferred` type (INTEGER) — cosmetic.

`migrate_schema_v2.py` only manages the v2 columns (`address_fingerprint`, `first_seen`, `last_seen`, `is_active`, `price_change_count`) + `price_history` table; it is already applied. It does NOT cover V15 columns — those came in via the auto-migrate path in `pipelines.py` and `init-db`.

---

## 4. WRITE / READ contract

### WRITE (only these write `listings`)
- **`property_scraper/pipelines.py` (`SQLitePipeline`)** — the ONLY component that INSERTs new listings. Smart upsert keyed on `(source, property_id)` with fingerprint + content fallbacks. Sets `first_seen` (once), `last_seen` (every scrape), `is_active=1`, logs `price_history`.
- **Canonical scrapers** (`rightmove`, `savills`, `knightfrank`, `chestertons`, `foxtons`) emit `PropertyItem`s → pipeline. They do NOT touch the DB directly. These 5 estate-agent sources are the ONLY ones wired into `cli/registry.py` and thus the only sources `scrape --all` feeds into canonical. (`johndwood` is a legacy source present in historical data, all inactive, not in the active spider set.)
- **`openrent` is intentionally NOT a canonical source** — see §4.1 below.
- Enrichers (floorplan, OCR, rightmove_enricher, features_enricher) and maintenance scripts (`dedupe*.py`, `standardize_data.py`, `backfill_fingerprints.py`, `mark-inactive`) **UPDATE only** — never INSERT. (Confirmed compliant by scraper-fox-or during #5; enrichers gained an optional `-a db_path=` arg for copy-testing, default `output/rentals.db`, production behavior unchanged.)
- `dataeng` owns all of: `pipelines.py` schema, `migrate_schema_v2.py`, `dedupe*.py`, `standardize_data.py`, `backfill_fingerprints.py`, and the canonical file itself.

### 4.1 OpenRent — fixed-but-dormant, intentionally NOT canonical (lead ruling 2026-06-15)

`openrent` has **0 rows** in canonical and that is **by design**, not data loss:
- The spider is **NOT registered in `cli/registry.py`**, so `scrape --all` never runs it.
- The spider was **broken** (0% extraction on price/address/postcode) and was **fixed** during task #5 by scraper-fox-or — it now pulls ~40 clean live items with all core fields. But it was deliberately left **unwired** (wiring touches `registry.py` = automation/CLI ownership).
- **Lead ruling:** keep it **fixed-but-dormant**. OpenRent is **landlord-direct (no agent fee)** — a structurally different market segment from the 5 estate-agent sources. Folding it into the valuation model without a **distribution/calibration check** could skew rent estimates. It is a **validated, ready future opt-in pending a distribution analysis**, to be surfaced to the user as an option.
- **To activate later:** add an `openrent` SpiderConfig to `cli/registry.py` (Playwright spider, `settings.py`), THEN run a distribution analysis vs the agent sources before trusting it in the model. Do not wire it silently.

### READ
- **Model trainers** (`rental_price_models_v20.py`, `train_model_postgres.py`, etc.) read `listings`.
  - Current filter: `size_sqft>0 AND bedrooms IS NOT NULL AND price_pcm>0 AND is_active=1 AND (is_short_let=0 OR is_short_let IS NULL)`.
  - **See §5 — `is_active=1` is the wrong recency gate for training on this frozen snapshot.**
- **Dashboard / serving** read via the Postgres mirror. `is_active` drives "active listings" counts and comps.

---

## 5. ⚠️ CRITICAL for MODELER (#6): training filter vs `is_active`

This DB is a **frozen snapshot** — last real scrape was **2026-01-16**. Today is 2026-06-15.

`mark-inactive` (correctly) measures staleness from the **last scrape cycle**, not wall-clock now (using wall-clock would mark 100% of listings inactive simply because nobody has scraped in 5 months — that is meaningless). After my run, **`is_active=1` = "seen in the final scrape cycle."**

Trainable-set sizes (`size_sqft>0 AND bedrooms NOT NULL AND price_pcm>0 AND is_short_let=0`):

| Filter | Trainable rows |
|---|---|
| recency-independent (no `is_active`) | **6,402** |
| `is_active=1` BEFORE mark-inactive | 5,642 |
| `is_active=1` AFTER mark-inactive (this run) | **4,806** |

**Recommendation to modeler:** for a one-shot bake-off on this frozen snapshot, train on the **recency-independent set (6,402)** — a listing that was active on Jan 9 is still valid training signal for a price model. Filtering `is_active=1` throws away ~1,600 valid rows for no modeling benefit. Reserve `is_active` for the *serving/dashboard* "what's on the market now" view, not for model training.

If you prefer to keep `is_active=1` for reproducibility with prior versions, that's fine — just know it's 4,806 rows post-cleanup, and the shrink is a recency artifact, not data quality.

**I did NOT change any `listings` row that the model reads except flipping `is_active` 1→0 on 1,122 stale rows. No rows deleted. No prices/sizes/structure altered.**

### 5.1 Canonical `is_active` reconstruction rule (idempotent — use this, never wall-clock)

`is_active` is a **pure function of `last_seen`** while the snapshot is frozen. To set or repair it, run this against the canonical DB (it is idempotent and always yields 7,763 active):

```sql
WITH m AS (SELECT MAX(last_seen) mx FROM listings)
UPDATE listings
SET is_active = CASE
    WHEN last_seen >= (SELECT datetime(mx, '-7 days') FROM m) THEN 1
    ELSE 0
END;
```

- Reference = `MAX(last_seen)` (= last scrape cycle, `2026-01-16T10:03:27`), NOT `datetime('now')`.
- Cutoff = `MAX(last_seen) − 7 days` = `2026-01-09 10:03:27`.
- Result: **7,763 active / 2,285 inactive** of 10,048. Per-source active: rightmove 5,746, chestertons 826, knightfrank 436, foxtons 387, savills 368, johndwood 0.
- Because it is derived from `last_seen` (which the wall-clock wipe did NOT touch), this fully reconstructs `is_active` from a 100%-inactive state — restoring the `.bak` is unnecessary and insufficient (the pre-mark-inactive `.bak` is ~9,285 active and would still need this pass).
- **Once a fresh scrape lands**, `last_seen` advances and `cli.main mark-inactive --days 7` (wall-clock) becomes correct again; this frozen-snapshot rule is a stopgap until then.

---

## 6. Dedupe state (reported, NOT mass-deleted)

| Signal | Count | Meaning |
|---|---|---|
| Exact `(source, property_id)` dupes | **0** | Enforced by UNIQUE constraint. Clean. |
| `listings` with `canonical_id != id` | **175** | Flagged as cross-source duplicates pointing at a canonical record (from a prior `dedupe_cross_source --mark` run). Not deleted. |
| Fingerprint clusters spanning >1 source (active) | **62** | Cross-source dupe candidates (same property on Rightmove + agent site). Merge-eligible, not yet merged. |
| Same-source exact relists (active, same fp+beds+price) | **2** | Residual same-source relists with different IDs. Trivial. |
| `duplicate_groups` table rows | **0** | The `--remove`/`--mark` path did not persist groups here in the current DB. |
| `price_history` orphans | **0** | Referential integrity clean. |

**No duplicates were deleted.** Cross-source duplicates are intentionally retained (the dedupe strategy *merges sqft from agent → Rightmove* rather than deleting). If the modeler wants a deduped training set, prefer one record per `canonical_id` (keep the row where `canonical_id IS NULL OR canonical_id=id`).

---

## 7. What dataeng changed in this audit (2026-06-15)

1. **Backed up** canonical DB → `output/rentals.db.pre_dataeng_<ts>.bak`. Integrity check: OK.
2. **Fixed zombie run**: `scrape_runs.id=4` (knightfrank, run `20260114_170343`) `running`→`failed`, set `finished_at`, `exit_reason='zombie_cleanup'`, `error_summary`. Zero remaining `running` zombies.
3. **mark-inactive (7d from last scrape cycle)**: flipped `is_active` 1→0 on **1,122** stale listings. Active% now reflects the final scrape cycle, not wall-clock. (See §5.)
4. **Published this contract.** No schema DDL changes to the canonical SQLite file (it is already at the target schema). Postgres/dashboard drift items handed to serving (#8) in §3.
5. **Durable zombie-run fix** (task #20) in BOTH run-lifecycle writers (dataeng-owned, not the spiders): `property_scraper/extensions/audit_logger.py` (SQLite) AND `property_scraper/extensions/audit_logger_postgres.py` (Postgres). Added `_reap_stale_runs()`, called on `spider_opened` before the INSERT. Any `scrape_runs` row left `status='running'` for >`AUDIT_STALE_RUN_HOURS` (default **4h**, safely above `CLOSESPIDER_TIMEOUT`=1h) is auto-marked `failed` / `exit_reason='stale_orphan'`. Excludes the current `run_id` so concurrent spiders are never falsely reaped; non-fatal on error. Root cause (diagnosed with scraper-kf-ch): NOT a spider hang — every clean/errored/timeout close fires `spider_closed` which updates the row; a row only stays `running` after hard external process death (SIGKILL/OOM/reboot). Self-healing on next scrape; no manual cleanup. Verified with the real extension class (hard-kill orphan reaped, current run preserved, timeout-close finalizes correctly). Full writeup: **`AUDIT_LOGGER_FIX.md`**.

---

## 8. Rules of engagement (all agents)

- **Only `dataeng` writes `output/rentals.db`.** Validate against a *copy*, not the live file.
- If you need a schema change to the canonical DB, request it from `dataeng` — do not ALTER it yourself.
- Postgres `init-db` and `pipelines_postgres.py` are owned by serving (#8) but **must stay column-compatible with §2**. Coordinate drift fixes through this contract.
- Trust this file over `scrapy_project/CLAUDE.md`'s "Database Schema" and "Current Data Status" sections — both are stale (CLAUDE.md still says 2,650 listings and a wrong `price_history` schema).

---

## 9. Future data-cleanup backlog (deferred, non-blocking)

These are KNOWN data-quality items intentionally deferred — current canonical/model/serving work correctly as-is. Do NOT action without lead sign-off + the dependent agent (e.g. retrain) in the loop.

### 9.1 `postcode_normalized` is mostly NULL → district FE buckets into a fallback (flagged by modeler, #30)
- **State (verified 2026-06-15 against canonical):** `postcode_normalized` is NULL/empty on **9,174 / 10,048** rows overall (and ~93% within the v20 active-trainable frame). Of those, **~7,664 overall (~3,930 in-frame) are RECOVERABLE** — they have a clean non-empty raw `postcode` (e.g. `SW1X`, `SW1W`, `SW10`) that just wasn't normalized.
- **Impact:** model FE does `postcode_district = postcode_normalized.fillna('SW3')`, so all nulls collapse into **SW3**, inflating `postcode_freq(SW3)` to ~0.88 (not the real prime-market share). The model was FIT on this, so inference reproduces it faithfully — you CANNOT silently "fix" the column without retraining or you'll mismatch the trained freq maps.
- **Fix (dataeng domain — normalization lives in `pipelines.py` / data layer):** a backfill that re-extracts the outward district from raw `postcode` (the same regex the FE uses: `^([A-Z]+\d+[A-Z]?)`) and writes `postcode_normalized`. Recovers most real districts and de-inflates the SW3 fallback bucket.
- **Coupling:** this is a **clean→retrain** change, not a hot-patch. Sequence: (1) dataeng backfills `postcode_normalized` on a COPY, (2) modeler re-runs `retrain_canonical` + regenerates inference freq maps on the cleaned data, (3) validate prime-market discrimination improved, (4) promote. Ping modeler before/after. Until all four happen, leave the column as-is.
- **Status:** DEFERRED per lead. Logged here + in `MODEL_DECISION.md` (modeler). Not started.
