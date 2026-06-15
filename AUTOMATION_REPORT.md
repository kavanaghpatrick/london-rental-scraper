# AUTOMATION REPORT — retrain-after-every-scrape + CI chain

**Author:** `automation` (Task #9)
**Date:** 2026-06-15
**Core ask delivered:** *"the model retrains after every scrape and updates, and everything is served properly"* — as a reliable, idempotent, logged loop, both locally and in CI.

---

## TL;DR

One canonical model (**v20**, `rental_model_canonical.pkl`) now flows through a single
loop end-to-end:

```
scrape → enrich → dedupe → RETRAIN(canonical) → EXPORT artifacts → SYNC to Neon → deploy
```

- **Local:** `python -m cli.main daily` (or `python -m automation.daily_pipeline`) runs the
  whole loop. `--dry-run` and `--stage` supported.
- **CI:** `daily-scrape.yml` (cron) scrapes → retrains the canonical model → commits artifacts →
  `generate-predictions.yml` fires off it and rebuilds the predictions cache from the **same**
  model.
- **Version drift killed:** every path (workflows, predictions, on-demand predict, extension)
  now loads `rental_model_canonical.pkl` / `_features.pkl`. No more v15/v16/v19/v20 split.

Everything was validated with **real runs** (no mocks). Prod writes remain **lead-gated**.

---

## What was broken (5 root bugs)

| # | Bug | Impact |
|---|-----|--------|
| 1 | `generate-predictions.yml` triggered on `workflow_run: ["Train Model"]` — **no such workflow exists** | predictions never auto-fired after a scrape; chain severed at the trigger |
| 2 | 3 model versions live at once: scrape committed v20, predictions/predict loaded **v15**, extension had v16/v19 | served model ≠ predicting model |
| 3 | `generate-predictions.yml` + `predict.yml` rebuilt a **v15-era inline feature dict** (~50 keys) | feeding a 135-feature model a 97-feature schema → silently wrong predictions |
| 4 | `generate-predictions.yml` read `sqlite3.connect('output/rentals.db')` — gitignored & untracked; scrape writes **Postgres** | ~0 predictions in CI |
| 5 | `daily_pipeline._run_train` called `rental_price_models_v14.py` — **does not exist** | local pipeline retrain was dead |

Plus a regression found & fixed during validation (#25): wall-clock `mark-inactive`
zeroed `is_active` on the frozen snapshot, and `--dry-run` mutated the DB.

---

## The loop (stages)

`automation/daily_pipeline.py` — each stage idempotent, logged to `logs/<run_id>/`, summarized to `summary.json`.

| Stage | What it does | Owner of the called tool |
|-------|--------------|--------------------------|
| preflight | kill stale procs, disk/integrity check, **backup**, cycle-relative mark-inactive | automation |
| scrape | all spiders (`cli.main scrape`) | scrapers (#3–5) |
| enrich | floorplan enrichment | scrapers |
| dedupe | cross-source sqft merge | dataeng/scrapers |
| **train** | `retrain_canonical.py --db output/rentals.db` → `rental_model_canonical.pkl` + smoke-test floor | modeler (#6) |
| **export** | orchestrates `v20.export_to_chrome(canonical)` → model.json/features.json + `export_similar_listings.py` → similar_listings.json | artifacts (#7) |
| **sync** | calls `scripts/sync_sqlite_to_postgres.py` **DRY-RUN** (mirror SQLite → Neon) | serving (#8) |
| **deploy** | trigger dashboard/extension refresh — **gated, no-op until lead confirms** | automation |
| report | negotiation report | — |
| postflight | final stats, cleanup old logs/backups | automation |

Ownership boundaries (lead-approved): the pipeline **orchestrates** the export generators
(#7) and **calls** the prod-sync script (#8) — it does **not** reimplement either.

### Idempotency
- `retrain_canonical.py` writes deterministic paths; re-running overwrites the same files.
- Export overwrites the served pair; verified by feature-count/order parity, not exit code.
- `sync_sqlite_to_postgres.py` is `TRUNCATE + reload` inside one transaction — re-running converges.

---

## CI chain (fixed)

```
daily-scrape.yml ("Daily Property Scrape", cron 06:00 UTC)
  scrape → Postgres
  pg_to_sqlite.py  (Postgres → local SQLite, so the SQLite trainer has data)
  retrain_canonical.py → rental_model_canonical.pkl (+features, +meta)
  export_to_chrome(canonical) → model.json / features.json
  commit canonical artifacts [skip ci]
        │ workflow_run: completed
        ▼
generate-predictions.yml ("Generate Predictions Cache")
  generate_predictions.py  (loads canonical pkl, reads POSTGRES_URL,
                            features via engineer_features_v20) → predictions.json
  commit predictions.json [skip ci]
```

`predict.yml` (on-demand, `repository_dispatch`) now calls `predict_one.py` → canonical model.

### New helper scripts (automation-owned)
| Script | Purpose |
|--------|---------|
| `scripts/generate_predictions.py` | predictions cache from canonical model + `engineer_features_v20`; reads Postgres (falls back to SQLite). Replaces the inline v15 dict. |
| `scripts/predict_one.py` | single-property prediction from canonical model; replaces `predict.yml`'s inline v15 dict. |
| `scripts/pg_to_sqlite.py` | materializes Postgres `listings` → SQLite so the SQLite-only trainer runs in CI after a Postgres scrape. |
| `scripts/_canonical_features.py` | version-agnostic resolver of the canonical feature pipeline from `retrain_canonical.py`; the foundation `canonical_predict.py` builds on. |

### Single source of truth: `canonical_predict.py` (modeler-owned)

Per the lead's "import, not reimplement" ruling, all prediction + export now route through the modeler's blessed module `canonical_predict.py` (`build_features` / `predict` / `predict_one` / `export_to_chrome` / `retrain`):
- `generate_predictions.py` → `cp.predict(df)`
- `predict_one.py` → `cp.predict_one(**kwargs)`
- pipeline EXPORT stage + `daily-scrape.yml` → `cp.export_to_chrome('chrome-extension/api')`

`canonical_predict.py` itself imports `scripts/_canonical_features.py`, so the whole stack stays version-agnostic (flip `retrain_canonical.py`'s `CANON_VERSION` and everyone follows). Its `export_to_chrome` serializes via `get_booster().save_model`, which **eliminates** the earlier xgboost-wrapper `save_model` non-zero-exit quirk — the export stage now exits clean with zero warnings. **No inline feature dict remains anywhere in automation's files.**

---

## Single local entrypoint

```bash
# Whole loop:
python -m cli.main daily                 # (alias of automation.daily_pipeline)
python -m automation.daily_pipeline

# Preview without writing anything (now truly non-mutating):
python -m cli.main daily --dry-run

# Run specific stages:
python -m cli.main daily --stage train --stage export --stage sync
```

---

## Validation (real runs, 2026-06-15)

| Check | Result |
|-------|--------|
| Full `--dry-run` (10 stages) | ✓ all wire in order; **is_active unchanged 7763→7763** (was being zeroed) |
| Real retrain (`--stage train`) | ✓ R²=0.8213, MAE=£1,533, 135 feat, 29s — matches MODEL_DECISION.md; passes floor (R²≥0.78 / MAE≤£1,800) |
| Real export (`--stage export`) | ✓ features.json (135) matches canonical order; model.json 11 MB; similar_listings.json 7,741 listings / 1.6 MB |
| Real sync DRY-RUN (`--stage sync`) | ✓ connected to Neon, **rolled back, no writes**; in-txn sim → **1,720 active prime-central comps queryable** (proves the prod-empty bug is fixed once executed) |
| Deploy stage | ✓ skipped (lead-gated) |
| `predict_one.py` | ✓ 2-bed/800sqft SW3 @£4,500 → fv £3,801, +18.4% "overpriced" |
| `generate_predictions.py` | ✓ 7,048 predictions, median £4,273/mo (sane); empty case writes `{}` gracefully |
| All 3 workflows | ✓ YAML valid; v15 refs & dead trigger gone |
| mark-inactive guard | ✓ 3 layers: frozen-snapshot guard fires on stale DB; cycle-relative cutoff on fresh data; >50% abort on any bulk flip (synthetic 91%-flip → aborted) |

---

## #25 fix — mark-inactive (regression found & fixed)

`_mark_inactive_listings()` previously used wall-clock `utcnow() - 7d`. On the frozen
canonical snapshot (max `last_seen` 2026-01-16) that zeroed **all** `is_active`, and it
ran even under `--dry-run` (preflight is exempt from dry-skip). Now hardened with THREE
independent guard layers (any one prevents the zeroing):
1. **Frozen-snapshot guard**: if the newest row is older than N days by wall clock, skip entirely (fires on the current snapshot → never touches the restored 7,763).
2. **Cycle-relative cutoff**: `max(last_seen) - N days`, not `utcnow() - N days` (matches DATA_LAYER_CONTRACT.md §5.1; fresh scrape unchanged, stale snapshot never wall-clock-zeroes).
3. **>50% abort** (dataeng's requested belt-and-braces): refuse the write if a single pass would flip >50% of currently-active rows inactive.
Plus `--dry-run` now writes nothing in preflight (counts only).

Verified (live + synthetic): live frozen snapshot stays 7,763 active even on a NON-dry
run; a synthetic 91%-flip scenario aborts (marked=0). Data restore to 7,763 active was
done by dataeng; these guards prevent recurrence.

---

## Still GATED (lead action required — NOT done by automation)

- **No live cron enabled / no prod commit** from these changes yet.
- **Prod sync execute** is gated: real load = `python3 scripts/sync_sqlite_to_postgres.py --execute --i-have-rotated-the-secret`, run by the lead **after POSTGRES_URL rotation**. The pipeline only ever dry-runs it.
- **Deploy stage** is `deploy_enabled=False` until confirmed.

## Open items handed off

- `rental_price_models_v20.py::export_to_chrome()` exits non-zero on a trailing
  `model.n_estimators` print (xgboost/sklearn quirk) **after** the JSON is written. Artifact
  is correct; the pipeline is robust to it. Source line should be fixed by **artifacts (#7)**.
- Prod Postgres currently holds 22,317 listings vs canonical 10,048 — the gated sync
  TRUNCATE+reload corrects this when the lead runs it.
