# ARTIFACT EXPORT REPORT — Chrome Extension (Task #7)

**Author:** `artifacts` (Task #7)
**Date:** 2026-06-15
**Canonical model:** v20 (per `MODEL_DECISION.md`, modeler #6)
**Status:** Artifacts regenerated + verified, staged + committed LOCALLY. **Push to GitHub `main` is gated on team-lead review** (outward-facing — makes the live `raw.githubusercontent.com` URLs resolve).

---

## TL;DR

`chrome-extension/api/model.json` + `features.json` now ARE the canonical v20
model, byte-for-byte (exported directly from `output/rental_model_canonical.pkl`,
**no retrain**). Verified by re-implementing the `xgboost.js` tree-traversal
arithmetic and matching the pkl's own predictions to **3e-5** (log space) over 200
random vectors. `similar_listings.json` (the live 404) is regenerated with 7,741
active listings. The four superseded `model_v16/v19` + `features_v16/v19` files are
handed to janitor for archive. Zero drift: extension, serving (#8/#9), and
automation all converge on v20.

---

## What changed in `chrome-extension/api/`

| file | before | after | how |
|------|--------|-------|-----|
| `model.json` | 150-feat v20-equiv (retrained, drifted; live GitHub copy was a different build) | **135-feat canonical v20**, 1500 trees, base_score `[8.364275E0]` | `booster.save_model()` from `rental_model_canonical.pkl` |
| `features.json` | 150 names (local) / 153 (live GitHub — DRIFTED) | **135 names**, exact canonical training order | `json.dump(canonical_features_pkl)` |
| `similar_listings.json` | **MISSING → 404 live** | **7,741 records**, 1.6 MB | `scripts/export_similar_listings.py` from `output/rentals.db` |
| `predictions.json` | present, stale | **unchanged** (see note) | n/a |

New helper committed: `scripts/export_canonical_to_extension.py` (the drift-free
exporter; reusable by automation #9).

---

## Why direct-pkl-export, NOT `rental_price_models_v20.py --export`

`--export` RE-TRAINS a fresh booster inside `main()` via `load_and_clean_data()`,
which filters `is_active = 1` (and reads Postgres if `POSTGRES_URL` is set). That is
a **different row set** than the canonical recency-independent retrain (5,147 rows,
no `is_active`). Same hyperparameters (`build_xgboost()` params verified identical to
canonical `meta.json`), but different data ⇒ a **different booster ⇒ drift** — exactly
what this task kills. Exporting the canonical pkl's booster directly guarantees
`model.json` IS the served/automation model.

The 135 vs 150 feature gap is a **non-issue** for the extension: `xgboost.js`
`buildFeatures()` can emit 151 distinct keys and the loader builds the vector by
`features.map(name => featureDict[name] ?? 0)` — it **zero-fills** any name the model
doesn't use. Confirmed: **all 135 canonical features ARE produced by
`buildFeatures()`** (0 missing). The extra 15 columns in the 150-export were unused
one-hot dummies.

---

## Verification (real, not assumed)

1. **Schema**: `model.json` has `learner.gradient_booster.model.trees[]` with
   `left_children`, `right_children`, `split_indices`, `split_conditions`,
   `default_left`, and `base_score` in the `[8.364275E0]` string-array form
   `xgboost.js` already parses. Max `split_index` used = 133 < 135 features. ✔
2. **Order contract**: `model.json` embedded `feature_names` == `features.json` ==
   `rental_model_canonical_features.pkl`, identical content AND order. ✔
3. **Numeric round-trip**: re-implemented the `xgboost.js` `predict()` /
   `predictTree()` path in Python against the ON-DISK `model.json`+`features.json`
   and compared to `rental_model_canonical.pkl.predict()` on 200 random vectors:
   **max |Δ log| = 2.98e-05**. A realistic 2-bed/900 sqft case → `expm1` ≈ **£3,486
   pcm**. ✔
4. **No dangling refs**: no JS references `model_v16/v19` or `features_v16/v19`;
   `manifest.json` `web_accessible_resources` already lists only
   `api/model.json` + `api/features.json`. ✔

---

## The 404 fix (`similar_listings.json`)

Root cause: `content.js` `SIMILAR_URL` points at
`raw.githubusercontent.com/.../chrome-extension/api/similar_listings.json`, but the
file was **never committed** (`api/.gitignore` only ignores `*.pkl`) → 404 live.
Per lead ruling, fixed by regenerating + committing the **static JSON** (reliable
path; the Vercel `/api/similar` comps route is separately repaired by serving #8 but
is not the dependable fallback).

Data note (pre-existing, not a regression): 14.1% of records have an empty postcode
district `p` (the source rows have NULL/empty `postcode` in the DB), and 31.4% have
`s`=0 (no sqft). The comps feature still works for the ~86% with a district; lat/lon
present on 79%. Backfilling postcodes is a data-layer concern (#2), out of scope here.

`predictions.json` left unchanged: the `getCachedPrediction` cache that consumes it
is **commented out** in `content.js` (live prediction always runs), so it has no
runtime effect. Regenerating it would need a batch-prediction pipeline that isn't
wired up. Flagged for automation #9 if the cache is ever re-enabled.

---

## Handoff to janitor — files to ARCHIVE (do NOT delete from history)

All git-tracked, unreferenced by any JS, superseded by canonical v20:

```
chrome-extension/api/model_v16.json      (1,999,359 bytes)
chrome-extension/api/features_v16.json   (2,037 bytes)
chrome-extension/api/model_v19.json      (12,108,969 bytes)
chrome-extension/api/features_v19.json   (2,061 bytes)
```

---

## Push gate

Staged + committed LOCALLY only. **team-lead pushes to GitHub `main`** after review —
that push is what makes the extension's `raw.githubusercontent.com` artifact URLs
resolve to the new files (outward-facing).
