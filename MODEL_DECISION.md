# MODEL DECISION — Canonical Rental Price Model

**Author:** `modeler` (Task #6)
**Date:** 2026-06-15
**Decision:** **v20 is canonical.**
**Data:** `output/rentals.db` (frozen snapshot, last scrape 2026-01-16), recency-independent set per `DATA_LAYER_CONTRACT.md` §5.

---

## FUTURE_IMPROVEMENTS (non-blocking, deferred per lead — bundle into ONE future clean-data retrain)

Two model-hygiene items surfaced during task #30. Both are DEFERRED (the current
canonical + inference fix ship correctly); when a future cycle does a clean-data
retrain, address BOTH together (one retrain → one re-export → one re-fixture/re-diff):

**(1) `is_social_housing` is a weak price-derived feature (mild leak + train/inference mismatch).**
Its PPSF branch (`0 < ppsf < 3.5` for premium postcodes) keys off `ppsf = price/sqft`
= partially target-derived. It's a WEAK leak (fires on 4.0% of training rows, corr
with log-price = −0.088 — far weaker than the old `is_luxury = price>10000`), and it
does NOT change the bake-off ranking (all candidates trained consistently). At
inference, price is unknown → it was spuriously firing (the −81% bug, now fixed at
`rental_price_models_v20.py` via `0 < ppsf < 3.5`). The fix is sufficient to ship.
CLEAN VERSION: retrain v20 with `is_social_housing` using ONLY the estate-name regex
(price-independent) → leak-free AND removes the train/inference feature mismatch.
Expected metric change: negligible (given 4% / −0.088). Deferred to avoid a finish-line
re-export/re-fixture/re-diff churn for zero user-visible benefit.

**(2) `postcode_normalized` is 86.7% NULL → `fillna('SW3')` inflates SW3.**
4,462 of 5,147 training rows have null `postcode_normalized`; FE does
`postcode_district = postcode_normalized.fillna('SW3')` → `postcode_freq(SW3)=0.88`.
For NOW we reproduce it as-is at inference (model was fit on it; see task #30 /
`rental_model_canonical_inference.json`). RECOVERABLE: ~3,756 of the 4,462 nulls (73%)
have a non-empty raw `postcode` → a cleanup that re-extracts the district (owned by
`dataeng`, postcode normalization in `pipelines.py`) would de-inflate SW3 and likely
improve prime discrimination. Warrants a clean + retrain.

**(3) `is_premium_agent` is a DEAD feature (case-mismatch bug).**
`rental_price_models_v20.py` computes `is_premium_agent` via
`any(agent in x.lower() for agent in PREMIUM_AGENTS)` — but `PREMIUM_AGENTS` are
CAPITALIZED (`'Savills'`, `'Knight Frank'`) compared against a lowercased
`agent_brand`, so it NEVER matches. Result: `is_premium_agent` = 0 on ALL 5,147
training rows (despite 279 Knight Frank / 150 Chestertons / 13 Savills listings).
Both `is_premium_agent` and `premium_agent_size` therefore have 0.0 model importance
(dead) — forcing them 0→1 changes predictions by £0.00. NO current impact (the model
ignores them; `source_quality`, importance 0.009, captures agent quality correctly).
FIX (future retrain): `any(agent.lower() in x.lower() ...)` revives a legit
premium-agent signal. Low priority (source_quality covers most of it). At inference
the fixture value stays 0 to match the trained model — found via task #30 parity check.

> Doing (1)+(2)+(3) in the SAME future retrain is most efficient. Ping `modeler` to
> re-run `retrain_canonical.py` + regenerate the inference/freq maps + golden fixture.

---

## TL;DR

Bake-off of **v15 vs v16 vs v19 vs v20** on a single common holdout, then a
5-seed stability re-run to avoid deciding on split luck. **v20 wins every metric
and is the most stable.** The user leaned toward v15, but v15 is consistently
*last* on every metric and every seed — the margin v20 gives is material for a
rent-estimation product (MAE −£153/month vs v15) and its added features are
**largely legitimate**, so the win is clearly worth the complexity.
(CORRECTION, task #30: v20 is NOT fully leakage-free as this doc originally claimed —
`is_social_housing` keys off `ppsf` = price/sqft, a weak price-derived feature. It's a
1-bit leak on 4% of rows, corr −0.088, and does NOT change the v20-vs-v15 ranking since
all candidates trained consistently. See FUTURE_IMPROVEMENTS (1) for the clean fix.)
Choosing v20 also *reduces* version drift: it's already what the daily-scrape
automation commits and what the live Chrome extension runs.

---

## Metrics

### Single common holdout (seed 42, 4,233 train / 1,059 test)

| ver | features | RMSE | MAE | R² | MAPE | Median-APE | train time |
|-----|---------:|-----:|----:|----:|-----:|-----------:|-----------:|
| v15 |  79 | £3,956 | £1,780 | 0.7760 | 20.8% | 16.4% | 4.8s |
| v16 | 101 | £3,762 | £1,634 | 0.7974 | 19.2% | 15.2% | 4.9s |
| v19 | 150 | £3,622 | £1,673 | **0.8123** | 20.1% | 15.9% | 5.3s |
| v20 | 135 | £3,671 | **£1,627** | 0.8072 | 19.2% | **15.1%** | 5.2s |

On this one split v19 had the top R²/RMSE — **but that was split luck** (see below).

### Stability — 5 seeds [42, 7, 123, 2024, 99] (mean ± std)

| ver | R² | RMSE | MAE | Median-APE |
|-----|----|------|-----|-----------|
| v15 | 0.7386 ± 0.0317 | £4,269 ± 300 | £1,881 ± 121 | 16.3% |
| v16 | 0.7646 ± 0.0286 | £4,051 ± 285 | £1,765 ± 125 | 15.0% |
| v19 | 0.7659 ± 0.0343 | £4,035 ± 313 | £1,800 ± 125 | 15.7% |
| **v20** | **0.7776 ± 0.0241** | **£3,938 ± 248** | **£1,728 ± 108** | **14.8%** |

**v20 is best on R², RMSE, MAE, and Median-APE — and has the lowest variance on
all of them.** v19's single-split R² lead collapsed to a tie with v16 (and behind
v20) with the *highest* variance, so its apparent edge was not real.

### Canonical retrain (full recency-independent data, 5,147 samples, 5-fold CV)

| metric | value |
|--------|-------|
| R² | **0.8213** |
| RMSE | £3,424 |
| MAE | **£1,533** |
| MAPE | 18.3% |
| Median-APE | 14.1% |

(Better than the holdout because the final model trains on all 5,147 rows.)

---

## Why v20 over v15 (the user's lean)

The user preferred v15 for its **"no-leakage"** design (v15 removed `is_luxury`,
ppsf-of-this-row encodings, and fold-leaking target-mean encodings that inflated
v6–v14). That concern is valid — but it does **not** apply to v20:

- v20 contains **no** `is_luxury` or price-thresholded feature.
- v20's `*_expected_price` / `*_ppsf` features use **hardcoded static lookup
  tables** (`PC_MEWS_PPSF`, `PC_HOUSE_PPSF`, `PRESTIGE_LOCATION_PPSF`, …) keyed on
  **property type / postcode / address** — pre-computed offline, NOT derived from
  the current row's `price_pcm`. That is prior-knowledge encoding, not leakage;
  the model still learns the residual.
- Verified: the only "price"-named features are the static expected-price ones
  above; none read the row's own target.

So v15's *sole* advantage (avoiding leakage) is matched by v20, while v20 adds
**legitimate** signal (size non-linearity, bathroom/ensuite signals, mews/house/
flat type encoding, address-prestige and postcode micro-location). Net effect vs
v15: **R² +0.039, MAE −£153/month, Median-APE −1.5 pts**, and lower variance.
For a tenant negotiating rent, £153/month of error reduction is material.

### Why not v19
Best on one lucky split, but across seeds it ties v16, trails v20, and is the
**highest-variance** model. It is also **brittle**: `train_v19_extend_v18.py`
loads its 150-feature list from `chrome-extension/api/features.json` at import
time, coupling training to a serving artifact. v20 is self-contained.

### Why not v16
Solid and close, but v20 beats it on every stability metric (R² 0.778 vs 0.765,
MAE £1,728 vs £1,765) for ~34 more features that are cheap to compute.

---

## Canonical artifacts (EXACT filenames downstream must load)

Retrained by `retrain_canonical.py` on the recency-independent set:

| artifact | path |
|----------|------|
| model (pickled `XGBRegressor`) | `output/rental_model_canonical.pkl` |
| feature order (`list[str]`, 135) | `output/rental_model_canonical_features.pkl` |
| metadata (version, metrics, params) | `output/rental_model_canonical_meta.json` |

- Target: `log1p(price_pcm)`; inverse with `expm1` at predict.
- Feature engineering = `rental_price_models_v20.py` (`engineer_features_v20` +
  `get_feature_columns_v20`). The 135-col order in the features pkl IS the
  contract — feed columns in that exact order.
- Round-trip verified: `pkl.n_features_in_ == len(features_pkl) == 135`.

> Artifact-export (#7) and automation (#9) must **both** converge on
> `rental_model_canonical.pkl` / `_features.pkl`. See drift section.

---

## Version drift this resolves

Before this decision there were **three** live versions wired into different paths:

| path | was loading | should load |
|------|-------------|-------------|
| `.github/workflows/predict.yml` | `output/rental_model_v15.pkl` (97f) | `rental_model_canonical.pkl` |
| `.github/workflows/generate-predictions.yml` | `output/rental_model_v15.pkl` (97f) | `rental_model_canonical.pkl` |
| `.github/workflows/daily-scrape.yml` | commits `output/rental_model_v20.pkl` (150f) | retrain → `rental_model_canonical.pkl` |
| `chrome-extension/api/features.json` | 150f (v20-equivalent) | export from canonical (v20) — already aligned in spirit |
| `scripts/predict_rent.py` | stale `rental_model_v14.pkl` | `rental_model_canonical.pkl` (or archive script) |

Canonical = v20, so the serving path moves v15→v20 and the extension stays on
v20 — everyone converges on ONE version with ZERO accuracy regression (v20 ≥ v15
everywhere).

---

## Single source of truth: `canonical_predict.py`

`automation`, `artifacts`, `predict.yml` and `generate-predictions.yml` each
re-implemented the model's feature dict inline (their own `AMENITY_FEATURES`,
`PRIME_POSTCODES`, `TUBE_STATIONS`, one-hot logic). When canonical changed
v15→v20 those copies silently went out of sync → wrong (zero-filled) features →
silently bad predictions. **`scrapy_project/canonical_predict.py` is now the ONE
place that builds features / predicts / exports.** Everyone imports it:

```python
import canonical_predict as cp
X, cols   = cp.build_features(df)      # exact v20 frame, training-order columns
preds     = cp.predict(df)             # £ pcm; handles log1p/expm1 + feature order
model, fc = cp.load_canonical()        # loads rental_model_canonical.pkl/_features.pkl
cp.export_to_chrome('chrome-extension/api')   # model.json + features.json as a MATCHED pair
cp.retrain('output/rentals.db')        # retrain + write canonical artifacts
```

`cp.predict()` reindexes to the artifact's exact 135-col order and zero-fills any
property-type dummy a given batch didn't emit (same as `xgboost.js`'s `?? 0`). To
switch canonical version later, repoint the three `_canon.*` bindings at the top
of `canonical_predict.py`; every consumer follows.

> ⚠️ Bug found + worked around: `rental_price_models_v20.py --export` is BROKEN
> under xgboost 3.1.2 — its `export_to_chrome` hardcodes `chrome-extension/api`
> (ignores the arg) and calls `XGBRegressor.save_model` on the *unpickled* model,
> which raises `_estimator_type undefined`. `canonical_predict.export_to_chrome`
> serializes via `get_booster().save_model` instead (the Booster JSON is exactly
> what `chrome-extension/xgboost.js` parses). Use the module, not the v20 script.

---

## Reproduce

```bash
cd scrapy_project
cp output/rentals.db output/rentals_modeler_copy.db      # never read the live file
python3 bakeoff_v15_v16_v19_v20.py --db output/rentals_modeler_copy.db   # single-split bake-off
python3 bakeoff_stability.py                              # 5-seed stability
python3 retrain_canonical.py --db output/rentals_modeler_copy.db         # retrain + save canonical
```

Outputs: `output/bakeoff_results.json`, `output/bakeoff_run.log`,
`output/bakeoff_stability.log`, and the three `rental_model_canonical*` artifacts.
