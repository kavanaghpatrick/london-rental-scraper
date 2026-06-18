# Research — unseen-postcode `np.log1p` crash (PROBLEM 3): root cause + final decision

Task #2 (research-model). Records the root cause and the **final shipped decision**, which
diverged from this research's initial parity recommendation. **Status: FIXED via neutral
median constant (`c375320`); CITY_CENTER rejected as out-of-distribution.** Artifacts
re-validated in `fa669cd`.

## Root cause (confirmed from the real CI log)

CI failure `tests/test_model_inference.py::test_unseen_postcode_uses_default_freq` (run
27636202638, push 2bbbf12) was **NOT** the postcode-freq `default:null` theory in the
context doc. The real CI traceback:
```
rental_price_models_v20.py:783: df['log_center_distance'] = np.log1p(df['center_distance_km'])
center_distance_km dtype: object, value NaN
AttributeError: 'float' object has no attribute 'log1p'
```
Chain (single-row, unseen district `ZZ9`, `latitude=0`): `get_coords` finds no
`POSTCODE_CENTROIDS['ZZ9']` → `[0,0]` → `get_distance_to_center(0,0)` hits its
`if lat==0: return None` guard → `center_distance_km=None` → **object-dtype** Series →
`median()` of a single all-None column = `NaN` → `fillna(NaN)` is a no-op → `np.log1p`
over an object array crashes. `tube_distance_km` had the identical latent bug.

**Why CI failed but local passed:** pandas-version downcast. Local pandas 2.3.3 silently
downcasts object→float64 on `.fillna()` (deprecation-warned), so `log1p(float64 NaN)`=NaN,
no error. The CI runner's pandas keeps object dtype → crash. Deterministic local repro:
`pd.set_option('future.no_silent_downcasting', True)` then call `engineer_features_v20`
on a coordless row. (This option is now set in `tests/conftest.py` so local == CI.)

**Fix locus:** `rental_price_models_v20.engineer_features_v20` distance block (NOT the
freq machinery — `gen_inference_stats.py` / `canonical_predict.py` freq override / the
committed `inference.json` were all already correct; that theory is disproven by: no
committed `inference.json` in any commit ever carried `default:null`, the reader is
float-safe, and a missing artifact yields `1.0` not `None`).

## FINAL DECISION (supersedes this research's parity recommendation)

> **This research recommended setting `lat_filled/lon_filled = CITY_CENTER` (→
> `center_distance_km = 0`) for coordless rows, to mirror the JS (`xgboost.js:897`).
> The lead + architect chose NEUTRAL FROZEN CONSTANTS instead, and that is what
> shipped. Their choice is the better one** — recorded here so the decision is honest.

**Deciding evidence (architect/model-dev):** training `center_distance_km` has **min
0.416 km and ZERO rows at 0**, so CITY_CENTER's `dist=0` is **out-of-distribution**.
Scoring a coordless `ZZ9` row with `center_distance_inv = 1/(1+0) = 1.0` (maximum
centrality) put an **upward bias on exactly the least-informative rows** → £5,102. The
neutral training-median constant gives a sane £3,938.

**What shipped** (`rental_price_models_v20.py`):
```python
DEFAULT_CENTER_DISTANCE_KM = 3.3892584524370477   # training median (post centroid-fallback)
DEFAULT_TUBE_DISTANCE_KM   = 0.6075240563353417
...
df['tube_distance_km']   = df['tube_distance_km'].fillna(DEFAULT_TUBE_DISTANCE_KM)
df['center_distance_km'] = df['center_distance_km'].fillna(DEFAULT_CENTER_DISTANCE_KM)
```
These constants are the canonical training medians (N=8716, computed by research-model on
request; full float64 precision is retained so the JS literals in `xgboost.js` are
byte-exact for parity). `fillna(DEFAULT_*)` is **unconditional** (always the constant, NOT
`median-if-notna-else-constant`) for three reasons: (1) pandas-version-proof — no object
dtype survives, no `median()`-of-empty edge case; (2) train/inference consistency — on a
full training frame the frame-median equals the constant anyway; (3) **batch robustness**
(model-dev's deciding rationale) — a coordless row inside a MULTI-row inference batch would
otherwise be filled with *that batch's* frame-median rather than the fixed constant,
introducing a batch-vs-single skew. Always-constant removes that skew. TRAIN behaviour is
unchanged (real rows always have computed distances; the constant only materializes for
genuinely coordless rows).

**Parity:** `chrome-extension/xgboost.js` was changed to mirror the SAME constants (NOT
CITY_CENTER), and a coordless golden sample (`coordless_unseen_postcode`) was added to
`gen_feature_parity_golden.py`. `_fixture_diff.mjs` is 0/0 across 9 samples; vendored copy
re-deployed.

## Verification (re-checked at HEAD by research-model)

- Coordless `ZZ9` under `future.no_silent_downcasting=True`: `center_distance_km=3.3893`
  (float64), `log_center_distance=1.4792` — **no crash**.
- `tests/test_model_inference.py` 24/24; full suite 180 passed; `_fixture_diff.mjs` 0/0.

## Tests that lock this in
- `test_coordless_row_distance_is_numeric_no_log1p_crash` — version-independent RED (float
  dtype + finite distances on a coordless row), green post-fix.
- `tests/conftest.py` sets `future.no_silent_downcasting=True` (local == CI).
- The coordless golden sample → `_fixture_diff.mjs` enforces JS↔Python on this path.
- Social-housing target-leak fix untouched (distance block only).
