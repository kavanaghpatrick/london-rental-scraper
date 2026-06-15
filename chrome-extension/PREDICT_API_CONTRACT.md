# /api/predict Request Contract + Shared Predictor (Task #28)

**Owner:** artifacts · **Consumers:** chrome-extension (content.js) + serving (`dashboard/src/app/api/predict/route.ts`)
**Model:** canonical v20 (135 features). **Status: GREEN** — JS predictor proven byte-equal to Python.

> **ARCHITECTURE (lead DEFINITIVE ruling, 2026-06-15): B — pure-Node shared JS.**
> `/api/predict` runs the SHARED JS module `chrome-extension/xgboost.js` (NOT Python).
> The JS feature engineering was made byte-equal to Python v20 `engineer_features_v20`
> and validated key-by-key + £ against the modeler's golden fixture
> (`output/feature_parity_golden.json`): **0 feature mismatch + 0 £ mismatch across all
> 7 samples**. A CI drift guard (`_fixture_diff.mjs`) prevents future drift.
> content.js POSTs raw fields to `/api/predict` as PRIMARY; the in-browser JS path is the
> FALLBACK (now also canonical-accurate — uses the same module + inputs).

---

## Shared module (`chrome-extension/xgboost.js`)

One source for both client and server — no client/server drift.

- **Pure / Node-safe**: no `window`/`document`/`location` inside `buildFeatures`/`predict`.
- **Server (Node/CommonJS):** `const { XGBoostPredictor, XGBFeatures } = require('<path>/xgboost.js')`.
- **Extension (browser global):** `window.XGBFeatures` / `window.XGBoostPredictor` (set when `window` exists).
- **Model loading:** `predictor.load(MODEL_URL, FEATURES_URL)` fetches model.json/features.json
  from the GitHub-raw URL (one source) and caches the parse at module scope
  (`globalThis.__XGB_MODEL_CACHE__`) for warm serverless reuse.
- **Predict:** `const f = XGBFeatures.buildFeatures(raw); const pcm = Math.round(Math.expm1(predictor.predict(f)));`

---

## Request contract (POST /api/predict)

Body = RAW property fields (`feature_parity_golden.json.required_input_fields`). Pass
straight to `buildFeatures(body)` — do NOT hand-build features. **Load-bearing for parity:**
`property_type_std`, `source`, `agent_brand`, and the explicit floor flags.

```jsonc
{
  "bedrooms": 2, "bathrooms": 2, "size_sqft": 1000,
  "postcode": "SW3 4AA", "postcode_normalized": "SW3", "area": "Chelsea",
  "property_type": "flat", "property_type_std": "flat",     // _std drives type one-hot/is_*
  "address": "12 Some Street",
  "latitude": 51.4934, "longitude": -0.1610,
  "source": "rightmove",            // rightmove|knightfrank|chestertons|savills|foxtons -> source_quality
  "agent_brand": "Knight Frank",    // premium-agent detection
  "let_type": "long",
  "features": "", "description": "",
  "floor_count": 0,                 // explicit floors (else 0 = training default)
  "has_basement": 0, "has_ground": 0, "has_first_floor": 0,
  "has_second_floor": 0, "has_third_floor": 0, "has_fourth_plus": 0, "has_roof_terrace": 0,
  "pageUrl": ""                     // optional; source_quality falls back to it if source absent
}
```

## Response (proposed; serving owns final shape)

```jsonc
{ "estimate_pcm": 4799, "model_version": "v20", "currency": "GBP",
  "range_low": 3791, "range_high": 5807 }   // estimate * 0.79 / 1.21
```

content.js reads `estimate_pcm` (+ optional `range_low/high`); on any non-OK /
no-estimate / 6s timeout it falls back to the in-browser model (footer "~estimate (offline)").

## Validation (golden fixture = oracle)

`node chrome-extension/_fixture_diff.mjs` → must exit 0 (PASS). Current: **0 feature + 0 £
mismatch across 7 samples**: belgravia £11,189 · chelsea £4,862 · SANITY_sw3 **£4,799** ·
SANITY_4_south_eaton **£11,886** · mews £6,758 · studio £1,947 · huge £27,635. Serving's
route should hit these same £ values. This harness is the CI drift guard (automation wires it).

## Parity fixes applied to xgboost.js (for the record)

freq maps → v20 training maps (`rental_model_canonical_inference.json`); explicit floor
flags consumed; floor_count default 0; `property_type_std` drives type features;
`source_quality` from `source` (.fillna 2); `has_furnished` not from free text;
`is_ultra_luxury_address` uses `ULTRA_PRESTIGE_STREETS`; **predictTree compares in float32
(`Math.fround`)** to match XGBoost; **`parseAmenities('')` returns zeroed set** (was `{}` →
NaN `premium_amenity_count`).
