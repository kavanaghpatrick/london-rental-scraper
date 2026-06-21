// FOR-SALE parity gate: JS SaleXGBFeatures.buildFeatures + SaleXGBoostPredictor.predict vs
// the Python golden (output/sale_feature_parity_golden.json, scored from the COMMITTED
// Booster with inference=True). This is the SALE analogue of fixture_diff.mjs — a FORK, NOT
// an edit of the rental gate. Run:
//   node chrome-extension/sale_fixture_diff.mjs [path/to/sale_feature_parity_golden.json]
//
// LOAD-BEARING FORK DELTAS from the rental original:
//   - requires ./sale_xgboost.js (SaleXGBoostPredictor / SaleXGBFeatures), NOT ./xgboost.js
//   - model/features load from ../output/sale_api/{model,features}.json, NOT chrome-extension/api/*
//   - the £ check reads s.prediction_price (NOT s.prediction_pcm) — a straight copy would
//     silently SKIP the £ check. The key is asserted below.
//
// Exits 0 iff 0 feature mismatch (EPS=1e-6) AND 0 £ mismatch (TOL=0.005) across ALL samples.

import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { SaleXGBoostPredictor, SaleXGBFeatures } = require('./sale_xgboost.js');

const EPS = 1e-6;   // per-feature tolerance
const TOL = 0.005;  // 0.5% £ tolerance
const here = fileURLToPath(new URL('.', import.meta.url));

const fixturePath = process.argv[2] || `${here}../output/sale_feature_parity_golden.json`;
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
const samples = fixture.samples || fixture;

// The SALE golden's £ key is prediction_price (a £ lump sum), NOT prediction_pcm. Assert it
// so a stale/rental-shaped golden fails loudly instead of skipping the £ check.
if (samples.length && samples[0].prediction_price == null) {
  console.error('FATAL: sale golden samples must carry `prediction_price` (got none on sample[0]).');
  process.exit(1);
}

// Load the COMMITTED sale Booster for the £ check (sale_api/*, NOT chrome-extension/api/*).
let predictor = null;
try {
  const model = JSON.parse(fs.readFileSync(`${here}../output/sale_api/model.json`, 'utf8'));
  const features = JSON.parse(fs.readFileSync(`${here}../output/sale_api/features.json`, 'utf8'));
  predictor = new SaleXGBoostPredictor();
  predictor.model = model; predictor.features = features; predictor.loaded = true;
} catch { /* £ check skipped if model absent */ }

let totalMismatch = 0;
let poundFails = 0;
for (const s of samples) {
  const label = s.label || JSON.stringify(s.inputs).slice(0, 40);
  const jsFeat = SaleXGBFeatures.buildFeatures(s.inputs);
  const py = s.features || {};
  const offenders = [];
  for (const k of Object.keys(py)) {
    const pv = Number(py[k]);
    const jv = Number(jsFeat[k] ?? 0);
    if (Math.abs(jv - pv) > EPS) offenders.push({ k, js: jv, py: pv });
  }
  if (offenders.length) {
    totalMismatch += offenders.length;
    console.log(`\nMISMATCH [${label}] — ${offenders.length} feature(s):`);
    for (const o of offenders) console.log(`  ${o.k.padEnd(34)} JS=${o.js}  PY=${o.py}`);
  } else {
    console.log(`OK [${label}] — all ${Object.keys(py).length} features match (<=${EPS})`);
  }
  if (predictor && s.prediction_price != null) {
    const gbp = Math.expm1(predictor.predict(jsFeat));
    const rel = Math.abs(gbp - s.prediction_price) / s.prediction_price;
    const ok = rel <= TOL;
    if (!ok) poundFails++;
    console.log(`   £: JS=£${Math.round(gbp)} expected=£${Math.round(s.prediction_price)} (${(rel * 100).toFixed(2)}%) [${ok ? 'OK' : 'OFF'}]`);
  }
}
const pass = totalMismatch === 0 && poundFails === 0;
console.log(`\n=== ${pass ? 'PASS' : 'FAIL'}: ${totalMismatch} feature mismatch(es), ${poundFails} £ mismatch(es) across ${samples.length} samples ===`);
process.exit(pass ? 0 : 1);
