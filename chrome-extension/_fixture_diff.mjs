// Key-by-key drift guard: JS XGBFeatures.buildFeatures vs the modeler's golden
// fixture (Python engineer_features_v20 = ground truth). Run:
//   node chrome-extension/_fixture_diff.mjs [path/to/feature_parity_golden.json]
//
// Fixture schema (output/feature_parity_golden.json):
//   { canonical_version, n_features, feature_order:[...135], required_input_fields,
//     samples: [ { label, inputs:{raw fields}, prediction_pcm, features:{name:value} } ] }
//
// Exits non-zero if ANY feature in ANY sample differs beyond EPS, OR (if the model
// is present) the JS £ estimate differs from prediction_pcm beyond TOL. This is the
// CI drift guard for task #28 (architecture B): JS must stay byte-equal to Python.

import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { XGBoostPredictor, XGBFeatures } = require('./xgboost.js');

const EPS = 1e-6;      // per-feature tolerance
const TOL = 0.005;     // 0.5% £ tolerance
const here = fileURLToPath(new URL('.', import.meta.url));

const fixturePath = process.argv[2] || `${here}../output/feature_parity_golden.json`;
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
const samples = fixture.samples || fixture;

// Optional £ check: load the canonical model if present
let predictor = null;
try {
  const model = JSON.parse(fs.readFileSync(`${here}api/model.json`, 'utf8'));
  const features = JSON.parse(fs.readFileSync(`${here}api/features.json`, 'utf8'));
  predictor = new XGBoostPredictor();
  predictor.model = model; predictor.features = features; predictor.loaded = true;
} catch { /* £ check skipped if model absent */ }

let totalMismatch = 0;
let poundFails = 0;
for (const s of samples) {
  const label = s.label || JSON.stringify(s.inputs).slice(0, 40);
  const jsFeat = XGBFeatures.buildFeatures(s.inputs);
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
  if (predictor && s.prediction_pcm != null) {
    const gbp = Math.expm1(predictor.predict(jsFeat));
    const rel = Math.abs(gbp - s.prediction_pcm) / s.prediction_pcm;
    const ok = rel <= TOL;
    if (!ok) poundFails++;
    console.log(`   £: JS=£${Math.round(gbp)} expected=£${Math.round(s.prediction_pcm)} (${(rel * 100).toFixed(2)}%) [${ok ? 'OK' : 'OFF'}]`);
  }
}
const pass = totalMismatch === 0 && poundFails === 0;
console.log(`\n=== ${pass ? 'PASS' : 'FAIL'}: ${totalMismatch} feature mismatch(es), ${poundFails} £ mismatch(es) across ${samples.length} samples ===`);
process.exit(pass ? 0 : 1);
