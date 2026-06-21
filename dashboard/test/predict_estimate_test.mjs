/**
 * REAL behavioral test for /api/predict's estimate path — NO Postgres, NO network.
 *
 * WHAT IT GUARDS:
 *   /api/predict runs the certified shared JS predictor INLINE (ruling B): it calls
 *   XGBFeatures.buildFeatures(raw) -> XGBoostPredictor.predict() -> expm1 -> round,
 *   then a x0.79/x1.21 range. This harness exercises that EXACT pipeline against the
 *   REAL committed canonical v20 model (chrome-extension/api/model.json +
 *   features.json) and asserts a SANE estimate_pcm for a known prime-central property,
 *   plus the response-envelope invariants the route returns.
 *
 * HOW IT AVOIDS THE NETWORK (the route fetches model.json from GitHub-raw):
 *   The predictor caches parsed {model, features} in globalThis.__XGB_MODEL_CACHE__
 *   keyed by the (modelUrl|featuresUrl) pair. We pre-seed that cache from the committed
 *   model files, then call load() with matching sentinel URLs -> it hits the cache and
 *   NEVER fetches. So CI needs only node + the committed artifacts (no GitHub-raw call,
 *   no flakiness). This is the same model the route serves post-go-live.
 *
 * It also asserts the route file actually WIRES this pipeline (the vendored predictor
 * is byte-identical to the source, and the route computes estimate_pcm via expm1 +
 * the 0.79/1.21 range), so the tested pipeline can't drift from the production route.
 *
 * Run: node dashboard/test/predict_estimate_test.mjs   (exit 0 = pass, 1 = fail)
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..');
const ROUTE_DIR = join(__dirname, '..', 'src', 'app', 'api', 'predict');
const API_DIR = join(ROOT, 'chrome-extension', 'api');

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) {
    console.log(`OK   ${name}`);
  } else {
    failures++;
    console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`);
  }
}

// --------------------------------------------------------------------------- //
// Load the predictor module the ROUTE uses (the vendored byte-identical copy).
// --------------------------------------------------------------------------- //
const VENDORED = join(ROUTE_DIR, 'xgboost.predictor.js');
const { XGBoostPredictor, XGBFeatures } = require(VENDORED);

// --------------------------------------------------------------------------- //
// Pre-seed the model cache from the committed v20 artifacts so load() is offline.
// --------------------------------------------------------------------------- //
const MODEL_URL = 'fixture://model.json';
const FEATURES_URL = 'fixture://features.json';

function seedCache() {
  const model = JSON.parse(readFileSync(join(API_DIR, 'model.json'), 'utf8'));
  const features = JSON.parse(readFileSync(join(API_DIR, 'features.json'), 'utf8'));
  const cache =
    globalThis.__XGB_MODEL_CACHE__ || (globalThis.__XGB_MODEL_CACHE__ = new Map());
  cache.set(`${MODEL_URL}|${FEATURES_URL}`, { model, features });
  return features.length;
}

// Mirror the route's predictFairValue() envelope math EXACTLY.
function predictFairValue(predictor, raw) {
  const featureDict = XGBFeatures.buildFeatures(raw);
  const predLog = predictor.predict(featureDict);
  const estimate = Math.round(Math.expm1(predLog));
  return {
    estimate_pcm: estimate,
    model_version: 'v20',
    currency: 'GBP',
    range_low: Math.round(estimate * 0.79),
    range_high: Math.round(estimate * 1.21),
  };
}

async function main() {
  const nFeatures = seedCache();
  check('committed model + features artifacts load', nFeatures > 0, `${nFeatures} features`);

  const predictor = new XGBoostPredictor();
  await predictor.load(MODEL_URL, FEATURES_URL); // hits cache, no fetch
  check('predictor.load() resolved from cache (offline)', predictor.loaded);

  // --- a known prime-central property: 2-bed 1000sqft flat in Chelsea (SW3) ---
  const subject = {
    bedrooms: 2,
    bathrooms: 2,
    size_sqft: 1000,
    postcode: 'SW3 5RA',
    property_type: 'flat',
    address: '10 Cheyne Walk',
    latitude: 51.4839,
    longitude: -0.17,
  };
  const r = predictFairValue(predictor, subject);

  check('estimate_pcm is a finite positive number', Number.isFinite(r.estimate_pcm) && r.estimate_pcm > 0, `${r.estimate_pcm}`);
  // Prime-central 2-bed should land in a broad-but-real band (£2k–£40k pcm). This is a
  // sanity envelope, not a golden value (the golden-£ parity is fixture_diff's job).
  check('estimate_pcm is in a sane prime-central band (2k–40k)', r.estimate_pcm >= 2000 && r.estimate_pcm <= 40000, `${r.estimate_pcm}`);
  check('model_version is v20', r.model_version === 'v20');
  check('currency is GBP', r.currency === 'GBP');
  check('range_low < estimate < range_high', r.range_low < r.estimate_pcm && r.estimate_pcm < r.range_high, `${r.range_low}/${r.estimate_pcm}/${r.range_high}`);
  check('range_low ≈ 0.79x estimate', r.range_low === Math.round(r.estimate_pcm * 0.79));
  check('range_high ≈ 1.21x estimate', r.range_high === Math.round(r.estimate_pcm * 1.21));

  // --- monotonic-ish sanity: a bigger, more-bedroomed flat should not be cheaper ---
  const bigger = predictFairValue(predictor, {
    ...subject, bedrooms: 3, bathrooms: 3, size_sqft: 1800,
  });
  check('a larger 3-bed estimate is >= the 2-bed estimate', bigger.estimate_pcm >= r.estimate_pcm, `${bigger.estimate_pcm} vs ${r.estimate_pcm}`);

  // --- determinism: same input -> same estimate ---
  const again = predictFairValue(predictor, subject);
  check('prediction is deterministic', again.estimate_pcm === r.estimate_pcm, `${again.estimate_pcm} vs ${r.estimate_pcm}`);

  // --- the route file WIRES this exact pipeline (no drift from production) ---
  const routeSrc = readFileSync(join(ROUTE_DIR, 'route.ts'), 'utf8');
  check('route computes estimate via Math.expm1', routeSrc.includes('Math.expm1('));
  check('route uses the 0.79 range floor', routeSrc.includes('0.79'));
  check('route uses the 1.21 range ceiling', routeSrc.includes('1.21'));
  check('route requires the vendored predictor', routeSrc.includes("require('./xgboost.predictor.js')"));
  check('route calls buildFeatures', routeSrc.includes('buildFeatures('));

  // --- the vendored predictor must be byte-identical to the source (parity lock) ---
  const src = readFileSync(join(ROOT, 'chrome-extension', 'xgboost.js'));
  const vend = readFileSync(VENDORED);
  check('vendored xgboost.predictor.js is byte-identical to chrome-extension/xgboost.js',
    Buffer.compare(src, vend) === 0);

  if (failures) {
    console.log(`\n=== FAIL: ${failures} /api/predict estimate check(s) failed. ===`);
    process.exit(1);
  }
  console.log(`\n=== PASS: /api/predict produces a sane v20 estimate (£${r.estimate_pcm} pcm for the SW3 subject). ===`);
}

main().catch((e) => {
  console.error('predict_estimate_test crashed:', e);
  process.exit(1);
});
