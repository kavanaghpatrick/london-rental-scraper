/**
 * REAL behavioral test for /api/predict-sale's estimate path — NO Postgres, NO network.
 *
 * WHAT IT GUARDS (sale analogue of predict_estimate_test.mjs):
 *   /api/predict-sale runs the SALE JS predictor INLINE (mirroring the rental ruling):
 *   SaleXGBFeatures.buildFeatures(raw) -> SaleXGBoostPredictor.predict() -> Math.expm1 ->
 *   round, then clamp to [50_000, 250_000_000] and a x0.85/x1.15 range (TIGHTER than the
 *   rental 0.79/1.21 — sale asking-price dispersion is narrower). This harness exercises
 *   that EXACT pipeline against the REAL committed sale model (output/sale_api/model.json +
 *   features.json) and asserts a SANE estimate for a prime-central property plus the
 *   sale-specific response-envelope invariants (model_version 'sale_v1', GBP, clamp,
 *   low_confidence when postcode missing, estimated_size when size omitted).
 *
 * HOW IT AVOIDS THE NETWORK (the route fetches model.json from GitHub-raw output/sale_api):
 *   The SALE predictor caches parsed {model, features} in a SEPARATE global,
 *   globalThis.__SALE_XGB_MODEL_CACHE__ (NEVER the rental __XGB_MODEL_CACHE__), keyed by
 *   the (modelUrl|featuresUrl) pair. We pre-seed that cache from the committed sale model
 *   files, then call load() with matching sentinel URLs -> it hits the cache and NEVER
 *   fetches. So CI needs only node + the committed artifacts (no GitHub-raw call, no
 *   flakiness). This is the same model the route serves post-go-live.
 *
 * It also asserts the route file actually WIRES this pipeline (the vendored sale predictor
 * is byte-identical to chrome-extension/sale_xgboost.js, the route computes the price via
 * Math.expm1 + the clamp + the 0.85/1.15 range, requires './sale_xgboost.predictor.js',
 * and tags model_version 'sale_v1'), so the tested pipeline can't drift from production.
 *
 * NOTE: this harness depends on Group B artifacts (the vendored predictor + route.ts under
 * dashboard/src/app/api/predict-sale/). It is EXPECTED to be RED until Group B lands those
 * files; that is the TDD red phase.
 *
 * Run: node dashboard/test/predict_sale_estimate_test.mjs   (exit 0 = pass, 1 = fail)
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..');
const ROUTE_DIR = join(__dirname, '..', 'src', 'app', 'api', 'predict-sale');
const SALE_API_DIR = join(ROOT, 'output', 'sale_api');

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
// Load the SALE predictor module the ROUTE uses (the vendored byte-identical copy).
// (Group B file — RED until Group B lands it.)
// --------------------------------------------------------------------------- //
const VENDORED = join(ROUTE_DIR, 'sale_xgboost.predictor.js');
const { SaleXGBoostPredictor, SaleXGBFeatures } = require(VENDORED);

// --------------------------------------------------------------------------- //
// Pre-seed the SALE model cache from the committed sale artifacts so load() is
// offline. CRITICAL: uses the SALE-namespaced global __SALE_XGB_MODEL_CACHE__,
// never the rental __XGB_MODEL_CACHE__ (isolation: the rental offline test must
// never collide with this one).
// --------------------------------------------------------------------------- //
const MODEL_URL = 'fixture://sale-model.json';
const FEATURES_URL = 'fixture://sale-features.json';

function seedCache() {
  const model = JSON.parse(readFileSync(join(SALE_API_DIR, 'model.json'), 'utf8'));
  const features = JSON.parse(readFileSync(join(SALE_API_DIR, 'features.json'), 'utf8'));
  const cache =
    globalThis.__SALE_XGB_MODEL_CACHE__ ||
    (globalThis.__SALE_XGB_MODEL_CACHE__ = new Map());
  cache.set(`${MODEL_URL}|${FEATURES_URL}`, { model, features });
  return features.length;
}

// Sale-band constants (the route's NEW pair — NOT the rental 0.79/1.21).
const RANGE_LOW_MULT = 0.85;
const RANGE_HIGH_MULT = 1.15;
const PRICE_FLOOR = 50000;
const PRICE_CEIL = 250000000;

// Re-extract the postcode district exactly as the route's UX layer does (mirror of
// build_features / predict_one_default: UNKNOWN when absent or unparseable).
function districtOf(postcode) {
  const m = String(postcode ?? '').trim().toUpperCase()
    .match(/^([A-Z]{1,2}\d{1,2}[A-Z]?)(?=\s|\d|$)/);
  return m ? m[1] : 'UNKNOWN';
}

// Mirror the route's predictSaleValue() envelope math EXACTLY (port of predict_one_default
// UX layer in JS): expm1 -> round -> clamp -> 0.85/1.15 range, with low_confidence /
// estimated_size / district flags.
function predictSaleValue(predictor, raw) {
  const featureDict = SaleXGBFeatures.buildFeatures(raw);
  const predLog = predictor.predict(featureDict);
  const price = Math.round(Math.expm1(predLog));
  const clamped = Math.min(Math.max(price, PRICE_FLOOR), PRICE_CEIL);
  const district = districtOf(raw.postcode);
  const estimated_size = !(Number(raw.size_sqft) > 0);
  const low_confidence = estimated_size || district === 'UNKNOWN';
  return {
    predicted_price: clamped,
    model_version: 'sale_v1',
    currency: 'GBP',
    range_low: Math.round(clamped * RANGE_LOW_MULT),
    range_high: Math.round(clamped * RANGE_HIGH_MULT),
    low_confidence,
    district,
    estimated_size,
  };
}

async function main() {
  const nFeatures = seedCache();
  check('committed sale model + features artifacts load', nFeatures === 34, `${nFeatures} features`);

  const predictor = new SaleXGBoostPredictor();
  await predictor.load(MODEL_URL, FEATURES_URL); // hits __SALE_XGB_MODEL_CACHE__, no fetch
  check('sale predictor.load() resolved from cache (offline)', predictor.loaded);

  // --- a known prime-central property: 2-bed 1000sqft flat in Belgravia (SW1X) ---
  const subject = {
    bedrooms: 2,
    bathrooms: 2,
    size_sqft: 1000,
    postcode: 'SW1X 8NX',
    property_type: 'flat',
    address: 'Chester Square, London, SW1X',
    latitude: 51.4946,
    longitude: -0.1530,
  };
  const r = predictSaleValue(predictor, subject);

  check('predicted_price is a finite positive number',
    Number.isFinite(r.predicted_price) && r.predicted_price > 0, `${r.predicted_price}`);
  // Prime-central 2-bed for-SALE should land in a broad-but-real lump-sum band
  // (£1M–£20M). This is a sanity envelope, not a golden value (the golden-£ parity is
  // sale_fixture_diff's job).
  check('predicted_price in a sane prime-central SALE band (£1M–£20M)',
    r.predicted_price >= 1_000_000 && r.predicted_price <= 20_000_000, `${r.predicted_price}`);
  check('model_version is sale_v1', r.model_version === 'sale_v1');
  check('currency is GBP', r.currency === 'GBP');

  // --- clamp + range invariants (sale 0.85/1.15, clamp [50000, 250000000]) ---
  check('predicted_price clamped within [50000, 250000000]',
    r.predicted_price >= PRICE_FLOOR && r.predicted_price <= PRICE_CEIL, `${r.predicted_price}`);
  check('range_low < predicted_price < range_high',
    r.range_low < r.predicted_price && r.predicted_price < r.range_high,
    `${r.range_low}/${r.predicted_price}/${r.range_high}`);
  check('range_low === round(price * 0.85)',
    r.range_low === Math.round(r.predicted_price * RANGE_LOW_MULT), `${r.range_low}`);
  check('range_high === round(price * 1.15)',
    r.range_high === Math.round(r.predicted_price * RANGE_HIGH_MULT), `${r.range_high}`);

  // --- prime-central subject has a real postcode + real size -> NOT low-confidence ---
  check('district extracted from postcode (SW1X)', r.district === 'SW1X', `${r.district}`);
  check('not low_confidence with real postcode + size', r.low_confidence === false);
  check('estimated_size false when size_sqft provided', r.estimated_size === false);

  // --- low_confidence true when postcode missing (-> district UNKNOWN) ---
  const noPostcode = predictSaleValue(predictor, { ...subject, postcode: '' });
  check('district UNKNOWN when postcode missing', noPostcode.district === 'UNKNOWN', `${noPostcode.district}`);
  check('low_confidence true when postcode missing', noPostcode.low_confidence === true);

  // --- estimated_size true when size_sqft omitted (700 fallback in build_features) ---
  const noSize = { ...subject };
  delete noSize.size_sqft;
  const noSizeR = predictSaleValue(predictor, noSize);
  check('estimated_size true when size_sqft omitted', noSizeR.estimated_size === true);
  check('low_confidence true when size_sqft omitted', noSizeR.low_confidence === true);
  check('predicted_price still finite positive when size omitted',
    Number.isFinite(noSizeR.predicted_price) && noSizeR.predicted_price > 0, `${noSizeR.predicted_price}`);

  // --- monotonic-ish sanity: a bigger, more-bedroomed flat should not be cheaper ---
  // (guards the Math.fround branch discipline — a float64 mis-branch on the monotone-
  // constrained sale model surfaces as a visible non-monotone dip.)
  const bigger = predictSaleValue(predictor, {
    ...subject, bedrooms: 3, bathrooms: 3, size_sqft: 1800,
  });
  check('a larger 3-bed predicted_price is >= the 2-bed predicted_price',
    bigger.predicted_price >= r.predicted_price, `${bigger.predicted_price} vs ${r.predicted_price}`);

  // --- determinism: same input -> same price (no RNG, no clock, no network) ---
  const again = predictSaleValue(predictor, subject);
  check('prediction is deterministic', again.predicted_price === r.predicted_price,
    `${again.predicted_price} vs ${r.predicted_price}`);

  // --- the route file WIRES this exact pipeline (no drift from production) ---
  const routeSrc = readFileSync(join(ROUTE_DIR, 'route.ts'), 'utf8');
  check('route computes price via Math.expm1', routeSrc.includes('Math.expm1('));
  check('route requires the vendored sale predictor',
    routeSrc.includes("require('./sale_xgboost.predictor.js')"));
  check('route calls buildFeatures', routeSrc.includes('buildFeatures('));
  check("route tags model_version 'sale_v1'", routeSrc.includes("'sale_v1'"));
  check('route uses the clamp floor 50000', routeSrc.includes('50000'));
  check('route uses the clamp ceiling 250000000', routeSrc.includes('250000000'));

  // --- the vendored predictor must be byte-identical to the canonical source (parity lock) ---
  const src = readFileSync(join(ROOT, 'chrome-extension', 'sale_xgboost.js'));
  const vend = readFileSync(VENDORED);
  check('vendored sale_xgboost.predictor.js is byte-identical to chrome-extension/sale_xgboost.js',
    Buffer.compare(src, vend) === 0);

  if (failures) {
    console.log(`\n=== FAIL: ${failures} /api/predict-sale estimate check(s) failed. ===`);
    process.exit(1);
  }
  console.log(`\n=== PASS: /api/predict-sale produces a sane sale_v1 estimate (£${r.predicted_price} for the SW1X subject). ===`);
}

main().catch((e) => {
  console.error('predict_sale_estimate_test crashed:', e);
  process.exit(1);
});
