/**
 * EXTRACTION -> FEATURES BRIDGE TEST (the missing test CLASS, spec A11/R5).
 *
 * Every other extension harness tests ONE side of the pipe in isolation:
 *   - *_extract_test.mjs  : page DOM -> propertyData (extractors only)
 *   - fixture_diff.mjs    : synthetic raw inputs -> buildFeatures (parity only)
 * NOTHING tested the JOIN: the real content.js extract -> rawFields the model is
 * actually fed in the browser. That gap hid a real bug.
 *
 * THE BUG this locks down (R5): content.js built rawFields with `description:` and
 * NEVER set `summary`. The canonical model's text_blob = (summary + ' ' + description)
 * (rental_price_models_v20.py:888) and in the training DB the listing prose lives in
 * `summary` (rightmove + savills 100%) while `description` is ~100% EMPTY. So the
 * served JS fed the model a field it did not train on — `rawFields.summary` was
 * undefined. The fix populates rawFields.summary from the page prose (the same source
 * the model trains on) via the pure buildRawFields() seam.
 *
 * This test drives the REAL content.js path end to end against the REAL captured live
 * page tests/fixtures/extension/rightmove_169944029.html:
 *   1. extractPropertyDataRightmove()  -> propertyData      (real extractor)
 *   2. buildRawFields(propertyData,…)  -> rawFields         (real request contract)
 *   3. window.XGBFeatures.buildFeatures(rawFields) -> feats (real shared FE)
 *
 * Asserts:
 *   A. rawFields.summary is populated (FAILS before the fix — summary was never set).
 *   B. rawFields.summary carries the page prose (amenity keywords present in it).
 *   C. amenity feature(s) the page prose carries actually FIRE in buildFeatures
 *      (balcony/pool/gym/porter — the 169944029 prose lists all of them).
 *   D. NO load-bearing feature collapses to a default:
 *        - postcode district is the real SW1W, NOT the SW3 fillna fallback,
 *        - size is the page sqft (1539), NOT a beds-estimated value.
 *
 * Pure / network-free / deterministic: size comes from the page JSON (sizings=1539),
 * so no OCR/fetch is needed; buildRawFields is a pure seam (no async, no DOM).
 *
 * Run: node chrome-extension/extract_to_features_bridge_test.mjs   (exit 0 = pass)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { XGBFeatures, XGBoostPredictor } = require('./xgboost.js');

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const SRC = readFileSync(join(__dirname, 'content.js'), 'utf8');

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK   ${name}`);
  else { failures++; console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`); }
}

// ---------------------------------------------------------------------------
// Minimal DOM shim — same zero-dep parser the other extractor harnesses use, so
// the REAL extractPropertyDataRightmove (which reads window.__PAGE_MODEL out of a
// <script>) runs against the captured fixture. We reuse the shared shim's parser
// by importing loadExtractors only for the extractor; but we ALSO need
// buildRawFields out of the same closure AND window.XGBFeatures wired so
// buildFeatures is callable. The shared shim doesn't export buildRawFields or wire
// XGBFeatures, so we run our own instrumented load here (same trampoline trick as
// similar_properties_test.mjs), reusing the shim's DOM via loadExtractors for the
// extractor step and this loader for the bridge step.
// ---------------------------------------------------------------------------
import { loadExtractors } from './extract_test_shim.mjs';

// Load content.js with a sandbox whose `window` already carries the REAL XGBFeatures
// (from xgboost.js) and expose the pure seams we need (extractors + buildRawFields).
function loadBridge(html, { hostname, pathname }) {
  // Reuse the shared shim to get a faithful document/window for the fixture, then
  // run an instrumented copy of content.js that ALSO publishes buildRawFields.
  const baseEx = loadExtractors(html, { hostname, pathname });

  // Build a second sandbox for the bridge that mirrors the shim's window/document but
  // pre-populates window.XGBFeatures / window.XGBoostPredictor so buildRawFields and
  // (if needed) analyzeProperty's guard see the shared feature builder.
  const exported = {};
  const windowShim = {
    location: { hostname, pathname, href: `https://${hostname}${pathname}` },
    addEventListener() {},
    XGBFeatures,
    XGBoostPredictor,
    __rentFairValueLoaded: undefined,
  };
  windowShim.window = windowShim;

  // We only need buildRawFields + the read-only extractors out of the closure for the
  // bridge; the extractor RESULT comes from baseEx (already parsed against the fixture).
  // buildRawFields is pure (no DOM) once given propertyData, so a stub document suffices.
  const documentShim = {
    title: '', body: { appendChild() {} },
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => ({
      id: '', className: '', textContent: '', innerHTML: '', style: {}, dataset: {},
      setAttribute() {}, getAttribute: () => null, appendChild() {}, append() {},
      remove() {}, addEventListener() {}, click() {},
      classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    }),
    addEventListener() {},
  };
  const sandbox = {
    window: windowShim,
    document: documentShim,
    history: { pushState() {}, replaceState() {} },
    location: windowShim.location,
    setInterval: () => 0, clearInterval: () => {},
    setTimeout: () => 0, clearTimeout: () => {},
    MutationObserver: class { observe() {} disconnect() {} },
    fetch: async () => ({ ok: false, status: 599, json: async () => ({}) }),
    Tesseract: undefined,
    chrome: { runtime: { sendMessage: () => {}, getURL: () => '', lastError: null } },
    console: { log() {}, warn() {}, error() {} },
    __export: exported,
  };

  const EXPORT_SHIM = '\n;' +
    ['buildRawFields', 'extractListingProse'].map(
      (n) => `try{__export.${n}=${n};}catch(e){__export.${n}=undefined;}`
    ).join('') + '\n';
  const closeIdx = SRC.lastIndexOf('})();');
  if (closeIdx === -1) throw new Error('could not find IIFE close `})();` in content.js');
  const instrumented = SRC.slice(0, closeIdx) + EXPORT_SHIM + SRC.slice(closeIdx);
  const names = Object.keys(sandbox);
  const vals = names.map((n) => sandbox[n]);
  // eslint-disable-next-line no-new-func
  const fn = new Function(...names, `'use strict';\n${instrumented}`);
  fn(...vals);
  return { ...baseEx, buildRawFields: exported.buildRawFields, extractListingProse: exported.extractListingProse };
}

// =============================================================================
// REAL FIXTURE — rightmove_169944029.html (2-bed SW1W, page sqft=1539, rich prose:
// "Balcony", "Residents' pool, spa & gym", "Concierge", "Allocated parking included").
// =============================================================================
{
  const html = readFileSync(join(ROOT, 'tests/fixtures/extension', 'rightmove_169944029.html'), 'utf8');
  const ex = loadBridge(html, { hostname: 'www.rightmove.co.uk', pathname: '/properties/169944029' });

  check('bridge: content.js exposes the buildRawFields seam',
    typeof ex.buildRawFields === 'function', `got ${typeof ex.buildRawFields}`);

  // 1. REAL extractor -> propertyData
  const pd = ex.extractPropertyDataRightmove();
  check('bridge: extractor returns propertyData from the real fixture',
    pd && typeof pd === 'object', `got ${pd}`);

  // size comes from the page JSON (sizings=1539) -> no OCR/network needed.
  const sizeSqft = ex.extractSqftFromPage(pd);
  check('bridge: page sqft recovered (1539) so size is NOT beds-estimated',
    sizeSqft === 1539, `got ${sizeSqft}`);

  // 2. REAL request-contract builder -> rawFields (size already recovered above).
  const rawFields = ex.buildRawFields(pd, { sizeSqft, floors: {}, ocrText: '' });

  // ---- A. THE RED ASSERTION: summary must be populated (was never set pre-fix) ----
  check('bridge[A]: rawFields.summary is populated (R5 — model trains on `summary`)',
    typeof rawFields.summary === 'string' && rawFields.summary.trim().length > 0,
    `summary=${JSON.stringify(rawFields.summary)}`);

  // ---- B. summary carries the real page prose (the amenity keywords live in it) ----
  const sumLower = String(rawFields.summary || '').toLowerCase();
  check('bridge[B]: rawFields.summary carries the page prose (balcony/pool/gym keywords)',
    sumLower.includes('balcony') && sumLower.includes('pool') && sumLower.includes('gym'),
    `summary(first120)=${JSON.stringify(sumLower.slice(0, 120))}`);

  // 3. REAL shared feature builder over the rawFields the model is actually fed.
  const feats = XGBFeatures.buildFeatures(rawFields);

  // ---- C. amenity feature(s) the prose carries actually FIRE ----
  check('bridge[C]: has_balcony fires from the page prose', feats.has_balcony === 1, `got ${feats.has_balcony}`);
  check('bridge[C]: has_pool fires from the page prose', feats.has_pool === 1, `got ${feats.has_pool}`);
  check('bridge[C]: has_gym fires from the page prose', feats.has_gym === 1, `got ${feats.has_gym}`);
  check('bridge[C]: has_porter fires from the page prose', feats.has_porter === 1, `got ${feats.has_porter}`);

  // The amenity wiring is the load-bearing claim: the prose-driven features must be
  // sourced from `summary` (the trained field), not silently from a stray description.
  // Cross-check: with summary BLANKED and ONLY description carrying prose, the model
  // would still fire (text_blob includes both) — but the contract the model trains on
  // requires the prose under `summary`. Assert summary alone is sufficient:
  const summaryOnly = XGBFeatures.buildFeatures({ ...rawFields, description: '' });
  check('bridge[C2]: amenities fire from summary ALONE (description blanked)',
    summaryOnly.has_balcony === 1 && summaryOnly.has_pool === 1 && summaryOnly.has_gym === 1,
    `balcony=${summaryOnly.has_balcony} pool=${summaryOnly.has_pool} gym=${summaryOnly.has_gym}`);

  // ---- D. NO load-bearing feature collapses to a default ----
  // The v20 FE frequency-encodes the postcode district (no per-district one-hot), so a
  // "geography collapse" shows up as postcode_freq snapping to the UNKNOWN/SW3 fallback
  // value instead of the real district's. Reference values from the committed model's
  // POSTCODE_FREQ map: SW1W=0.027521879…, SW3 fillna=0.084062643…, UNKNOWN=0.122524182…
  const SW1W_FREQ = 0.027521879318286504;
  const SW3_FREQ = 0.08406264394288346;
  const UNKNOWN_FREQ = 0.12252418240442192;
  // What the SAME rawFields would encode to if the postcode collapsed to empty (UNKNOWN)
  // — an independent recompute, so the check fails loudly if the real postcode is lost.
  const collapsed = XGBFeatures.buildFeatures({ ...rawFields, postcode: '', postcode_normalized: '' });
  check('bridge[D1]: postcode_freq is the REAL SW1W value, NOT the UNKNOWN/SW3 fallback',
    Math.abs(feats.postcode_freq - SW1W_FREQ) < 1e-9 &&
    Math.abs(feats.postcode_freq - UNKNOWN_FREQ) > 1e-6 &&
    Math.abs(feats.postcode_freq - SW3_FREQ) > 1e-6,
    `postcode_freq=${feats.postcode_freq} (collapsed→${collapsed.postcode_freq}, SW1W=${SW1W_FREQ})`);

  // D2: size feature reflects the page sqft (1539), not a beds estimate.
  // log_sqft = log1p(1539) ≈ 7.3397; size_sqft is the raw feature too.
  const expectedLogSqft = Math.log1p(1539);
  check('bridge[D2]: size feature uses page sqft 1539 (not beds-estimated)',
    Math.abs((feats.size_sqft ?? 0) - 1539) < 1e-9 &&
    Math.abs((feats.log_sqft ?? 0) - expectedLogSqft) < 1e-6,
    `size_sqft=${feats.size_sqft} log_sqft=${feats.log_sqft} expected log≈${expectedLogSqft.toFixed(4)}`);

  // sanity: a real £ comes out (model present) — proves the bridge end to end.
  // (Skipped silently if the committed model is absent — feature asserts above are
  // the load-bearing checks.)
  try {
    const model = JSON.parse(readFileSync(join(__dirname, 'api', 'model.json'), 'utf8'));
    const features = JSON.parse(readFileSync(join(__dirname, 'api', 'features.json'), 'utf8'));
    const predictor = new XGBoostPredictor();
    predictor.model = model; predictor.features = features; predictor.loaded = true;
    const gbp = Math.expm1(predictor.predict(feats));
    check('bridge: end-to-end £ estimate is a finite positive number',
      Number.isFinite(gbp) && gbp > 0, `got £${gbp}`);
  } catch (e) {
    console.log(`SKIP end-to-end £ check (committed model absent: ${e.message})`);
  }
}

console.log('');
if (failures === 0) {
  console.log('=== PASS: extract->features bridge green (rawFields.summary populated, prose amenities fire, no default collapse) ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) failed (RED until content.js populates rawFields.summary from the page prose) ===`);
  process.exit(1);
}
