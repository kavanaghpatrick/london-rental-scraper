/**
 * A6 — RENTAL route-HANDLER tests (Wave 2, Group SERVING).
 *
 * Unlike predict_estimate_test.mjs (which re-implements the route's envelope math and
 * asserts the route *file text* wires it), THIS harness invokes the ACTUAL Next.js
 * handlers — POST / GET / OPTIONS exported by the real route.ts — via a sucrase
 * transpile (_route_loader.mjs). So it proves the HTTP contract the deployed function
 * actually returns:
 *
 *   /api/predict (POST):
 *     - 400 on malformed JSON body
 *     - 400 on a body missing a required field (bathrooms / size_sqft / postcode / type)
 *     - OPTIONS returns CORS headers (Allow-Origin *, Allow-Methods POST, OPTIONS)
 *     - 503 when the predictor backend is not ready (model fetch fails -> PredictorNotReady)
 *     - 200 with a sane v20 estimate when the model cache is pre-seeded (offline success),
 *       proving the 503 path is a real not-ready signal, not a permanent failure.
 *
 *   /api/similar (GET):
 *     - OPTIONS returns CORS headers (Allow-Methods GET, OPTIONS)
 *     - 400 on missing required params, 400 on invalid beds / price
 *     - 500 when the DB layer throws a GENUINE error  ── the "500-vs-empty" distinction:
 *     - 200 (NOT 500) with peer_count 0 when the DB layer returns empty peers
 *     The DB layer (@/lib/db) is stubbed via a require-override so we exercise the route's
 *     own try/catch + response envelope WITHOUT a Postgres (the real query is covered by
 *     similar_query_test.mjs against a service container). No dashboard/src is edited.
 *
 * Offline + deterministic: no network, no Postgres. Run: node dashboard/test/route_handler_test.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadRoute, getRequest, postRequest } from './_route_loader.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..');
const PREDICT_ROUTE = join(__dirname, '..', 'src', 'app', 'api', 'predict', 'route.ts');
const SIMILAR_ROUTE = join(__dirname, '..', 'src', 'app', 'api', 'similar', 'route.ts');
const API_DIR = join(ROOT, 'chrome-extension', 'api');

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK   ${name}`);
  else { failures++; console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`); }
}

// The GH-raw URL pair the predict route loads from (parsed out of route.ts so the model
// cache key we pre-seed matches the route's exact URLs — no hardcoded duplication).
function predictModelUrls(routeSrc) {
  const m = routeSrc.match(/GH_RAW_BASE\s*=\s*\n?\s*['"`]([^'"`]+)['"`]/);
  if (!m) throw new Error('could not parse GH_RAW_BASE from predict/route.ts');
  const base = m[1];
  return { MODEL_URL: `${base}/model.json`, FEATURES_URL: `${base}/features.json` };
}

async function testPredictRoute() {
  console.log('\n--- /api/predict (real handler) ---');
  const routeSrc = readFileSync(PREDICT_ROUTE, 'utf8');

  // OPTIONS -> CORS
  {
    const route = loadRoute(PREDICT_ROUTE);
    const res = await route.OPTIONS();
    check('predict OPTIONS -> 200', res.status === 200, `${res.status}`);
    check('predict OPTIONS sets Allow-Origin *', res.headers.get('access-control-allow-origin') === '*');
    check('predict OPTIONS Allow-Methods includes POST',
      (res.headers.get('access-control-allow-methods') || '').includes('POST'),
      res.headers.get('access-control-allow-methods'));
  }

  // 400 on malformed JSON
  {
    const route = loadRoute(PREDICT_ROUTE);
    const res = await route.POST(postRequest('http://localhost/api/predict', '{ not json'));
    check('predict malformed body -> 400', res.status === 400, `${res.status}`);
    const body = await res.json();
    check('predict malformed body -> JSON error message', typeof body.error === 'string', JSON.stringify(body));
    check('predict 400 carries CORS', res.headers.get('access-control-allow-origin') === '*');
  }

  // 400 on each missing required field
  {
    const route = loadRoute(PREDICT_ROUTE);
    const cases = [
      ['missing bathrooms', { bedrooms: 2, size_sqft: 1000, postcode: 'SW3 5RA', property_type: 'flat' }],
      ['missing size_sqft', { bedrooms: 2, bathrooms: 2, postcode: 'SW3 5RA', property_type: 'flat' }],
      ['missing postcode',  { bedrooms: 2, bathrooms: 2, size_sqft: 1000, property_type: 'flat' }],
      ['missing property_type', { bedrooms: 2, bathrooms: 2, size_sqft: 1000, postcode: 'SW3 5RA' }],
      ['zero size_sqft (must be positive)', { bedrooms: 2, bathrooms: 2, size_sqft: 0, postcode: 'SW3 5RA', property_type: 'flat' }],
    ];
    for (const [label, body] of cases) {
      const res = await route.POST(postRequest('http://localhost/api/predict', JSON.stringify(body)));
      check(`predict ${label} -> 400`, res.status === 400, `${res.status}`);
    }
  }

  // 503 when the predictor backend is not ready (model fetch fails).
  {
    const savedFetch = globalThis.fetch;
    globalThis.fetch = async () => { throw new Error('getaddrinfo ENOTFOUND raw.githubusercontent.com'); };
    try {
      const route = loadRoute(PREDICT_ROUTE);
      const valid = { bedrooms: 2, bathrooms: 2, size_sqft: 1000, postcode: 'SW3 5RA', property_type: 'flat' };
      const res = await route.POST(postRequest('http://localhost/api/predict', JSON.stringify(valid)));
      check('predict not-ready (fetch fails) -> 503', res.status === 503, `${res.status}`);
      const body = await res.json();
      check('predict 503 tags model_version v20', body.model_version === 'v20', JSON.stringify(body));
      check('predict 503 carries CORS', res.headers.get('access-control-allow-origin') === '*');
    } finally {
      globalThis.fetch = savedFetch;
    }
  }

  // 200 success when the model cache is pre-seeded from the committed v20 artifacts.
  // (Proves the 503 above is a genuine not-ready signal, not a permanent dead route, and
  //  exercises the route's real predictFairValue -> expm1 -> range envelope.)
  {
    const { MODEL_URL, FEATURES_URL } = predictModelUrls(routeSrc);
    const model = JSON.parse(readFileSync(join(API_DIR, 'model.json'), 'utf8'));
    const features = JSON.parse(readFileSync(join(API_DIR, 'features.json'), 'utf8'));
    const cache = globalThis.__XGB_MODEL_CACHE__ || (globalThis.__XGB_MODEL_CACHE__ = new Map());
    cache.set(`${MODEL_URL}|${FEATURES_URL}`, { model, features });

    const savedFetch = globalThis.fetch;
    globalThis.fetch = async () => { throw new Error('network must not be hit when cache is seeded'); };
    try {
      const route = loadRoute(PREDICT_ROUTE);
      const valid = {
        bedrooms: 2, bathrooms: 2, size_sqft: 1000, postcode: 'SW3 5RA',
        property_type: 'flat', address: '10 Cheyne Walk', latitude: 51.4839, longitude: -0.17,
      };
      const res = await route.POST(postRequest('http://localhost/api/predict', JSON.stringify(valid)));
      check('predict seeded-cache success -> 200', res.status === 200, `${res.status}`);
      const body = await res.json();
      check('predict 200 returns finite positive estimate_pcm',
        Number.isFinite(body.estimate_pcm) && body.estimate_pcm > 0, `${body.estimate_pcm}`);
      check('predict 200 model_version v20', body.model_version === 'v20', `${body.model_version}`);
      check('predict 200 range_low < estimate < range_high',
        body.range_low < body.estimate_pcm && body.estimate_pcm < body.range_high,
        `${body.range_low}/${body.estimate_pcm}/${body.range_high}`);
    } finally {
      globalThis.fetch = savedFetch;
    }
  }
}

async function testSimilarRoute() {
  console.log('\n--- /api/similar (real handler) ---');

  // A controllable stub for @/lib/db so we drive the route's try/catch WITHOUT a Postgres.
  let dbBehavior = () => ({ peers: [], stats: { peer_count: 0, your_percentile: 50 } });
  const dbStub = { getSimilarListings: async (p) => dbBehavior(p) };
  const route = loadRoute(SIMILAR_ROUTE, { '@/lib/db': dbStub });

  // OPTIONS -> CORS
  {
    const res = await route.OPTIONS();
    check('similar OPTIONS -> 200', res.status === 200, `${res.status}`);
    check('similar OPTIONS sets Allow-Origin *', res.headers.get('access-control-allow-origin') === '*');
    check('similar OPTIONS Allow-Methods includes GET',
      (res.headers.get('access-control-allow-methods') || '').includes('GET'),
      res.headers.get('access-control-allow-methods'));
  }

  // 400 missing required params
  {
    const res = await route.GET(getRequest('http://localhost/api/similar'));
    check('similar missing params -> 400', res.status === 400, `${res.status}`);
  }

  // 400 invalid beds / price
  {
    const r1 = await route.GET(getRequest('http://localhost/api/similar?postcode=SW3&beds=abc&price=5000'));
    check('similar invalid beds -> 400', r1.status === 400, `${r1.status}`);
    const r2 = await route.GET(getRequest('http://localhost/api/similar?postcode=SW3&beds=2&price=0'));
    check('similar non-positive price -> 400', r2.status === 400, `${r2.status}`);
  }

  // 500-vs-empty distinction:
  // (a) a GENUINE DB error -> 500
  {
    dbBehavior = () => { throw new Error('connection refused'); };
    const res = await route.GET(getRequest('http://localhost/api/similar?postcode=SW3%204AJ&beds=2&price=5000'));
    check('similar genuine DB error -> 500', res.status === 500, `${res.status}`);
    check('similar 500 carries CORS', res.headers.get('access-control-allow-origin') === '*');
    const body = await res.json();
    check('similar 500 returns generic error (no leak)', body.error === 'Internal server error', JSON.stringify(body));
  }
  // (b) empty peers -> 200 with peer_count 0 (NOT 500)
  {
    dbBehavior = () => ({ peers: [], stats: { peer_count: 0, your_percentile: 50 } });
    const res = await route.GET(getRequest('http://localhost/api/similar?postcode=SW3%204AJ&beds=2&price=5000'));
    check('similar empty peers -> 200 (not 500)', res.status === 200, `${res.status}`);
    const body = await res.json();
    check('similar empty peers -> peer_count 0', body.stats?.peer_count === 0, JSON.stringify(body.stats));
    check('similar 200 attaches query_ms', typeof body.query_ms === 'number', `${body.query_ms}`);
  }
}

async function main() {
  await testPredictRoute();
  await testSimilarRoute();

  if (failures) {
    console.log(`\n=== FAIL: ${failures} rental route-handler check(s) failed. ===`);
    process.exit(1);
  }
  console.log('\n=== PASS: /api/predict + /api/similar real handlers return the documented HTTP contract. ===');
}

main().catch((e) => {
  console.error('route_handler_test crashed:', e);
  process.exit(1);
});
