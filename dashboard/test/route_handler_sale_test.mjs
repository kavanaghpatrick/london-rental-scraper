/**
 * A6 — FOR-SALE route-HANDLER tests (Wave 2, Group SERVING).
 *
 * Sale analogue of route_handler_test.mjs. Invokes the ACTUAL Next.js handlers exported
 * by the real /api/predict-sale and /api/similar-sale route.ts (sucrase transpile via
 * _route_loader.mjs), asserting the real HTTP contract — including the sale-specific
 * DIVERGENCES from the rental routes:
 *
 *   /api/predict-sale (POST):
 *     - 400 only on a non-object / malformed body (postcode + size_sqft are NOT hard-
 *       required — the rental route 400s on those; the sale route does NOT)
 *     - a missing postcode / missing size_sqft body still validates (NOT 400) and the UX
 *       signal is carried by low_confidence / estimated_size in the 200/503 body, not a 400
 *     - OPTIONS returns CORS (Allow-Methods POST, OPTIONS)
 *     - 503 when the sale predictor backend is not ready (model fetch fails)
 *     - 200 with a sane sale_v1 price + sale band (0.85/1.15) + low_confidence/district/
 *       estimated_size flags when the SALE model cache is pre-seeded (offline success)
 *
 *   /api/similar-sale (GET):
 *     - OPTIONS CORS, 400 missing params, 400 invalid beds/price
 *     - 500-vs-empty: a GENUINE DB error -> 500; an empty result -> 200 peer_count 0
 *
 * Offline + deterministic: no network, no Postgres. The sale model cache is the
 * SALE-namespaced global (__SALE_XGB_MODEL_CACHE__), never the rental one.
 * Run: node dashboard/test/route_handler_sale_test.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadRoute, getRequest, postRequest } from './_route_loader.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..');
const PREDICT_SALE_ROUTE = join(__dirname, '..', 'src', 'app', 'api', 'predict-sale', 'route.ts');
const SIMILAR_SALE_ROUTE = join(__dirname, '..', 'src', 'app', 'api', 'similar-sale', 'route.ts');
const SALE_API_DIR = join(ROOT, 'output', 'sale_api');

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK   ${name}`);
  else { failures++; console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`); }
}

// Parse the SALE GH-raw URL pair out of the route so the seeded model-cache key matches.
function saleModelUrls(routeSrc) {
  const m = routeSrc.match(/SALE_GH_RAW_BASE\s*=\s*\n?\s*['"`]([^'"`]+)['"`]/);
  if (!m) throw new Error('could not parse SALE_GH_RAW_BASE from predict-sale/route.ts');
  const base = m[1];
  return { MODEL_URL: `${base}/model.json`, FEATURES_URL: `${base}/features.json` };
}

async function testPredictSaleRoute() {
  console.log('\n--- /api/predict-sale (real handler) ---');
  const routeSrc = readFileSync(PREDICT_SALE_ROUTE, 'utf8');

  // OPTIONS -> CORS
  {
    const route = loadRoute(PREDICT_SALE_ROUTE);
    const res = await route.OPTIONS();
    check('predict-sale OPTIONS -> 200', res.status === 200, `${res.status}`);
    check('predict-sale OPTIONS sets Allow-Origin *', res.headers.get('access-control-allow-origin') === '*');
    check('predict-sale OPTIONS Allow-Methods includes POST',
      (res.headers.get('access-control-allow-methods') || '').includes('POST'),
      res.headers.get('access-control-allow-methods'));
  }

  // 400 only on malformed JSON / non-object body.
  // NOTE: an array body proceeds PAST validate() into the predictor (arrays are objects),
  // which would make a REAL GH-raw fetch and could seed the model cache, poisoning the 503
  // test below. So we block the network here and assert the array case is simply NOT a 400.
  {
    const savedFetch = globalThis.fetch;
    globalThis.fetch = async () => { throw new Error('network blocked in 400-validation block'); };
    try {
      const route = loadRoute(PREDICT_SALE_ROUTE);
      const res = await route.POST(postRequest('http://localhost/api/predict-sale', '{ not json'));
      check('predict-sale malformed body -> 400', res.status === 400, `${res.status}`);
      check('predict-sale 400 carries CORS', res.headers.get('access-control-allow-origin') === '*');

      const arr = await route.POST(postRequest('http://localhost/api/predict-sale', JSON.stringify([1, 2, 3])));
      // Arrays ARE objects in JS — validate() only rejects non-object/null, so an array body
      // does NOT 400 here. It proceeds to the predictor (which 503s offline). Assert NOT 400
      // to pin the documented "only non-object body 400s" contract.
      check('predict-sale array body is NOT a 400 (only non-object/null 400s)', arr.status !== 400, `${arr.status}`);
    } finally {
      globalThis.fetch = savedFetch;
    }
  }

  // DIVERGENCE: a missing postcode / missing size_sqft body must NOT 400 (the rental route
  // would). With no model loaded it 503s — the point is it is NOT a 400 validation reject.
  {
    const savedFetch = globalThis.fetch;
    globalThis.fetch = async () => { throw new Error('ENOTFOUND (not-ready)'); };
    try {
      const route = loadRoute(PREDICT_SALE_ROUTE);
      const noPostcode = { bedrooms: 2, bathrooms: 2, size_sqft: 1000, property_type: 'flat' };
      const r1 = await route.POST(postRequest('http://localhost/api/predict-sale', JSON.stringify(noPostcode)));
      check('predict-sale missing postcode is NOT 400 (sale divergence)', r1.status !== 400, `${r1.status}`);

      const noSize = { bedrooms: 2, bathrooms: 2, postcode: 'SW1X 8NX', property_type: 'flat' };
      const r2 = await route.POST(postRequest('http://localhost/api/predict-sale', JSON.stringify(noSize)));
      check('predict-sale missing size_sqft is NOT 400 (sale divergence)', r2.status !== 400, `${r2.status}`);
    } finally {
      globalThis.fetch = savedFetch;
    }
  }

  // 503 when the sale predictor backend is not ready.
  // Defensively evict any cached parse for this URL pair so load() is forced to fetch (and
  // fail) — guarantees the not-ready path regardless of earlier blocks' global state.
  {
    const { MODEL_URL, FEATURES_URL } = saleModelUrls(routeSrc);
    if (globalThis.__SALE_XGB_MODEL_CACHE__) {
      globalThis.__SALE_XGB_MODEL_CACHE__.delete(`${MODEL_URL}|${FEATURES_URL}`);
    }
    const savedFetch = globalThis.fetch;
    globalThis.fetch = async () => { throw new Error('getaddrinfo ENOTFOUND raw.githubusercontent.com'); };
    try {
      const route = loadRoute(PREDICT_SALE_ROUTE);
      const valid = { bedrooms: 2, bathrooms: 2, size_sqft: 1000, postcode: 'SW1X 8NX', property_type: 'flat' };
      const res = await route.POST(postRequest('http://localhost/api/predict-sale', JSON.stringify(valid)));
      check('predict-sale not-ready (fetch fails) -> 503', res.status === 503, `${res.status}`);
      const body = await res.json();
      check('predict-sale 503 tags model_version sale_v1', body.model_version === 'sale_v1', JSON.stringify(body));
      check('predict-sale 503 carries CORS', res.headers.get('access-control-allow-origin') === '*');
    } finally {
      globalThis.fetch = savedFetch;
    }
  }

  // 200 success with the SALE model cache pre-seeded (offline). Exercises the route's real
  // predictSaleValue -> expm1 -> clamp -> 0.85/1.15 range + UX flags.
  {
    const { MODEL_URL, FEATURES_URL } = saleModelUrls(routeSrc);
    const model = JSON.parse(readFileSync(join(SALE_API_DIR, 'model.json'), 'utf8'));
    const features = JSON.parse(readFileSync(join(SALE_API_DIR, 'features.json'), 'utf8'));
    const cache = globalThis.__SALE_XGB_MODEL_CACHE__ || (globalThis.__SALE_XGB_MODEL_CACHE__ = new Map());
    cache.set(`${MODEL_URL}|${FEATURES_URL}`, { model, features });

    const savedFetch = globalThis.fetch;
    globalThis.fetch = async () => { throw new Error('network must not be hit when cache is seeded'); };
    try {
      const route = loadRoute(PREDICT_SALE_ROUTE);
      const valid = {
        bedrooms: 2, bathrooms: 2, size_sqft: 1000, postcode: 'SW1X 8NX',
        property_type: 'flat', address: 'Chester Square, London, SW1X', latitude: 51.4946, longitude: -0.153,
      };
      const res = await route.POST(postRequest('http://localhost/api/predict-sale', JSON.stringify(valid)));
      check('predict-sale seeded-cache success -> 200', res.status === 200, `${res.status}`);
      const body = await res.json();
      check('predict-sale 200 returns finite positive predicted_price',
        Number.isFinite(body.predicted_price) && body.predicted_price > 0, `${body.predicted_price}`);
      check('predict-sale 200 model_version sale_v1', body.model_version === 'sale_v1', `${body.model_version}`);
      check('predict-sale 200 range_low === round(price * 0.85)',
        body.range_low === Math.round(body.predicted_price * 0.85), `${body.range_low}`);
      check('predict-sale 200 range_high === round(price * 1.15)',
        body.range_high === Math.round(body.predicted_price * 1.15), `${body.range_high}`);
      check('predict-sale 200 district extracted (SW1X)', body.district === 'SW1X', `${body.district}`);
      check('predict-sale 200 not low_confidence (real postcode + size)', body.low_confidence === false);
      check('predict-sale 200 estimated_size false (size provided)', body.estimated_size === false);

      // UX flags: missing postcode -> UNKNOWN district + low_confidence true (still 200).
      const noPc = await route.POST(postRequest('http://localhost/api/predict-sale',
        JSON.stringify({ ...valid, postcode: '' })));
      check('predict-sale missing postcode still 200', noPc.status === 200, `${noPc.status}`);
      const noPcBody = await noPc.json();
      check('predict-sale missing postcode -> district UNKNOWN', noPcBody.district === 'UNKNOWN', `${noPcBody.district}`);
      check('predict-sale missing postcode -> low_confidence true', noPcBody.low_confidence === true);
    } finally {
      globalThis.fetch = savedFetch;
    }
  }
}

async function testSimilarSaleRoute() {
  console.log('\n--- /api/similar-sale (real handler) ---');

  let dbBehavior = () => ({ peers: [], stats: { peer_count: 0, your_percentile: 50 } });
  const dbStub = { getSimilarSaleListings: async (p) => dbBehavior(p) };
  const route = loadRoute(SIMILAR_SALE_ROUTE, { '@/lib/saleDb': dbStub });

  // OPTIONS -> CORS
  {
    const res = await route.OPTIONS();
    check('similar-sale OPTIONS -> 200', res.status === 200, `${res.status}`);
    check('similar-sale OPTIONS sets Allow-Origin *', res.headers.get('access-control-allow-origin') === '*');
    check('similar-sale OPTIONS Allow-Methods includes GET',
      (res.headers.get('access-control-allow-methods') || '').includes('GET'),
      res.headers.get('access-control-allow-methods'));
  }

  // 400 missing params + invalid beds/price
  {
    const r0 = await route.GET(getRequest('http://localhost/api/similar-sale'));
    check('similar-sale missing params -> 400', r0.status === 400, `${r0.status}`);
    const r1 = await route.GET(getRequest('http://localhost/api/similar-sale?postcode=SW3&beds=abc&price=3000000'));
    check('similar-sale invalid beds -> 400', r1.status === 400, `${r1.status}`);
    const r2 = await route.GET(getRequest('http://localhost/api/similar-sale?postcode=SW3&beds=2&price=0'));
    check('similar-sale non-positive price -> 400', r2.status === 400, `${r2.status}`);
  }

  // 500-vs-empty distinction
  {
    dbBehavior = () => { throw new Error('connection refused'); };
    const res = await route.GET(getRequest('http://localhost/api/similar-sale?postcode=SW3%204AJ&beds=2&price=3000000'));
    check('similar-sale genuine DB error -> 500', res.status === 500, `${res.status}`);
    check('similar-sale 500 carries CORS', res.headers.get('access-control-allow-origin') === '*');
  }
  {
    // The route comment notes a MISSING sale table degrades to empty in saleDb (graceful-
    // empty) — here we model the post-degrade empty result reaching the route: 200, not 500.
    dbBehavior = () => ({ peers: [], stats: { peer_count: 0, your_percentile: 50, avg_ppsf: null } });
    const res = await route.GET(getRequest('http://localhost/api/similar-sale?postcode=SW3%204AJ&beds=2&price=3000000'));
    check('similar-sale empty/graceful result -> 200 (not 500)', res.status === 200, `${res.status}`);
    const body = await res.json();
    check('similar-sale empty -> peer_count 0', body.stats?.peer_count === 0, JSON.stringify(body.stats));
    check('similar-sale 200 attaches query_ms', typeof body.query_ms === 'number', `${body.query_ms}`);
  }
}

async function main() {
  await testPredictSaleRoute();
  await testSimilarSaleRoute();

  if (failures) {
    console.log(`\n=== FAIL: ${failures} sale route-handler check(s) failed. ===`);
    process.exit(1);
  }
  console.log('\n=== PASS: /api/predict-sale + /api/similar-sale real handlers return the documented HTTP contract. ===');
}

main().catch((e) => {
  console.error('route_handler_sale_test crashed:', e);
  process.exit(1);
});
