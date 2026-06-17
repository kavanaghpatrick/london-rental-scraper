/**
 * Behavioral + static test for content.js findSimilarProperties() — the
 * "Similar Properties" comp picker. Guards the location HARD GATE fix
 * (see SIMILAR_PROPERTIES_DEBUG.md).
 *
 * THE BUG this locks down: location used to be a SOFT bonus (+30/+15/+0) and the
 * include gate was `score >= 30`, so a listing within 10% of fair value (40 pts)
 * cleared the gate with ZERO location match. Result: comps from the wrong area
 * (the "totally off" symptom), worst in thin districts. The fix makes location a
 * hard gate: tier0 = same district, tier1 = same area-prefix, everything else
 * (including empty/garbage district) DISCARDED; tier0 always outranks tier1.
 *
 * This test runs the REAL findSimilarProperties from content.js (loaded in a
 * browser shim, same approach as _content_load_test.mjs) against:
 *   1. a synthetic fixture (target + a mix of in-district, in-area, and
 *      far-district same-price listings) — assert NO returned comp is outside
 *      the area gate, and a same-price wrong-area listing never appears.
 *   2. the REAL committed api/similar_listings.json — assert every returned comp
 *      for several targets shares the target's area, the SW10 repro returns
 *      same-district comps, and a thin district degrades to fewer (never wrong).
 *
 * Run: node _similar_properties_test.mjs   (exit 0 = pass, 1 = fail)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(__dirname, 'content.js'), 'utf8');
const REAL = JSON.parse(
  readFileSync(join(__dirname, 'api', 'similar_listings.json'), 'utf8')
);

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) {
    console.log(`OK   ${name}`);
  } else {
    failures++;
    console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`);
  }
}

const areaOf = d => (d || '').match(/^([A-Z]+)/)?.[1] || '';

// --- Load content.js and capture findSimilarProperties out of the IIFE --------
// content.js is an IIFE; findSimilarProperties is a closure. We expose it by
// running the source in a shim and letting the script hang a reference onto a
// global we control. The script already assigns helpers onto `window`, so we
// append a tiny export shim after the source that publishes the function.
function loadFindSimilar(listingsForFetch) {
  const exported = {};
  const windowShim = {
    location: { hostname: 'www.rightmove.co.uk', pathname: '/p/1', href: 'https://www.rightmove.co.uk/properties/1' },
    addEventListener: () => {},
    __rentFairValueLoaded: undefined,
  };
  windowShim.window = windowShim;
  const documentShim = {
    title: '', body: { appendChild() {}, },
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => ({ textContent: '', innerHTML: '', set textContent(v) {}, }),
    addEventListener: () => {},
  };
  const sandbox = {
    window: windowShim,
    document: documentShim,
    history: { pushState() {}, replaceState() {} },
    location: windowShim.location,
    setInterval: () => 0,
    clearInterval: () => {},
    setTimeout: (fn) => 0,
    clearTimeout: () => {},
    MutationObserver: class { observe() {} disconnect() {} },
    // fetch returns the listings dict we want findSimilarProperties to score.
    fetch: async () => ({ ok: true, status: 200, json: async () => listingsForFetch }),
    Tesseract: undefined,
    chrome: { runtime: { sendMessage: () => {}, getURL: () => '', lastError: null } },
    console: { log() {}, warn() {}, error() {} }, // silence the script's own logs
    __export: exported,
  };
  // Inject an export trampoline INSIDE the IIFE (right before its closing
  // `})();`), where findSimilarProperties is in scope. Appending after the
  // source would run outside the IIFE, where the closure isn't visible.
  const EXPORT_SHIM = '\n;try{__export.findSimilarProperties=findSimilarProperties;}catch(e){__export.err=e;}\n';
  const closeIdx = SRC.lastIndexOf('})();');
  if (closeIdx === -1) throw new Error('could not find IIFE close `})();` in content.js');
  const instrumented = SRC.slice(0, closeIdx) + EXPORT_SHIM + SRC.slice(closeIdx);
  const names = Object.keys(sandbox);
  const vals = names.map(n => sandbox[n]);
  // eslint-disable-next-line no-new-func
  const fn = new Function(...names, `'use strict';\n${instrumented}`);
  fn(...vals);
  if (!exported.findSimilarProperties) {
    throw new Error('could not capture findSimilarProperties: ' + (exported.err || 'not in scope'));
  }
  return exported.findSimilarProperties;
}

// =============================================================================
// 1. SYNTHETIC FIXTURE — the core invariant
// =============================================================================
// Target: SW10, fairValue 4000, 2 bed. Mix of comps, several at the SAME price
// (4000) but in the WRONG area — these MUST NOT be returned.
const fixture = {
  'rm:in_district_a':  { u: 'x1', a: 'In District A', p: 'SW10', b: 2, ba: 1, pr: 4000, s: 800 },
  'rm:in_district_b':  { u: 'x2', a: 'In District B', p: 'SW10', b: 2, ba: 2, pr: 4200, s: 850 },
  'rm:in_district_c':  { u: 'x3', a: 'In District C', p: 'SW10', b: 1, ba: 1, pr: 3800, s: 600 },
  'rm:in_area_sw3':    { u: 'x4', a: 'Same Area SW3', p: 'SW3',  b: 2, ba: 1, pr: 4000, s: 820 },
  'rm:in_area_sw7':    { u: 'x5', a: 'Same Area SW7', p: 'SW7',  b: 2, ba: 1, pr: 4100, s: 810 },
  // Wrong-area, IDENTICAL price (the classic price-only false positive):
  'rm:wrong_nw1':      { u: 'x6', a: 'Wrong Area NW1', p: 'NW1', b: 2, ba: 1, pr: 4000, s: 800 },
  'rm:wrong_e14':      { u: 'x7', a: 'Wrong Area E14', p: 'E14', b: 2, ba: 1, pr: 4000, s: 800 },
  'rm:wrong_w8':       { u: 'x8', a: 'Wrong Area W8',  p: 'W8',  b: 2, ba: 1, pr: 4000, s: 800 },
  // Empty / garbage district at the same price — must never gate-pass:
  'rm:empty_p':        { u: 'x9', a: 'Empty district', p: '',    b: 2, ba: 1, pr: 4000, s: 800 },
  'rm:no_p':           { u: 'x10', a: 'Missing district', b: 2, ba: 1, pr: 4000, s: 800 },
};

const findSimilarFixture = loadFindSimilar(fixture);
const fixResult = await findSimilarFixture(4000, 'SW10', 2, 1, [], 3);

check('fixture: returns at most the limit (3)', fixResult.length <= 3, `got ${fixResult.length}`);
check('fixture: returns at least 1 comp (SW10 is well populated)', fixResult.length >= 1);
check('fixture: NO returned comp is outside the SW area gate',
  fixResult.every(c => areaOf(c.postcode) === 'SW'),
  'returned: ' + fixResult.map(c => `${c.address}(${c.postcode})`).join(', '));
check('fixture: same-price wrong-area comps (NW1/E14/W8) are NEVER returned',
  fixResult.every(c => !['NW1', 'E14', 'W8'].includes(c.postcode)),
  'returned: ' + fixResult.map(c => c.postcode).join(', '));
check('fixture: empty/missing-district comps are NEVER returned',
  fixResult.every(c => c.postcode && c.postcode.length > 0));
check('fixture: same-district (SW10) comps fill the top before same-area (SW3/SW7)',
  // We have 3 SW10 comps within price band, so all 3 results must be SW10.
  fixResult.length === 3 && fixResult.every(c => c.postcode === 'SW10'),
  'returned: ' + fixResult.map(c => c.postcode).join(', '));

// Thin-district case: target a district with only ONE in-district comp, but
// several same-price same-area comps. Should backfill from area, never wrong-area.
const thinFix = {
  'rm:only_one':   { u: 'y1', a: 'Only district comp', p: 'SW10', b: 2, ba: 1, pr: 4000 },
  'rm:area_sw3':   { u: 'y2', a: 'Area SW3', p: 'SW3', b: 2, ba: 1, pr: 4000 },
  'rm:area_sw7':   { u: 'y3', a: 'Area SW7', p: 'SW7', b: 2, ba: 1, pr: 4000 },
  'rm:wrong_nw1':  { u: 'y4', a: 'Wrong NW1', p: 'NW1', b: 2, ba: 1, pr: 4000 },
};
const findThin = loadFindSimilar(thinFix);
const thinResult = await findThin(4000, 'SW10', 2, 1, [], 3);
check('thin: same-district comp ranks FIRST (above same-area)',
  thinResult.length >= 1 && thinResult[0].postcode === 'SW10',
  'top: ' + (thinResult[0] && thinResult[0].postcode));
check('thin: backfills with same-area, still no wrong-area comp',
  thinResult.every(c => areaOf(c.postcode) === 'SW'),
  thinResult.map(c => c.postcode).join(', '));

// Truly empty district result: a district with no district/area comps at all.
const emptyFix = {
  'rm:far': { u: 'z1', a: 'Far', p: 'M1', b: 2, ba: 1, pr: 4000 }, // Manchester area
};
const findEmpty = loadFindSimilar(emptyFix);
const emptyResult = await findEmpty(4000, 'SW10', 2, 1, [], 3);
check('no-comps: returns EMPTY rather than a wrong-area comp',
  emptyResult.length === 0,
  'returned: ' + emptyResult.map(c => c.postcode).join(', '));

// =============================================================================
// 2. REAL DATA — committed api/similar_listings.json
// =============================================================================
const findSimilarReal = loadFindSimilar(REAL);

// Helper: median price among real listings of a given district (a realistic
// fairValue so the price band actually admits comps).
function medianPriceForDistrict(d) {
  const prices = Object.values(REAL).filter(v => v.p === d && v.pr).map(v => v.pr).sort((a, b) => a - b);
  return prices.length ? prices[Math.floor(prices.length / 2)] : 4000;
}

for (const district of ['SW10', 'SW3', 'NW8', 'W8']) {
  const fv = medianPriceForDistrict(district);
  const beds = 2;
  const res = await findSimilarReal(fv, district, beds, 1, [], 3);
  const area = areaOf(district);
  check(`real ${district}: returns comps`, res.length >= 1, `got ${res.length} (fv ${fv})`);
  check(`real ${district}: EVERY returned comp is in area ${area}`,
    res.every(c => areaOf(c.postcode) === area),
    'returned: ' + res.map(c => `${c.postcode}`).join(', '));
  check(`real ${district}: top comp is same-district (${district}) when district has comps`,
    res.length === 0 || res[0].postcode === district,
    'top: ' + (res[0] && res[0].postcode));
}

// Repro from SIMILAR_PROPERTIES_DEBUG.md: SW10 must return SW10 comps.
const sw10 = await findSimilarReal(medianPriceForDistrict('SW10'), 'SW10', 2, 1, [], 3);
check('real SW10 repro: at least one SW10 same-district comp',
  sw10.some(c => c.postcode === 'SW10'),
  'returned: ' + sw10.map(c => c.postcode).join(', '));

// Thin / absent district in real data: a district with 0 comps must NOT spill
// into a wrong area (returns [] or area-only, never another area).
const thinReal = await findSimilarReal(4000, 'SE99', 2, 1, [], 3); // SE area, no comps
check('real thin (SE99): no comp outside SE area',
  thinReal.every(c => areaOf(c.postcode) === 'SE'),
  'returned: ' + thinReal.map(c => c.postcode).join(', '));

// =============================================================================
// 3. STATIC GUARDS — the hard-gate structure must stay in the source
// =============================================================================
check('content.js: old global `if (score >= 30)` include gate is GONE',
  !/if\s*\(\s*score\s*>=\s*30\s*\)/.test(SRC),
  'the score>=30 include threshold must be replaced by the location gate');
check('content.js: tiered buckets present (tier0/tier1)',
  /tier0/.test(SRC) && /tier1/.test(SRC));
check('content.js: empty listingDistrict cannot pass the area gate (guarded startsWith)',
  /listingDistrict\s*&&\s*listingDistrict\.startsWith\(/.test(SRC) ||
  /targetArea\s*&&\s*listingDistrict\s*&&\s*listingDistrict\.startsWith\(/.test(SRC),
  'startsWith must be guarded by a truthy listingDistrict so empty p never matches');
check('content.js: result is tier0.concat(tier1) (same-district first)',
  /tier0\.concat\(tier1\)/.test(SRC));

console.log('');
if (failures === 0) {
  console.log('=== PASS: similar-properties location hard-gate green (fixture + real data + static) ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) failed ===`);
  process.exit(1);
}
