/**
 * WS1 — RIGHTMOVE extractor test (TDD, RED before the __PAGE_MODEL fix).
 *
 * THE BUG this locks down: Rightmove listings NO LONGER ship
 * <script id="__NEXT_DATA__">. The inline property data moved to
 *   window.__PAGE_MODEL = {"data":"<escaped JSON flat array>","encoding":"on"}
 * (DOUBLE underscore), a devalue/flatten reference-indexed array (arr[0] maps top
 * keys → indices; a value is a reference only when it's an int index pointing at a
 * CONTAINER — scalar leaves like bedrooms=2 are literal). The current
 * extractPropertyDataRightmove() only tries (1) __NEXT_DATA__ (dead) and
 * (2) /window\.PAGE_MODEL/ (SINGLE underscore — misses the double-underscore
 * global), so extraction fails and no popup renders.
 *
 * This test runs the REAL extractPropertyDataRightmove + the shared normalization
 * helpers (extractSqftFromPage / extractPostcode / parsePrice / getFloorplanUrl /
 * extractPropertyType / extractLetType) out of content.js, via the shared zero-dep
 * DOM shim, against REAL captured live Rightmove pages under tests/fixtures/extension/:
 *
 *   rightmove_169944029.html  HAS sqft   (sizings carries ha+sqft+sqm+ac → pick sqft)
 *   rightmove_163282418.html  NO  sqft   (empty sizings → extractSqftFromPage null)
 *
 * Ground truth was decoded from the fixtures' own __PAGE_MODEL blob.
 *
 * It also guards BACKWARD COMPAT: a synthetic <script id="__NEXT_DATA__"> page
 * still extracts (Strategy 1 order preserved), and a search/non-property page with
 * no __PAGE_MODEL returns null (no popup).
 *
 * Run: node chrome-extension/rightmove_extract_test.mjs   (exit 0 = pass)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadExtractors } from './extract_test_shim.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK   ${name}`);
  else { failures++; console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`); }
}

const RM = (file, pathname) => loadExtractors(
  readFileSync(join(ROOT, 'tests/fixtures/extension', file), 'utf8'),
  { hostname: 'www.rightmove.co.uk', pathname });

// =============================================================================
// 1. REAL FIXTURE — 169944029 (HAS sqft, mixed-unit sizings)
// =============================================================================
{
  const ex = RM('rightmove_169944029.html', '/properties/169944029');
  const pd = ex.extractPropertyDataRightmove();

  check('169944029: extractor returns property data (NOT null) from __PAGE_MODEL',
    pd && typeof pd === 'object', `got ${pd}`);

  if (pd) {
    check('169944029: bedrooms === 2', pd.bedrooms === 2, `got ${pd.bedrooms}`);
    check('169944029: bathrooms === 2', pd.bathrooms === 2, `got ${pd.bathrooms}`);
    // sizings has ha (0.01) + sqft (1539) + sqm (143) + ac (0.04); MUST pick sqft.
    check('169944029: extractSqftFromPage === 1539 (sqft chosen over ha/sqm/ac)',
      ex.extractSqftFromPage(pd) === 1539, `got ${ex.extractSqftFromPage(pd)}`);
    check('169944029: parsePrice(primaryPrice) === 19000 (pcm)',
      ex.parsePrice(pd.prices && pd.prices.primaryPrice) === 19000,
      `price=${pd.prices && pd.prices.primaryPrice} → ${ex.parsePrice(pd.prices && pd.prices.primaryPrice)}`);
    check("169944029: extractPostcode === 'SW1W 8BQ'",
      ex.extractPostcode(pd) === 'SW1W 8BQ', `got ${ex.extractPostcode(pd)}`);
    check("169944029: extractPropertyType === 'apartment'",
      ex.extractPropertyType(pd) === 'apartment', `got ${ex.extractPropertyType(pd)}`);
    check("169944029: getFloorplanUrl contains '/property-floorplan/'",
      String(ex.getFloorplanUrl(pd) || '').includes('/property-floorplan/'),
      `got ${ex.getFloorplanUrl(pd)}`);
    check("169944029: extractLetType === 'long' (letType 'Long term')",
      ex.extractLetType(pd) === 'long', `got ${ex.extractLetType(pd)}`);
    check('169944029: location lat/lng decoded as scalars (51.489118 / -0.153977)',
      pd.location && Math.abs(pd.location.latitude - 51.489118) < 1e-4 &&
      Math.abs(pd.location.longitude - (-0.153977)) < 1e-4,
      `got ${pd.location && JSON.stringify(pd.location)}`);
  }
}

// =============================================================================
// 2. REAL FIXTURE — 163282418 (NO sqft: empty sizings)
// =============================================================================
{
  const ex = RM('rightmove_163282418.html', '/properties/163282418');
  const pd = ex.extractPropertyDataRightmove();

  check('163282418: extractor returns property data (NOT null) from __PAGE_MODEL',
    pd && typeof pd === 'object', `got ${pd}`);

  if (pd) {
    check('163282418: bedrooms === 3', pd.bedrooms === 3, `got ${pd.bedrooms}`);
    check('163282418: bathrooms === 3', pd.bathrooms === 3, `got ${pd.bathrooms}`);
    check('163282418: extractSqftFromPage === null (empty sizings)',
      ex.extractSqftFromPage(pd) === null, `got ${ex.extractSqftFromPage(pd)}`);
    check('163282418: parsePrice(primaryPrice) === 20583 (pcm)',
      ex.parsePrice(pd.prices && pd.prices.primaryPrice) === 20583,
      `price=${pd.prices && pd.prices.primaryPrice}`);
    check("163282418: extractPostcode === 'SW1W 9AA'",
      ex.extractPostcode(pd) === 'SW1W 9AA', `got ${ex.extractPostcode(pd)}`);
    check("163282418: getFloorplanUrl contains '/property-floorplan/'",
      String(ex.getFloorplanUrl(pd) || '').includes('/property-floorplan/'),
      `got ${ex.getFloorplanUrl(pd)}`);
  }
}

// =============================================================================
// 3. BACKWARD COMPAT — Strategy 1 (__NEXT_DATA__) still wins when present
// =============================================================================
{
  const nextData = {
    props: { pageProps: { propertyData: {
      bedrooms: 1, bathrooms: 1,
      prices: { primaryPrice: '£2,500 pcm' },
      address: { outcode: 'E1', incode: '6AN', displayAddress: 'A Street, E1 6AN' },
      sizings: [{ unit: 'sqft', minimumSize: 500 }],
      propertySubType: 'Flat',
    } } },
  };
  const html = `<!DOCTYPE html><html><head><title>x</title></head><body>` +
    `<script id="__NEXT_DATA__" type="application/json">${JSON.stringify(nextData)}</script>` +
    `</body></html>`;
  const ex = loadExtractors(html, { hostname: 'www.rightmove.co.uk', pathname: '/properties/1' });
  const pd = ex.extractPropertyDataRightmove();
  check('backward-compat: __NEXT_DATA__ page still extracts (Strategy 1 preserved)',
    pd && pd.bedrooms === 1 && ex.extractSqftFromPage(pd) === 500,
    `got ${pd && JSON.stringify({ b: pd.bedrooms, s: ex.extractSqftFromPage(pd) })}`);
}

// =============================================================================
// 4. NEGATIVE — a search/non-property page (no __PAGE_MODEL) returns null
// =============================================================================
{
  const html = `<!DOCTYPE html><html><head><title>Property search</title></head>` +
    `<body><script>window.__SEARCH_MODEL = {results:[]};</script></body></html>`;
  const ex = loadExtractors(html, { hostname: 'www.rightmove.co.uk', pathname: '/property-to-rent/find.html' });
  const pd = ex.extractPropertyDataRightmove();
  check('negative: search page with no __PAGE_MODEL returns null (no popup)',
    pd === null, `got ${pd}`);
}

console.log('');
if (failures === 0) {
  console.log('=== PASS: Rightmove __PAGE_MODEL extractor green (real fixtures + back-compat + negative) ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) failed (RED until extractPropertyDataRightmove reads window.__PAGE_MODEL) ===`);
  process.exit(1);
}
