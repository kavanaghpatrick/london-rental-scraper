/**
 * WS — FOXTONS extractor REGRESSION test against a REAL captured page.
 *
 * Foxtons detail pages are server-rendered Next.js with a clean single-property blob
 * at <script id="__NEXT_DATA__"> -> props.pageProps.propertyDetail.
 * extractPropertyDataFoxtons() (content.js) reads that and BUILDS the canonical
 * propertyData shape the shared helpers consume. This extractor already works; this
 * test LOCKS IT DOWN so the WS1 Rightmove __PAGE_MODEL change (which touches the same
 * file + the shared __NEXT_DATA__ / sizings / price paths) can't silently regress it.
 *
 * Fixture: tests/fixtures/extension/foxtons_chpk0327321.html (REAL page). Ground truth
 * decoded from its own __NEXT_DATA__ propertyDetail + cross-checked vs output/rentals.db:
 *   bedrooms=2 bathrooms=2 receptions=1 ; pricePcm=15999 (pcm, NOT the £3,692 pw) ;
 *   floorArea=1213 (already sqft) ; postcode "SW1X 8AF" ; type=flat ;
 *   floorplan on web-prod-page-assets.foxtons.co.uk.
 *
 * EXPECTED STATE: fully GREEN now and after the Rightmove fix (pure regression guard).
 *
 * Run: node chrome-extension/foxtons_extract_test.mjs   (exit 0 = pass)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadExtractors } from './extract_test_shim.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK   ${name}`);
  else { failures++; console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`); }
}

const html = readFileSync(
  join(__dirname, '..', 'tests', 'fixtures', 'extension', 'foxtons_chpk0327321.html'), 'utf8');
const ex = loadExtractors(html, {
  hostname: 'www.foxtons.co.uk',
  pathname: '/properties-to-rent/SW1X/chpk0327321',
});
const data = ex.extractPropertyDataFoxtons();

check('Foxtons: extractor returns data (NOT null) from __NEXT_DATA__',
  data && typeof data === 'object', `got ${data}`);

if (data) {
  check('Foxtons: bedrooms === 2', data.bedrooms === 2, `got ${data.bedrooms}`);
  check('Foxtons: bathrooms === 2', data.bathrooms === 2, `got ${data.bathrooms}`);
  check('Foxtons: receptions === 1', data.receptions === 1, `got ${data.receptions}`);
  check('Foxtons: extractSqftFromPage === 1213 (floorArea is already sqft)',
    ex.extractSqftFromPage(data) === 1213, `got ${ex.extractSqftFromPage(data)}`);
  // pricePcm (15999) is the per-calendar-month figure; must NOT use the £3,692 weekly.
  check('Foxtons: parsePrice(primaryPrice) === 15999 (pcm, NOT £3,692 pw)',
    ex.parsePrice(data.prices && data.prices.primaryPrice) === 15999,
    `raw="${data.prices && data.prices.primaryPrice}" → ${ex.parsePrice(data.prices && data.prices.primaryPrice)}`);
  check("Foxtons: extractPostcode === 'SW1X 8AF'",
    ex.extractPostcode(data) === 'SW1X 8AF', `got ${ex.extractPostcode(data)}`);
  check("Foxtons: extractPropertyType === 'flat'",
    ex.extractPropertyType(data) === 'flat', `got ${ex.extractPropertyType(data)}`);
  check('Foxtons: getFloorplanUrl on web-prod-page-assets.foxtons.co.uk',
    String(ex.getFloorplanUrl(data) || '').includes('web-prod-page-assets.foxtons.co.uk'),
    `got ${ex.getFloorplanUrl(data)}`);
}

console.log('');
if (failures === 0) {
  console.log('=== PASS: Foxtons __NEXT_DATA__ extractor green (regression guard) ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) failed ===`);
  process.exit(1);
}
