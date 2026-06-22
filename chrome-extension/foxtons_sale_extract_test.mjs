/**
 * INC4 — FOXTONS for-SALE detection + extraction test (TDD).
 *
 * Spec: /tmp/inc4_extension_fix.md  §"Foxtons (BROKEN — extractor rental-shaped, bails
 * before fork)".
 *
 * Fixture: tests/fixtures/for_sale/extension/foxtons_for_sale_detail.json
 *   REAL captured __NEXT_DATA__ propertyDetail from
 *   https://www.foxtons.co.uk/properties-for-sale/sw7/chpk2514513 (HTTP 200, 2026-06-22).
 *   instructionType='sale', priceFrom=5150000 LUMP SUM, pricePcm=null, beds 5, baths 4,
 *   receptions 3, floorArea 3401 sqft, postcode SW7 3RE, type house_terraced, priceAskingType='ASPR'.
 *   The raw JSON is wrapped in a minimal HTML shell <script id="__NEXT_DATA__">…</script>
 *   so extractPropertyDataFoxtons reads it via the SAME getElementById path as the rental test.
 *
 * THE BUGS this targets (RED pre-fix):
 *   - extractPropertyDataFoxtons (content.js ~1244-1250) reads pricePcm then, when it's
 *     null (as on EVERY sale page), labels priceFrom as 'pw' → "£5,150,000 pw" →
 *     parsePrice does ×52/12 → 22,316,667 (a NONSENSE figure). For SALE it must surface
 *     a BARE LUMP SUM ('£5,150,000', no pw/pcm) and coerce Number(String(priceFrom)).
 *   - detectTenure already returns 'sale' for Foxtons via the /properties-for-sale/ URL
 *     regex (asserted GREEN here so the rent-vs-sale split is locked), but instructionType
 *     ='sale' is the belt-and-braces data marker the fix should also honour.
 *
 * EXPECTED STATE NOW: the detection + field checks PASS; the LUMP-SUM price check FAILS
 * RED (current extractor emits "£5,150,000 pw" → parsePrice 22,316,667).
 *
 * Run: node chrome-extension/foxtons_sale_extract_test.mjs   (exit 0 = pass)
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

// Load the real captured __NEXT_DATA__ propertyDetail and wrap it in the HTML shell the
// extractor expects (getElementById('__NEXT_DATA__')). Strip the fixture's _fixture_note.
const fixture = JSON.parse(
  readFileSync(join(ROOT, 'tests/fixtures/for_sale/extension/foxtons_for_sale_detail.json'), 'utf8'));
const nextData = { props: fixture.props };
const html = `<!DOCTYPE html><html><head><title>For sale (Foxtons fixture)</title></head><body>` +
  `<script id="__NEXT_DATA__" type="application/json">${JSON.stringify(nextData)}</script>` +
  `</body></html>`;

const PATH = '/properties-for-sale/sw7/chpk2514513';
const URL = 'https://www.foxtons.co.uk' + PATH;
const ex = loadExtractors(html, { hostname: 'www.foxtons.co.uk', pathname: PATH, href: URL });

check('shim sanity: site detected as foxtons', ex.currentSite === 'foxtons', `got ${ex.currentSite}`);

// Ground-truth marker straight off the captured blob (not via the extractor).
check("Foxtons-sale: fixture instructionType==='sale' (rentals carry 'letting')",
  fixture.props.pageProps.propertyDetail.instructionType === 'sale',
  `got ${fixture.props.pageProps.propertyDetail.instructionType}`);

const data = ex.extractPropertyDataFoxtons();
check('Foxtons-sale: extractor returns data (NOT null) from __NEXT_DATA__',
  data && typeof data === 'object', `got ${data}`);

if (data) {
  // --- 1. SALE DETECTION (URL-based, already works — locked as GREEN) ---------------
  check("Foxtons-sale: detectTenure → 'sale' for /properties-for-sale/ URL",
    ex.detectTenure(URL, data) === 'sale', `got ${ex.detectTenure(URL, data)}`);

  // --- 2. MODEL-INPUT FIELDS resolve -----------------------------------------------
  check('Foxtons-sale: bedrooms === 5', data.bedrooms === 5, `got ${data.bedrooms}`);
  check('Foxtons-sale: bathrooms === 4', data.bathrooms === 4, `got ${data.bathrooms}`);
  check('Foxtons-sale: receptions === 3', data.receptions === 3, `got ${data.receptions}`);
  check('Foxtons-sale: extractSqftFromPage === 3401 (floorArea is already sqft)',
    ex.extractSqftFromPage(data) === 3401, `got ${ex.extractSqftFromPage(data)}`);
  check("Foxtons-sale: extractPostcode === 'SW7 3RE'", ex.extractPostcode(data) === 'SW7 3RE',
    `got ${ex.extractPostcode(data)}`);
  check("Foxtons-sale: extractPropertyType resolves (house*)",
    /house/.test(String(ex.extractPropertyType(data))), `got ${ex.extractPropertyType(data)}`);

  // --- 3. PRICE: must be the LUMP SUM 5150000, NOT a 'pw' ×52/12 mangle -------------
  check("Foxtons-sale: primaryPrice is a bare lump sum (no 'pw'/'pcm' token)",
    data.prices && !/pw|pcm/i.test(String(data.prices.primaryPrice)),
    `got "${data.prices && data.prices.primaryPrice}" — current extractor labels the sale lump sum as 'pw'`);
  check('Foxtons-sale: parsePrice(primaryPrice) === 5150000 (lump sum, NOT 22,316,667)',
    ex.parsePrice(data.prices && data.prices.primaryPrice) === 5150000,
    `got ${ex.parsePrice(data.prices && data.prices.primaryPrice)} from "${data.prices && data.prices.primaryPrice}"`);
}

console.log('');
if (failures === 0) {
  console.log('=== PASS: Foxtons for-sale detection + extraction green (real fixture) ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) RED (sale extractor mislabels the lump sum as 'pw') ===`);
  process.exit(1);
}
