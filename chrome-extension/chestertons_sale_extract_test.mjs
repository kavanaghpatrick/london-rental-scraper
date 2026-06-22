/**
 * INC4 — CHESTERTONS for-SALE detection + extraction test (TDD).
 *
 * Spec: /tmp/inc4_extension_fix.md  §"Chestertons (BROKEN → the user's live repro FUL250188)".
 * The user's live repro: https://www.chestertons.co.uk/properties/21855578/sales/FUL250188
 *
 * Fixture: tests/fixtures/for_sale/extension/chestertons_for_sale_detail.html
 *   REAL captured (NOT derived) RSC __next_f flight chunk for the SUBJECT for-sale object,
 *   single-level \" escaping byte-faithful so content.js's unescape + balanced-brace scan
 *   parse it. Subject: primaryChannel="sales", priceValue=2150000 LUMP SUM, beds 5, baths 4,
 *   postcode "W6 8LY", squareFeetInternal=2390, floorplans[0] homeflow-assets, status "For sale".
 *
 * THE BUGS this targets (RED pre-fix) — three /lettings/-only regexes that return null/false
 * on a /sales/ URL so the popup never renders / subject never resolves:
 *   - isPropertyPage Chestertons branch (~content.js:173)  -> gates scheduleSpaRetry
 *   - extractChestertonsSubjectFromFlight() id anchor (~content.js:1484) -> gates subject extraction
 *   - extractPropertyId() Chestertons branch (~content.js:1998)
 * The fix mirrors the spider seam: /properties/\d+/(?:sales|lettings)/.
 *
 * The three "regex N matches /sales/" checks pull the LITERAL out of the LIVE content.js
 * source (anchored by the unique surrounding code) and eval it against the /sales/ URL, so
 * they test the SHIPPED regex — they go GREEN the moment the source adds (?:sales|lettings)
 * and never pass on a copy. The behavioural checks (extractPropertyId resolves; subject
 * extraction yields beds/baths/postcode/sqft/floorplan) exercise the same three regexes
 * end-to-end through the real extractor.
 *
 * EXPECTED STATE NOW:
 *   - detectTenure → 'sale' for the /sales/ URL ALREADY works (content.js:626) → asserted GREEN.
 *   - The 3 regex checks FAIL RED (current literals are /lettings/-only).
 *   - extractPropertyId() returns null on /sales/, and extractPropertyDataChestertons()
 *     returns NULL on the /sales/ fixture → every field check FAILS RED.
 *
 * Run: node chrome-extension/chestertons_sale_extract_test.mjs   (exit 0 = pass)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadExtractors } from './extract_test_shim.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const SRC = readFileSync(join(__dirname, 'content.js'), 'utf8');

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK   ${name}`);
  else { failures++; console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`); }
}

const SALES_URL_PATH = '/properties/21855578/sales/FUL250188';
const URL = 'https://www.chestertons.co.uk' + SALES_URL_PATH;

// Pull the FIRST /…/ regex literal out of the ONE source line the predicate selects and
// eval it. Returns the live RegExp, or null when the line/literal can't be located
// (itself reported as a check failure so a refactor can't silently hide the assertion).
function liveRegexFromLine(label, linePred) {
  const lines = SRC.split('\n').filter(linePred);
  if (lines.length === 0) { check(`${label}: source line located`, false, `no line matched in content.js`); return null; }
  if (lines.length > 1) { check(`${label}: anchor is unique`, false, `${lines.length} lines matched — tighten anchor`); return null; }
  const litMatch = lines[0].match(/\/(?:\\.|[^/\n])+\/[gimsuy]*/); // a /…/ regex literal
  if (!litMatch) { check(`${label}: regex literal parsed from line`, false, `no /…/ literal in: ${lines[0].trim()}`); return null; }
  try { return eval(litMatch[0]); } catch (e) { check(`${label}: regex literal evaluable`, false, String(e)); return null; }
}

// =============================================================================
// A. THE THREE /lettings/-only REGEXES must accept /sales/ too.
// =============================================================================
// Anchors are chosen to hit EXACTLY the one shipped line each (the only Chestertons
// branch with .test(url); the only `idMatch …pathname.match(`; the only
// `const match = pathname.match(` containing "lettings"). A copy elsewhere can't satisfy
// them, and a refactor that drops the line trips the "source line located" failure.
{
  const re = liveRegexFromLine('isPropertyPage',
    (l) => /\.test\(url\)/.test(l) && /lettings|sales/.test(l) && /properties/.test(l));
  if (re) check('Chestertons regex #1 (isPropertyPage, gates scheduleSpaRetry) matches a /sales/ URL',
    re.test(SALES_URL_PATH), `regex=${re} — currently /lettings/-only; want /properties/\\d+/(?:sales|lettings)/`);
}
{
  const re = liveRegexFromLine('flight-anchor id',
    (l) => /const idMatch = window\.location\.pathname\.match\(/.test(l));
  if (re) check('Chestertons regex #2 (flight id anchor, gates subject extraction) matches a /sales/ URL',
    re.test(SALES_URL_PATH), `regex=${re} — currently /lettings/-only`);
}
{
  const re = liveRegexFromLine('extractPropertyId',
    (l) => /const match = pathname\.match\(/.test(l) && /\(\\d\+\)/.test(l) && /lettings|sales/.test(l));
  if (re) check('Chestertons regex #3 (extractPropertyId) matches a /sales/ URL',
    re.test(SALES_URL_PATH), `regex=${re} — currently /lettings/-only`);
}

// =============================================================================
// B. DETECTION + END-TO-END SUBJECT EXTRACTION on the /sales/ fixture.
// =============================================================================
const ex = loadExtractors(
  readFileSync(join(ROOT, 'tests/fixtures/for_sale/extension/chestertons_for_sale_detail.html'), 'utf8'),
  { hostname: 'www.chestertons.co.uk', pathname: SALES_URL_PATH, href: URL });

check('shim sanity: site detected as chestertons', ex.currentSite === 'chestertons', `got ${ex.currentSite}`);

// detectTenure already keys on /sales/ (content.js:626) — GREEN.
check("Chestertons-sale: detectTenure → 'sale' for the /sales/ URL",
  ex.detectTenure(URL, null) === 'sale', `got ${ex.detectTenure(URL, null)}`);

// extractPropertyId must resolve on a /sales/ URL (RED until regex #3 fix). Current
// branch is /lettings/-only so it returns null on /sales/.
check("Chestertons-sale: extractPropertyId() resolves on /sales/ (NOT null) → '21855578_FUL250188'",
  ex.extractPropertyId() === '21855578_FUL250188',
  `got ${ex.extractPropertyId()} — extractPropertyId Chestertons branch is /lettings/-only`);

const data = ex.extractPropertyDataChestertons();
check('Chestertons-sale: extractor returns data (NOT null) on the /sales/ subject fixture',
  data && typeof data === 'object',
  `got ${data} — flight anchor /lettings/-only regex fails to find the subject on /sales/`);

if (data) {
  check('Chestertons-sale: bedrooms === 5 (from flight subject)', data.bedrooms === 5, `got ${data.bedrooms}`);
  check('Chestertons-sale: bathrooms === 4 (from flight subject)', data.bathrooms === 4, `got ${data.bathrooms}`);
  check("Chestertons-sale: extractPostcode === 'W6 8LY'", ex.extractPostcode(data) === 'W6 8LY',
    `got ${ex.extractPostcode(data)}`);
  check('Chestertons-sale: extractSqftFromPage === 2390 (subject squareFeetInternal, non-null)',
    ex.extractSqftFromPage(data) === 2390, `got ${ex.extractSqftFromPage(data)}`);
  check("Chestertons-sale: extractPropertyType === 'house'",
    ex.extractPropertyType(data) === 'house', `got ${ex.extractPropertyType(data)}`);
  check('Chestertons-sale: floorplan URL resolves (homeflow-assets /files/floorplan/) → sale OCR can fire',
    String(ex.getFloorplanUrl(data) || '').includes('/files/floorplan/'),
    `got ${ex.getFloorplanUrl(data)}`);
  check('Chestertons-sale: price is the SALE lump sum 2150000 (NOT a pcm figure)',
    ex.parsePrice(data.prices && data.prices.primaryPrice) === 2150000,
    `raw="${data.prices && data.prices.primaryPrice}" → ${ex.parsePrice(data.prices && data.prices.primaryPrice)}`);
}

console.log('');
if (failures === 0) {
  console.log('=== PASS: Chestertons for-sale detection + /sales/ subject extraction green (real fixture) ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) RED (the three /lettings/-only regexes block /sales/) ===`);
  process.exit(1);
}
