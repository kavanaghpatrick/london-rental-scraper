/**
 * FLOORPLAN-OCR DISCOVERY test (TDD).
 *
 * THE GAP this targets: Chestertons listing 21806747/WEL150145 shows
 * "SIZE UNKNOWN — EST. ONLY" because the floorplan was never OCR'd. The subject's
 * page sqft is genuinely null (squareFeet / squareFeetInternal both null in the RSC
 * flight stream), so OCR of the floorplan is the ONLY way to recover a real estimate.
 * But at document_idle the live Chestertons PDP has ZERO rendered floorplan <img>
 * tags (the Floorplans tab is a lazy React skeleton), and the subject floorplan URL
 * exists ONLY inside the escaped __next_f flight stream as floorplans[0] on the
 * subject "property":{"id":<NUMID>} object. The Chestertons extractor already parses
 * that subject object (for beds/baths/type/postcode) but DROPS the floorplans field,
 * so getFloorplanUrl() returns nothing -> no OCR -> estimateSqft -> SIZE UNKNOWN.
 *
 * THE FIX (RED pre-fix -> GREEN post-fix): surface subject.floorplans[0] from the
 * already-parsed flight object and seed data.floorplans = [{url}] BEFORE the DOM
 * fallbacks, so getFloorplanUrl(data) returns the real, clean (no trailing-backslash)
 * homeflow-assets URL on the first call and the existing ocrFloorplan path fires.
 *
 * GROUND TRUTH (real captured fixture tests/chestertons_pdp_fixture.html):
 *   subject id 21806747; bedrooms 1; squareFeetInternal null (-> needs OCR);
 *   floorplans[0] = https://mr0.homeflow-assets.co.uk/files/floorplan/image/7009/5976/_x_/WEL150145_12.jpg
 *
 * Plus a UNIT test of ocrFloorplan's sqft-PARSE patterns on REAL floorplan OCR text
 * (captured by running system tesseract --psm 11 on the actual WEL150145_12.jpg, and
 * a second known-good string) so the parse logic is locked independently of the
 * in-browser OCR (which is the user's end-to-end verification).
 *
 * Uses the shared zero-dependency DOM shim (extract_test_shim.mjs), same as the
 * Rightmove / Knight Frank / Foxtons harnesses.
 *
 * Run: node chrome-extension/floorplan_discovery_test.mjs   (exit 0 = pass)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadExtractors, SRC } from './extract_test_shim.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE = readFileSync(
  join(__dirname, '..', 'tests', 'chestertons_pdp_fixture.html'), 'utf8');

const REAL_FLOORPLAN_URL =
  'https://mr0.homeflow-assets.co.uk/files/floorplan/image/7009/5976/_x_/WEL150145_12.jpg';

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK   ${name}`);
  else { failures++; console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`); }
}

// ---------------------------------------------------------------------------
// PART A — DISCOVERY: the Chestertons extractor must surface the subject
// floorplan URL from the flight stream so getFloorplanUrl returns it.
// ---------------------------------------------------------------------------
const ex = loadExtractors(FIXTURE, {
  hostname: 'www.chestertons.co.uk',
  pathname: '/properties/21806747/lettings/WEL150145',
  href: 'https://www.chestertons.co.uk/properties/21806747/lettings/WEL150145',
});

check('shim sanity: site detected as chestertons', ex.currentSite === 'chestertons',
  `got ${ex.currentSite}`);

const data = ex.extractPropertyDataChestertons();
check('CH: extractor returns data (NOT null — price gate satisfied)',
  data && typeof data === 'object', `got ${data}`);

if (data) {
  // Subject parsed correctly (proves the flight-stream path ran for this fixture).
  check('CH: bedrooms === 1 (subject from flight)', data.bedrooms === 1, `got ${data.bedrooms}`);
  // Subject genuinely has no page sqft -> OCR is the only path to real sqft.
  check('CH: no page sqft (subject squareFeetInternal null) -> OCR required',
    ex.extractSqftFromPage(data) == null,
    `got ${ex.extractSqftFromPage(data)} (expected null/undefined)`);

  // THE DISCOVERY FIX (RED pre-fix): data.floorplans[0].url must be the real subject URL.
  const fp = data.floorplans && data.floorplans[0] && data.floorplans[0].url;
  check('CH: data.floorplans[0].url === real subject floorplan URL (DISCOVERY fix)',
    fp === REAL_FLOORPLAN_URL, `got ${fp}`);

  // getFloorplanUrl must return that clean URL (no trailing backslash, correct ref).
  const url = ex.getFloorplanUrl(data);
  check('CH: getFloorplanUrl(data) returns the real floorplan URL',
    url === REAL_FLOORPLAN_URL, `got ${url}`);
  check('CH: floorplan URL carries the SUBJECT ref WEL150145 (not a carousel comp)',
    typeof url === 'string' && url.includes('WEL150145'), `got ${url}`);
  check('CH: floorplan URL has NO trailing backslash (would 404 on fetch)',
    typeof url === 'string' && !/\\$/.test(url) && !url.includes('\\'), `got ${JSON.stringify(url)}`);
  check('CH: floorplan host is homeflow-assets (covered by host_permissions)',
    typeof url === 'string' && url.includes('homeflow-assets.co.uk'), `got ${url}`);
}

// ---------------------------------------------------------------------------
// PART B — sqft PARSE: lift ocrFloorplan's exact sqftPatterns/sqmPatterns out of
// content.js and run them on REAL floorplan OCR text. This locks the parse logic
// (the in-browser OCR itself is the user's end-to-end verification).
// ---------------------------------------------------------------------------
function makeOcrSqftParser() {
  // Extract the two pattern arrays verbatim from content.js so the test tracks the
  // real source (no second copy to drift). They are the only two `const xPatterns = [`
  // arrays inside ocrFloorplan.
  function grabArray(label) {
    const at = SRC.indexOf(`const ${label} = [`);
    if (at === -1) throw new Error(`could not find ${label} in content.js`);
    const open = SRC.indexOf('[', at);
    let depth = 0, end = -1;
    for (let i = open; i < SRC.length; i++) {
      if (SRC[i] === '[') depth++;
      else if (SRC[i] === ']') { depth--; if (depth === 0) { end = i + 1; break; } }
    }
    // eslint-disable-next-line no-eval
    return eval(SRC.slice(open, end)); // array of RegExp literals — safe, repo-controlled
  }
  const sqftPatterns = grabArray('sqftPatterns');
  const sqmPatterns = grabArray('sqmPatterns');
  // Mirror ocrFloorplan's selection logic (sqft first, then sqm->sqft) and bounds.
  return function parse(text) {
    for (const p of sqftPatterns) {
      const m = text.match(p);
      if (m) {
        const sqft = parseInt(m[1].replace(',', ''), 10);
        if (sqft >= 100 && sqft <= 15000) return sqft;
      }
    }
    for (const p of sqmPatterns) {
      const m = text.match(p);
      if (m) {
        const sqm = parseInt(m[1].replace(',', ''), 10);
        if (sqm >= 10 && sqm <= 1500) return Math.round(sqm * 10.764);
      }
    }
    return null;
  };
}

const parseSqft = makeOcrSqftParser();

// Real OCR text captured from the WEL150145_12.jpg floorplan (system tesseract --psm 11).
const REAL_OCR_TEXT =
  'Approximate Gross Internal Area\n372 sq ft / 34.6 sqm\nAll measurements and areas are approximate only';
check('OCR-parse: real WEL150145 floorplan text -> 372 sqft (vs estimateSqft(1bed)=500)',
  parseSqft(REAL_OCR_TEXT) === 372, `got ${parseSqft(REAL_OCR_TEXT)}`);

// A second known-good string (sqm-first -> converted): 72.6 sq m is read as "72" -> 775.
check('OCR-parse: "Approximate Gross Internal Area 72.6 sq m/782 sq ft" -> 782 (sqft wins)',
  parseSqft('Approximate Gross Internal Area 72.6 sq m/782 sq ft') === 782,
  `got ${parseSqft('Approximate Gross Internal Area 72.6 sq m/782 sq ft')}`);

// sqm-only fallback path (no sqft on the plan): 34 sq m -> round(34*10.764)=366.
check('OCR-parse: sqm-only "34 sq m" -> 366 sqft (sqm fallback)',
  parseSqft('Total area 34 sq m') === 366, `got ${parseSqft('Total area 34 sq m')}`);

console.log('');
if (failures === 0) {
  console.log('=== PASS: Chestertons floorplan-URL discovery + OCR sqft-parse green ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) failed ===`);
  process.exit(1);
}
