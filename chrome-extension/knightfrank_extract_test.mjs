/**
 * WS2 — KNIGHT FRANK extractor test (TDD).
 *
 * KF property-detail pages are a 100% client-rendered ANGULAR SPA: a curl/WebFetch
 * returns only a ~59KB nav-only shell with empty og: tags and NO embedded JSON
 * (no __NEXT_DATA__, no window state). So extractPropertyDataKnightFrank()
 * (content.js) reads the RENDERED DOM via querySelector/innerText at document_idle.
 * This test exercises that path against a representative RENDERED-DOM fixture
 * (tests/knightfrank_pdp_fixture.html) whose tags/classes are the REAL Angular ones
 * and whose numeric ground truth is the spider's own captured record for the
 * building (Alexandra Mansions, 333 Kings Road, Chelsea, SW3).
 *
 * GROUND TRUTH: price "£692 per week" → parsePrice ×52/12 → 2999 pcm; beds 2;
 * baths 1; receptions 1; size 625 sqft; postcode SW3 (OUTCODE ONLY — KF carries no
 * incode); type flat; floorplan on content.knightfrank.com.
 *
 * THE FIX this targets (RED pre-fix): the KF extractor's internal postcode block
 * uses a FULL-postcode regex (outcode + incode \d[A-Z]{2}) that NEVER matches KF's
 * outcode-only "...London, SW3", so data.address.outcode is left UNSET. The fix adds
 * an outcode-only fallback so the KF extractor sets outcode='SW3' directly (matching
 * the spider's own outcode-only store). The user-visible extractPostcode==='SW3'
 * already works via the shared helper's free-text fallback, so it is asserted GREEN
 * both before and after — the RED→GREEN target is the DIRECT outcode set.
 * ALSO guards the carousel first-£-wins trap (subject £692 chosen over £1,500).
 *
 * Uses the shared, zero-dependency DOM shim (extract_test_shim.mjs) — same approach
 * as the Rightmove + Foxtons harnesses; the CI node job runs these with no install.
 *
 * Run: node chrome-extension/knightfrank_extract_test.mjs   (exit 0 = pass)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadExtractors } from './extract_test_shim.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE = readFileSync(
  join(__dirname, '..', 'tests', 'knightfrank_pdp_fixture.html'), 'utf8');

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK   ${name}`);
  else { failures++; console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`); }
}

const ex = loadExtractors(FIXTURE, {
  hostname: 'www.knightfrank.co.uk',
  pathname: '/properties/residential/to-let/alexandra-mansions-333-kings-road-chelsea-london-sw3/chq012691683lg',
});

// --- Shim sanity (so a failing assert below is a real extractor bug, not the shim) -
check('shim sanity: site detected as knightfrank', ex.currentSite === 'knightfrank',
  `got ${ex.currentSite}`);

const data = ex.extractPropertyDataKnightFrank();
check('KF: extractor returns data (NOT null)', data && typeof data === 'object', `got ${data}`);

if (data) {
  // PRICE — first-£-wins must latch onto the subject's £692/wk, not the £1,500 carousel.
  check('KF: parsePrice(primaryPrice) === 2999 (£692 pw → ×52/12)',
    ex.parsePrice(data.prices && data.prices.primaryPrice) === 2999,
    `primaryPrice=${data.prices && data.prices.primaryPrice} → ${ex.parsePrice(data.prices && data.prices.primaryPrice)}`);
  check('KF: price is the SUBJECT £692, not the carousel £1,500',
    data.prices && /£?692/.test(data.prices.primaryPrice),
    `got ${data.prices && data.prices.primaryPrice}`);

  check('KF: bedrooms === 2', data.bedrooms === 2, `got ${data.bedrooms}`);
  check('KF: bathrooms === 1', data.bathrooms === 1, `got ${data.bathrooms}`);
  check('KF: receptions === 1', data.receptions === 1, `got ${data.receptions}`);
  check('KF: size 625 sqft', ex.extractSqftFromPage(data) === 625, `got ${ex.extractSqftFromPage(data)}`);
  check("KF: propertyType === 'flat' (text says apartment/flat)",
    ex.extractPropertyType(data) === 'apartment' || ex.extractPropertyType(data) === 'flat',
    `got ${ex.extractPropertyType(data)}`);
  check("KF: floorplan on content.knightfrank.com",
    String(ex.getFloorplanUrl(data) || '').includes('content.knightfrank.com'),
    `got ${ex.getFloorplanUrl(data)}`);

  // POSTCODE (user-visible): extractPostcode recovers 'SW3' from the free-text address
  // via the shared helper's outcode fallback — GREEN before and after the KF fix.
  check("KF: extractPostcode === 'SW3' (user-visible, drives similar comps)",
    ex.extractPostcode(data) === 'SW3', `got ${ex.extractPostcode(data)}`);

  // THE OUTCODE FIX (RED pre-fix → GREEN post-fix): the KF extractor must set
  // data.address.outcode='SW3' DIRECTLY via an outcode-only fallback (its current
  // full-postcode regex leaves outcode unset on KF's "...London, SW3").
  check("KF: data.address.outcode === 'SW3' set DIRECTLY by KF extractor (outcode-only fix)",
    data.address && data.address.outcode === 'SW3',
    `got ${data.address && JSON.stringify({ outcode: data.address.outcode, incode: data.address.incode })}`);
}

console.log('');
if (failures === 0) {
  console.log('=== PASS: Knight Frank rendered-DOM extractor green (incl. outcode-only postcode) ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) failed ===`);
  process.exit(1);
}
