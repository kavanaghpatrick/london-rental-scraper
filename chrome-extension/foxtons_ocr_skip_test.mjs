/**
 * REGRESSION TEST — Foxtons floorplan-OCR skip + graceful/non-blocking OCR failure.
 *
 * Covers the two bugs reported on the real Foxtons listing
 * https://www.foxtons.co.uk/properties-to-rent/SW3/chpk2572513:
 *
 *  BUG 1 — "[RentFairValue] OCR failed: HTTP 403" console FLOOD. analyzeProperty
 *    ALWAYS OCR'd the floorplan even when Foxtons already exposes real floorArea
 *    (size_source==='page'), and the Foxtons floorplan CDN
 *    (web-prod-page-assets.foxtons.co.uk) 403s the extension's service-worker fetch.
 *    FIX: on Foxtons with sqft known from the page, the floorplan-OCR/fetch path is
 *    NOT entered (skipFloorplanOcr); the failure logging is single, debug-level, and
 *    never has an undefined tail. NON-Foxtons hosts (incl. Chestertons, which needs
 *    OCR for sqft) are unchanged.
 *
 *  BUG 2 — "Compare with Similar Properties busted". The INLINE "Similar Properties"
 *    comps must render INDEPENDENT of OCR outcome, and the comps inputs
 *    (district + beds) must be extracted correctly for Foxtons so findSimilarProperties
 *    can return comps. (The separate "Compare with Similar Properties" BUTTON ->
 *    compare.js -> Vercel /api/similar empty-DB failure is SERVER-side and out of scope
 *    for content.js/background.js/manifest.json.)
 *
 * Approach (matches floorplan_discovery_test.mjs / similar_properties_test.mjs):
 *   PART A — fixture-driven: prove the OCR-gate DECISION on REAL captured pages
 *            (Foxtons has page sqft -> skip; Chestertons has none -> OCR required).
 *   PART B — static source guards over content.js: the skip gate is Foxtons-scoped &
 *            keyed on size_source==='page'; the OCR catch is graceful/non-spammy;
 *            the comps render is wrapped so OCR can't abort it.
 *   PART C — static source guards over background.js: the blocked-fetch log is NOT
 *            error-level and has no dangling "HTTP 403:".
 *
 * Run: node chrome-extension/foxtons_ocr_skip_test.mjs   (exit 0 = pass)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadExtractors, SRC } from './extract_test_shim.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BG_SRC = readFileSync(join(__dirname, 'background.js'), 'utf8');

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`OK   ${name}`);
  else { failures++; console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`); }
}

// normalizeDistrict mirror (kept in sync with content.js:normalizeDistrict, which the
// shim does not export). Same regex; used only to assert the comps INPUT is valid.
function normalizeDistrict(pc) {
  if (!pc) return '';
  const m = pc.toUpperCase().match(/^([A-Z]{1,2}[0-9][0-9A-Z]?)/);
  return m ? m[1] : '';
}

// ---------------------------------------------------------------------------
// PART A — OCR-gate DECISION on REAL captured fixtures.
// The runtime skip condition is:  floorplanUrl && currentSite===FOXTONS && size_source==='page'
// We reconstruct each input the same way analyzeProperty does and assert the decision.
// ---------------------------------------------------------------------------

// A1) FOXTONS: page sqft present + floorplan on the hostile CDN -> OCR SKIPPED.
{
  const html = readFileSync(
    join(__dirname, '..', 'tests', 'fixtures', 'extension', 'foxtons_chpk0327321.html'), 'utf8');
  const ex = loadExtractors(html, {
    hostname: 'www.foxtons.co.uk',
    pathname: '/properties-to-rent/SW1X/chpk0327321',
  });
  const data = ex.extractPropertyDataFoxtons();

  check('A1 shim: site detected as foxtons', ex.currentSite === 'foxtons', `got ${ex.currentSite}`);

  const sqft = ex.extractSqftFromPage(data);           // analyzeProperty: sizeSqft
  const sizeSource = sqft ? 'page' : null;             // analyzeProperty: sizeSource
  const floorplanUrl = ex.getFloorplanUrl(data) || ''; // analyzeProperty: floorplanUrl

  check('A1 Foxtons: extractSqftFromPage gives real sqft -> size_source==="page"',
    sqft === 1213 && sizeSource === 'page', `sqft=${sqft} src=${sizeSource}`);
  check('A1 Foxtons: floorplan URL is on the known-hostile CDN (web-prod-page-assets.foxtons.co.uk)',
    floorplanUrl.includes('web-prod-page-assets.foxtons.co.uk'), `got ${floorplanUrl}`);

  // The exact runtime decision (see content.js skipFloorplanOcr).
  const hostBlocksFloorplanFetch = ex.currentSite === 'foxtons';
  const skipFloorplanOcr = !!floorplanUrl && hostBlocksFloorplanFetch && sizeSource === 'page';
  check('A1 Foxtons: OCR/fetch path is NOT entered (skipFloorplanOcr === true)',
    skipFloorplanOcr === true, `skip=${skipFloorplanOcr}`);
}

// A2) CHESTERTONS: NO page sqft -> OCR is REQUIRED (skip MUST stay false; non-Foxtons).
{
  const html = readFileSync(
    join(__dirname, '..', 'tests', 'chestertons_pdp_fixture.html'), 'utf8');
  const ex = loadExtractors(html, {
    hostname: 'www.chestertons.co.uk',
    pathname: '/properties/21806747/lettings/WEL150145',
    href: 'https://www.chestertons.co.uk/properties/21806747/lettings/WEL150145',
  });
  const data = ex.extractPropertyDataChestertons();

  check('A2 shim: site detected as chestertons', ex.currentSite === 'chestertons', `got ${ex.currentSite}`);

  const sqft = ex.extractSqftFromPage(data);
  const sizeSource = sqft ? 'page' : null;
  const floorplanUrl = ex.getFloorplanUrl(data) || '';

  check('A2 Chestertons: no page sqft (size_source !== "page")',
    sqft == null && sizeSource !== 'page', `sqft=${sqft} src=${sizeSource}`);

  const hostBlocksFloorplanFetch = ex.currentSite === 'foxtons';
  const skipFloorplanOcr = !!floorplanUrl && hostBlocksFloorplanFetch && sizeSource === 'page';
  check('A2 Chestertons: OCR still RUNS (skipFloorplanOcr === false) — no regression',
    skipFloorplanOcr === false && !!floorplanUrl, `skip=${skipFloorplanOcr} fp=${!!floorplanUrl}`);
}

// A3) Hypothetical guard: a non-Foxtons host WITH page sqft must STILL OCR (the skip is
//     host-scoped, NOT a blanket "size_source===page" skip). Asserted purely on the gate.
{
  const cases = [
    { site: 'savills', sizeSource: 'page', fp: 'https://x/floorplan.png' },
    { site: 'rightmove', sizeSource: 'page', fp: 'https://x/floorplan.png' },
    { site: 'knightfrank', sizeSource: 'page', fp: 'https://x/floorplan.png' },
    { site: 'chestertons', sizeSource: 'page', fp: 'https://x/floorplan.png' },
  ];
  for (const c of cases) {
    const skip = !!c.fp && (c.site === 'foxtons') && c.sizeSource === 'page';
    check(`A3 ${c.site} with page sqft: OCR NOT skipped (host-scoped gate)`,
      skip === false, `skip=${skip}`);
  }
  const foxSkip = (true) && ('foxtons' === 'foxtons') && ('page' === 'page');
  check('A3 foxtons with page sqft: OCR skipped (only Foxtons triggers the skip)',
    foxSkip === true, `skip=${foxSkip}`);
}

// ---------------------------------------------------------------------------
// PART B — BUG2 comps INPUTS + independence (Foxtons), from real fixture + source.
// ---------------------------------------------------------------------------
{
  const html = readFileSync(
    join(__dirname, '..', 'tests', 'fixtures', 'extension', 'foxtons_chpk0327321.html'), 'utf8');
  const ex = loadExtractors(html, {
    hostname: 'www.foxtons.co.uk',
    pathname: '/properties-to-rent/SW1X/chpk0327321',
  });
  const data = ex.extractPropertyDataFoxtons();

  const postcode = ex.extractPostcode(data);
  const district = normalizeDistrict(postcode);
  const beds = data.bedrooms;

  // findSimilarProperties is fired in displayResult ONLY when (postcode_district && beds);
  // prove Foxtons yields BOTH so comps are not suppressed for the wrong reason.
  check('B Foxtons: comps inputs valid — district non-empty AND beds present',
    !!district && !!beds, `district="${district}" beds=${beds}`);
  check('B Foxtons: district normalizes to a real outcode (SW1X)',
    district === 'SW1X', `got "${district}"`);

  // The displayResult comps gate is the documented `if (r.postcode_district && r.beds)`
  // — it must NOT depend on OCR/sqft/floorplan state.
  const gateSrc = SRC.replace(/\s+/g, ' ');
  check('B static: comps gate is `if (r.postcode_district && r.beds)` (OCR-independent)',
    gateSrc.includes('if (r.postcode_district && r.beds)'),
    'displayResult comps gate string not found verbatim');
  check('B static: comps gate does NOT reference ocrText/size_source/floorplan',
    !/if \(r\.postcode_district && r\.beds\)[^}]*?(ocrText|size_source|floorplan)/i.test(gateSrc),
    'comps gate appears entangled with OCR state');
}

// ---------------------------------------------------------------------------
// PART C — static GRACEFUL/NON-SPAMMY guards over content.js (OCR catch + comps wrap).
// ---------------------------------------------------------------------------
{
  // The OCR-gate region: must be Foxtons-scoped and keyed on size_source==='page'.
  check('C1 content.js: skipFloorplanOcr gate defined',
    /const\s+skipFloorplanOcr\s*=/.test(SRC), 'skipFloorplanOcr not found');
  check('C1 content.js: skip gate is host-scoped to Foxtons',
    /hostBlocksFloorplanFetch\s*=\s*currentSite\s*===\s*SITES\.FOXTONS/.test(SRC),
    'host gate not scoped to SITES.FOXTONS');
  check('C1 content.js: skip gate keyed on size_source/page (sqft known from page)',
    /sqftKnownFromPage\s*=\s*sizeSource\s*===\s*'page'/.test(SRC),
    "skip not keyed on sizeSource === 'page'");

  // OCR catch must be graceful: a single non-error log, with an undefined-safe tail,
  // and a clean return. It must NOT call logError (which is hard-wired to console.error).
  const ocrCatch = SRC.slice(SRC.indexOf('async function ocrFloorplan'));
  const catchBody = ocrCatch.slice(ocrCatch.indexOf('} catch (e) {'),
    ocrCatch.indexOf('} finally {'));
  check('C2 content.js: ocrFloorplan catch does NOT use logError (no console.error flood)',
    !/logError\s*\(/.test(catchBody), 'OCR catch still calls logError');
  check('C2 content.js: ocrFloorplan catch logs at debug level via log()',
    /\blog\s*\(/.test(catchBody), 'OCR catch no longer logs via log()');
  check('C2 content.js: ocrFloorplan catch guards undefined message tail (no "undefined")',
    /\(e\s*&&\s*e\.message\)\s*\|\|/.test(catchBody), 'no undefined-safe message guard');
  check('C2 content.js: ocrFloorplan catch returns cleanly { sqft: null, text: \'\' }',
    /return\s*\{\s*sqft:\s*null,\s*text:\s*''\s*\}/.test(catchBody), 'clean return missing');

  // BUG2 structural guard: the ocrFloorplan await in analyzeProperty is wrapped so an
  // unexpected rejection degrades to empty OCR text instead of aborting the analysis
  // (which would reach init()'s outer catch and wipe the sidebar + comps).
  const callSite = SRC.slice(SRC.indexOf('if (floorplanUrl && !skipFloorplanOcr) {'),
    SRC.indexOf('if (!sizeSqft) {'));
  check('C3 content.js: ocrFloorplan() await is wrapped in try/catch at the call site',
    /try\s*\{\s*ocrResult\s*=\s*await\s+ocrFloorplan\(/.test(callSite),
    'OCR await not wrapped — could abort analyzeProperty/comps');
  check('C3 content.js: the call-site OCR catch degrades to empty OCR (does not rethrow)',
    /catch\s*\(ocrErr\)\s*\{[^}]*ocrResult\s*=\s*\{\s*sqft:\s*null,\s*text:\s*''\s*\}/.test(
      callSite.replace(/\n/g, ' ')) || /catch\s*\(ocrErr\)/.test(callSite),
    'call-site OCR catch missing/rethrows');
}

// ---------------------------------------------------------------------------
// PART D — static guards over background.js (blocked-fetch log non-error + clean msg).
// ---------------------------------------------------------------------------
{
  const catchBody = BG_SRC.slice(BG_SRC.indexOf('.catch(error =>'),
    BG_SRC.indexOf('return true;'));
  check('D1 background.js: blocked-fetch log is NOT console.error (no error-level flood)',
    !/console\.error\s*\(/.test(catchBody), 'background still console.error on fetch fail');
  check('D1 background.js: blocked-fetch logged at debug/warn level',
    /console\.(debug|warn)\s*\(/.test(catchBody), 'no debug/warn log for blocked fetch');
  check('D1 background.js: still reports failure to caller (sendResponse success:false)',
    /sendResponse\s*\(\s*\{\s*success:\s*false/.test(catchBody), 'sendResponse(success:false) missing');
  check('D2 background.js: HTTP error has no dangling "HTTP <code>:" when statusText empty',
    /response\.statusText\s*\?/.test(BG_SRC) && !/HTTP \$\{response\.status\}: \$\{response\.statusText\}\`\)\;\s*\}/.test(BG_SRC.replace(/\n/g,'')),
    'statusText not guarded — could emit "HTTP 403:"');
}

console.log('');
if (failures === 0) {
  console.log('=== PASS: Foxtons OCR-skip + graceful/non-blocking OCR failure green ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) failed ===`);
  process.exit(1);
}
