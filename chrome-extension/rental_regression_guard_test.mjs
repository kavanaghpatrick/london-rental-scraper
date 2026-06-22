/**
 * INC4 — RENTAL REGRESSION GUARD (TDD; must stay GREEN through the for-sale fix).
 *
 * Spec: /tmp/inc4_extension_fix.md  §HARD CONSTRAINTS ("ZERO rental regression") + §TESTS
 * ("Rental regression guard: … the rental detection still returns 'rent' on rental URLs").
 *
 * The for-sale work adds a detectTenure sale fork. The single most dangerous regression is
 * a sale signal that is TOO BROAD and starts classifying RENTAL pages as 'sale' — which
 * would route every rental into the sale predictor and silently break the shipped product.
 * This guard pins detectTenure → 'rent' on the REAL rental detail fixtures for every site:
 *
 *   Rightmove  tests/fixtures/extension/rightmove_169944029.html  (transactionType RENT / RES_LET, "£pcm")
 *   Foxtons    tests/fixtures/extension/foxtons_chpk0327321.html   (/properties-to-rent/, pricePcm)
 *   Knightfrank tests/knightfrank_pdp_fixture.html                 (/to-let/, "£692 per week")
 *   Chestertons tests/chestertons_pdp_fixture.html                 (/lettings/, primaryChannel lettings)
 *
 * It is GREEN now (pre-fix) and MUST STAY GREEN after the content.js writer adds the sale
 * fork. If a new sale heuristic misfires on a rental page, THIS test goes red. (The full
 * rental field-extraction regression lives in the existing rightmove/foxtons/knightfrank
 * _extract_test.mjs — wired separately; this guard is specifically the DETECTION split.)
 *
 * Run: node chrome-extension/rental_regression_guard_test.mjs   (exit 0 = pass)
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

// --- RIGHTMOVE rental: transactionType RENT / channel RES_LET, price "£… pcm" -------
{
  const PATH = '/properties/169944029';
  const URL = 'https://www.rightmove.co.uk' + PATH;
  const ex = loadExtractors(
    readFileSync(join(ROOT, 'tests/fixtures/extension/rightmove_169944029.html'), 'utf8'),
    { hostname: 'www.rightmove.co.uk', pathname: PATH, href: URL });
  const pd = ex.extractPropertyDataRightmove();
  check('RM rental: extractor returns data', pd && typeof pd === 'object', `got ${pd}`);
  check("RM rental: detectTenure → 'rent' (RENT/RES_LET blob, '£… pcm' price) — NOT 'sale'",
    ex.detectTenure(URL, pd) === 'rent', `got ${ex.detectTenure(URL, pd)}`);
}

// --- FOXTONS rental: /properties-to-rent/ URL, pricePcm populated -------------------
{
  const PATH = '/properties-to-rent/SW1X/chpk0327321';
  const URL = 'https://www.foxtons.co.uk' + PATH;
  const ex = loadExtractors(
    readFileSync(join(ROOT, 'tests/fixtures/extension/foxtons_chpk0327321.html'), 'utf8'),
    { hostname: 'www.foxtons.co.uk', pathname: PATH, href: URL });
  const data = ex.extractPropertyDataFoxtons();
  check('Foxtons rental: extractor returns data', data && typeof data === 'object', `got ${data}`);
  check("Foxtons rental: detectTenure → 'rent' (/properties-to-rent/ URL) — NOT 'sale'",
    ex.detectTenure(URL, data) === 'rent', `got ${ex.detectTenure(URL, data)}`);
}

// --- KNIGHTFRANK rental: /to-let/ URL, "£692 per week" -------------------------------
{
  const PATH = '/properties/residential/to-let/alexandra-mansions-333-kings-road-chelsea-london-sw3/chq012691683lg';
  const URL = 'https://www.knightfrank.co.uk' + PATH;
  const ex = loadExtractors(
    readFileSync(join(ROOT, 'tests/knightfrank_pdp_fixture.html'), 'utf8'),
    { hostname: 'www.knightfrank.co.uk', pathname: PATH, href: URL });
  const data = ex.extractPropertyDataKnightFrank();
  check('KF rental: extractor returns data', data && typeof data === 'object', `got ${data}`);
  check("KF rental: detectTenure → 'rent' ('£692 per week', /to-let/ URL) — NOT 'sale'",
    ex.detectTenure(URL, data) === 'rent', `got ${ex.detectTenure(URL, data)}`);
}

// --- CHESTERTONS rental: /lettings/ URL, primaryChannel "lettings" ------------------
{
  const PATH = '/properties/21806747/lettings/WEL150145';
  const URL = 'https://www.chestertons.co.uk' + PATH;
  const ex = loadExtractors(
    readFileSync(join(ROOT, 'tests/chestertons_pdp_fixture.html'), 'utf8'),
    { hostname: 'www.chestertons.co.uk', pathname: PATH, href: URL });
  // detectTenure for Chestertons keys purely on the URL (/lettings/ vs /sales/).
  check("Chestertons rental: detectTenure → 'rent' (/lettings/ URL) — NOT 'sale'",
    ex.detectTenure(URL, null) === 'rent', `got ${ex.detectTenure(URL, null)}`);
}

console.log('');
if (failures === 0) {
  console.log('=== PASS: rental detection unchanged — every rental page still → \'rent\' ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) — a for-sale heuristic is misfiring on a RENTAL page (regression) ===`);
  process.exit(1);
}
