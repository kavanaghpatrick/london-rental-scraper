/**
 * INC4 — STRUCTURAL FIX + SHARED SALE-OCR test (TDD).
 *
 * Spec: /tmp/inc4_extension_fix.md  §"STRUCTURAL FIX" and §"SALE OCR SIZE RECOVERY".
 *
 * TWO production defects this locks down:
 *
 *  (A) THE PRE-FORK POA BAIL.  In init() the FOR-SALE fork (content.js ~327) sits AFTER
 *      the rental price-parse bail (content.js ~311-316):
 *          const askingPrice = parsePrice(propertyData.prices?.primaryPrice);
 *          if (!askingPrice) { injectError('Could not parse price'); return; }   // <-- bail
 *          ...
 *          if (detectTenure(...) === 'sale') { ... analyzeSale ... return; }     // <-- fork
 *      On a POA sale page parsePrice returns null (no number), so init() bails with
 *      "Could not parse price" and the sale fork never runs — a POA sale ERRORS instead of
 *      rendering low-confidence. The structural fix moves the sale fork (and a sale-aware
 *      price parse that tolerates POA → null askingPrice) BEFORE that rental bail.
 *
 *  (B) SALE OCR NOT SHARED.  analyzeSale reads size from the page ONLY (sale OCR "not wired
 *      in 4a"), while analyzeProperty recovers sqft via floorplan-tab-click + OCR. The fix
 *      extracts a tenure-agnostic recoverSizeSqft(propertyData) helper called by BOTH.
 *
 * HOW WE TEST (A) without invoking the whole init(): a SOURCE-STRUCTURAL invariant — the
 * sale-fork must appear BEFORE the "Could not parse price" bail in init(). Plus behavioural
 * proofs that the bail's trigger is real: parsePrice('Price on application') === null, and
 * detectTenure → 'sale' on the POA fixture (so the page IS a sale that must reach the fork).
 *
 * EXPECTED STATE NOW: the structural-order check FAILS RED (fork is after the bail today);
 * the POA-behaviour checks document the trigger; the recoverSizeSqft check FAILS RED.
 *
 * Run: node chrome-extension/structural_sale_poa_test.mjs   (exit 0 = pass)
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

// =============================================================================
// (A) STRUCTURAL: the sale fork must run BEFORE the rental "Could not parse price" bail.
//     We compare the source positions of the bail and the sale fork. Today the fork is
//     AFTER the bail (a POA sale errors out); the fix must place it BEFORE.
// =============================================================================
const bailIdx = SRC.indexOf("injectError('Could not parse price')");
const forkIdx = SRC.search(/detectTenure\([^)]*\)\s*===\s*'sale'/);
check('structural: the "Could not parse price" rental bail exists in init()', bailIdx !== -1,
  'bail string not found');
check("structural: a detectTenure(...)==='sale' fork exists", forkIdx !== -1, 'sale fork not found');
check('structural: the SALE fork runs BEFORE the rental "Could not parse price" bail',
  forkIdx !== -1 && bailIdx !== -1 && forkIdx < bailIdx,
  `forkIdx=${forkIdx} vs bailIdx=${bailIdx} — today the fork is AFTER the bail, so a POA sale ` +
  `bails with "Could not parse price" before analyzeSale runs`);

// =============================================================================
// (B) POA BEHAVIOUR: prove the bail's trigger, and that the page is a detectable sale.
// =============================================================================
const PATH = '/properties/residential/for-sale/alexandra-mansions-333-kings-road-chelsea-london-sw3/chq012691683';
const URL = 'https://www.knightfrank.co.uk' + PATH;
const ex = loadExtractors(
  readFileSync(join(ROOT, 'tests/fixtures/for_sale/extension/knightfrank_for_sale_poa_detail.html'), 'utf8'),
  { hostname: 'www.knightfrank.co.uk', pathname: PATH, href: URL });

const data = ex.extractPropertyDataKnightFrank();
check('POA: extractor returns data (NOT null) from the POA rendered DOM',
  data && typeof data === 'object', `got ${data}`);

// The exact condition that fires the premature bail: parsePrice of a POA string is null.
check("POA: parsePrice('Price on application') === null (this is what trips the bail)",
  ex.parsePrice('Price on application') === null, `got ${ex.parsePrice('Price on application')}`);
if (data) {
  check('POA: parsePrice(primaryPrice) is null/0 on the POA page (no number to parse)',
    !ex.parsePrice(data.prices && data.prices.primaryPrice),
    `raw="${data.prices && data.prices.primaryPrice}" → ${ex.parsePrice(data.prices && data.prices.primaryPrice)}`);
  // The page IS a sale → it MUST reach analyzeSale (low confidence), not error.
  check("POA: detectTenure → 'sale' (so the POA page must render via the sale fork, not bail)",
    ex.detectTenure(URL, data) === 'sale', `got ${ex.detectTenure(URL, data)}`);
  // The sale feature builder needs the POA qualifier.
  check("POA: extractPriceQualifier === 'POA'", ex.extractPriceQualifier(data) === 'POA',
    `got ${ex.extractPriceQualifier(data)}`);
}

// =============================================================================
// (C) SHARED SALE-OCR: recoverSizeSqft exists and is referenced by BOTH analyze paths.
// =============================================================================
check('sale-OCR: recoverSizeSqft is a function exported from content.js',
  typeof ex.recoverSizeSqft === 'function',
  `typeof=${typeof ex.recoverSizeSqft} — the shared size-recovery helper is not yet extracted ` +
  `from analyzeProperty; analyzeSale still reads page-only size (sale OCR not wired)`);
// Structural: the helper must be DEFINED and CALLED from both analyzeProperty and analyzeSale.
const defIdx = SRC.search(/function recoverSizeSqft\b/);
check('sale-OCR: recoverSizeSqft is DEFINED in content.js (function recoverSizeSqft)',
  defIdx !== -1, 'no `function recoverSizeSqft` definition — size recovery not extracted into a shared helper');
const callCount = (SRC.match(/recoverSizeSqft\s*\(/g) || []).length;
check('sale-OCR: recoverSizeSqft is CALLED at least twice (shared by analyzeProperty + analyzeSale)',
  callCount >= 2, `found ${callCount} call site(s) — must be invoked from BOTH analyze paths`);

console.log('');
if (failures === 0) {
  console.log('=== PASS: structural sale-fork ordering + POA handling + shared sale OCR green ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) RED (POA sale bails pre-fork; sale OCR not shared) ===`);
  process.exit(1);
}
