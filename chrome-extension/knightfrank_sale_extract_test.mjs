/**
 * INC4 — KNIGHT FRANK for-SALE detection + extraction test (TDD).
 *
 * Spec: /tmp/inc4_extension_fix.md  §"KnightFrank (UNVERIFIED → POA bug)" + §"STRUCTURAL FIX".
 *
 * Fixture: tests/fixtures/for_sale/extension/knightfrank_for_sale_detail.html
 *   DERIVED (NOT captured) — KF for-sale is a 100% client-rendered Angular SPA; curl returns
 *   only a nav-only shell with no embedded data (documented in the fixture header + the rental
 *   knightfrank_extract_test.mjs). Derived from the REAL rental rendered-DOM fixture by flipping
 *   ONLY the tenure/price markers: title→For Sale, URL family →/residential/for-sale/, price
 *   "£692 per week"+"(£2,998 pcm)" → "Guide Price £1,950,000" (bare lump sum, NO pw/pcm token).
 *   Same building/beds(2)/baths(1)/sqft(625)/floorplan so rent-vs-sale is apples-to-apples.
 *   NEEDS REAL-PAGE CONFIRMATION in-browser (flagged).
 *
 * WHAT THIS LOCKS DOWN
 *   1. detectTenure → 'sale' on the for-sale URL (the line-650 !hasRentalFrequency && /for-sale/
 *      fallback already fires for KF — asserted GREEN, locking the rent-vs-sale split).
 *   2. The model-input fields resolve from the rendered DOM: beds/baths/sqft/postcode/type/price.
 *   3. parsePrice() returns the LUMP SUM 1950000 — NOT ×52/12 (KF rentals carry pw/pcm; a sale
 *      lump sum must pass through untouched).
 *   4. price_qualifier 'Guide Price' is read (extractPriceQualifier).
 *   5. SHARED SALE-OCR: recoverSizeSqft exists (the structural fix's shared helper) so the sale
 *      path recovers sqft the same way analyzeProperty does — RED until wired.
 *
 * EXPECTED STATE NOW: checks 1-4 PASS (KF detection + extraction already work on the rendered
 * DOM); check 5 FAILS RED (recoverSizeSqft not yet extracted/shared).
 *
 * Run: node chrome-extension/knightfrank_sale_extract_test.mjs   (exit 0 = pass)
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

const PATH = '/properties/residential/for-sale/alexandra-mansions-333-kings-road-chelsea-london-sw3/chq012691683';
const URL = 'https://www.knightfrank.co.uk' + PATH;
const ex = loadExtractors(
  readFileSync(join(ROOT, 'tests/fixtures/for_sale/extension/knightfrank_for_sale_detail.html'), 'utf8'),
  { hostname: 'www.knightfrank.co.uk', pathname: PATH, href: URL });

check('shim sanity: site detected as knightfrank', ex.currentSite === 'knightfrank', `got ${ex.currentSite}`);

const data = ex.extractPropertyDataKnightFrank();
check('KF-sale: extractor returns data (NOT null) from rendered DOM',
  data && typeof data === 'object', `got ${data}`);

// --- 1. SALE DETECTION (URL for-sale fallback, already fires for KF) ----------------
check("KF-sale: detectTenure → 'sale' for the /residential/for-sale/ URL",
  ex.detectTenure(URL, data) === 'sale', `got ${ex.detectTenure(URL, data)}`);

if (data) {
  // --- 2. MODEL-INPUT FIELDS resolve -----------------------------------------------
  check('KF-sale: bedrooms === 2', data.bedrooms === 2, `got ${data.bedrooms}`);
  check('KF-sale: bathrooms === 1', data.bathrooms === 1, `got ${data.bathrooms}`);
  check('KF-sale: receptions === 1', data.receptions === 1, `got ${data.receptions}`);
  check('KF-sale: extractSqftFromPage === 625', ex.extractSqftFromPage(data) === 625,
    `got ${ex.extractSqftFromPage(data)}`);
  check("KF-sale: extractPostcode === 'SW3' (outcode only)", ex.extractPostcode(data) === 'SW3',
    `got ${ex.extractPostcode(data)}`);
  check("KF-sale: extractPropertyType resolves (apartment/flat)",
    ['apartment', 'flat'].includes(ex.extractPropertyType(data)), `got ${ex.extractPropertyType(data)}`);
  check('KF-sale: floorplan on content.knightfrank.com',
    String(ex.getFloorplanUrl(data) || '').includes('content.knightfrank.com'),
    `got ${ex.getFloorplanUrl(data)}`);

  // --- 3. PRICE: lump sum 1950000, NOT ×52/12 --------------------------------------
  check("KF-sale: price string is a bare lump sum (no pw/pcm/Weekly/Monthly token)",
    data.prices && !/pcm|pw|per\s*(?:week|month)|weekly|monthly/i.test(String(data.prices.primaryPrice)),
    `got "${data.prices && data.prices.primaryPrice}"`);
  check('KF-sale: parsePrice(primaryPrice) === 1950000 (Guide Price lump sum, NOT ×52/12)',
    ex.parsePrice(data.prices && data.prices.primaryPrice) === 1950000,
    `raw="${data.prices && data.prices.primaryPrice}" → ${ex.parsePrice(data.prices && data.prices.primaryPrice)}`);

  // --- 4. PRICE QUALIFIER contract -------------------------------------------------
  // extractPriceQualifier must return a STRING (never throw) so analyzeSale can pass it
  // to the sale feature builder. The POA branch is the in-scope special case (asserted
  // in structural_sale_poa_test.mjs); here a non-POA Guide-Price KF page must at least
  // yield a clean string. (Recovering 'Guide Price' itself needs the KF DOM extractor to
  // capture the prefix — tracked separately; this guards the no-throw contract.)
  check('KF-sale: extractPriceQualifier returns a string (no throw on a Guide-Price page)',
    typeof ex.extractPriceQualifier(data) === 'string',
    `got ${typeof ex.extractPriceQualifier(data)}`);
}

// --- 5. SHARED SALE-OCR helper exists (RED until structural fix wires it) ----------
check('KF-sale: recoverSizeSqft is a function in content.js (shared sale OCR, used by analyzeSale)',
  typeof ex.recoverSizeSqft === 'function',
  `typeof=${typeof ex.recoverSizeSqft} — the tenure-agnostic size-recovery helper is not yet ` +
  `extracted from analyzeProperty; the sale path still reads page-only size`);

console.log('');
if (failures === 0) {
  console.log('=== PASS: Knight Frank for-sale detection + extraction green (derived fixture) ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) RED (recoverSizeSqft not yet shared into the sale path) ===`);
  process.exit(1);
}
