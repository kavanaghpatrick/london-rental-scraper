/**
 * INC4 — RIGHTMOVE for-SALE detection + extraction test (TDD).
 *
 * Spec: /tmp/inc4_extension_fix.md  §"Rightmove (PARTIAL → works on paper, no sale fixture)"
 * and §"SALE OCR SIZE RECOVERY".
 *
 * Fixture: tests/fixtures/for_sale/extension/rightmove_for_sale_detail.html
 *   REAL captured live /properties/88970700 (HTTP 200, 2026-06-22) — NOT derived.
 *   5-bed terraced, £1,595,000, W4 5JS, transactionType="BUY"/channel="RES_BUY",
 *   lettings absent, sizings carries sqft 1831.  Verbatim window.__PAGE_MODEL devalue blob.
 *
 * WHAT THIS LOCKS DOWN
 *   1. detectTenure → 'sale' on the real BUY blob (the existing transactionType/channel
 *      fork is correct — this proves it against a real page; NO code change expected here).
 *   2. analyzeSale's model-input fields all resolve from the real for-sale blob:
 *      postcode / beds / baths / size / type / price.
 *   3. parsePrice() returns the BARE LUMP SUM 1595000 — it must NOT ×52/12 a sale price.
 *   4. SHARED SALE-OCR: recoverSizeSqft (the tenure-agnostic size-recovery helper the
 *      structural fix extracts from analyzeProperty) is EXPORTED — i.e. it exists in
 *      content.js and analyzeSale can call it. This is RED until that helper is added.
 *
 * EXPECTED STATE NOW: checks 1-3 PASS against the real fixture (Rightmove detection
 * already works); check 4 FAILS RED (recoverSizeSqft not yet wired). The file is wired
 * into ci.yml + REQUIRED_HARNESSES so it cannot silently skip.
 *
 * Run: node chrome-extension/rightmove_sale_extract_test.mjs   (exit 0 = pass)
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

const PATH = '/properties/88970700';
const URL = 'https://www.rightmove.co.uk' + PATH;
const ex = loadExtractors(
  readFileSync(join(ROOT, 'tests/fixtures/for_sale/extension/rightmove_for_sale_detail.html'), 'utf8'),
  { hostname: 'www.rightmove.co.uk', pathname: PATH, href: URL });

// Shim sanity so a failing assert is a real defect, not a shim mis-load.
check('shim sanity: site detected as rightmove', ex.currentSite === 'rightmove', `got ${ex.currentSite}`);

const pd = ex.extractPropertyDataRightmove();
check('RM-sale: extractor returns property data (NOT null) from __PAGE_MODEL',
  pd && typeof pd === 'object', `got ${pd}`);

if (pd) {
  // --- 1. SALE DETECTION (transactionType/channel = BUY) ---------------------------
  check("RM-sale: blob carries transactionType==='BUY'", pd.transactionType === 'BUY',
    `got ${pd.transactionType}`);
  check("RM-sale: blob carries channel==='RES_BUY'", pd.channel === 'RES_BUY', `got ${pd.channel}`);
  check('RM-sale: lettings absent on a sale page (null/undefined)',
    pd.lettings == null, `got ${JSON.stringify(pd.lettings)}`);
  check("RM-sale: detectTenure → 'sale' for the real BUY blob",
    ex.detectTenure(URL, pd) === 'sale', `got ${ex.detectTenure(URL, pd)}`);

  // --- 2. MODEL-INPUT FIELDS resolve from the for-sale blob -------------------------
  check('RM-sale: bedrooms === 5', pd.bedrooms === 5, `got ${pd.bedrooms}`);
  check('RM-sale: bathrooms === 2', pd.bathrooms === 2, `got ${pd.bathrooms}`);
  check("RM-sale: extractPostcode === 'W4 5JS'", ex.extractPostcode(pd) === 'W4 5JS',
    `got ${ex.extractPostcode(pd)}`);
  check("RM-sale: extractPropertyType === 'terraced'", ex.extractPropertyType(pd) === 'terraced',
    `got ${ex.extractPropertyType(pd)}`);
  check('RM-sale: extractSqftFromPage === 1831 (sqft chosen over ha/sqm/ac)',
    ex.extractSqftFromPage(pd) === 1831, `got ${ex.extractSqftFromPage(pd)}`);

  // --- 3. PRICE: bare lump sum, NO ×52/12 ------------------------------------------
  check("RM-sale: prices.primaryPrice is a bare lump sum '£1,595,000' (no pcm/pw token)",
    pd.prices && /£1,595,000/.test(String(pd.prices.primaryPrice)) &&
    !/pcm|pw/i.test(String(pd.prices.primaryPrice)),
    `got ${pd.prices && pd.prices.primaryPrice}`);
  check('RM-sale: parsePrice(primaryPrice) === 1595000 (NOT ×52/12-mangled)',
    ex.parsePrice(pd.prices && pd.prices.primaryPrice) === 1595000,
    `got ${ex.parsePrice(pd.prices && pd.prices.primaryPrice)}`);
}

// --- 4. SHARED SALE-OCR helper exists (RED until structural fix wires it) ----------
check('RM-sale: recoverSizeSqft is a function in content.js (shared sale OCR, used by analyzeSale)',
  typeof ex.recoverSizeSqft === 'function',
  `typeof=${typeof ex.recoverSizeSqft} — the tenure-agnostic size-recovery helper is not yet ` +
  `extracted from analyzeProperty; analyzeSale still reads page-only size (sale OCR not wired)`);

console.log('');
if (failures === 0) {
  console.log('=== PASS: Rightmove for-sale detection + extraction green (real fixture) ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) RED (recoverSizeSqft not yet shared into the sale path) ===`);
  process.exit(1);
}
