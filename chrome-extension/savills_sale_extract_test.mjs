/**
 * INC4 / DEFERRED ITEM C — SAVILLS for-SALE beds extraction test (TDD).
 *
 * Fixtures (REAL, captured 2026-06-22):
 *   tests/fixtures/for_sale/extension/savills_for_sale_detail.html
 *     from https://search.savills.com/property-detail/gbkgrsknu250108  (a SALE page)
 *   tests/fixtures/for_sale/extension/savills_to_rent_detail.html       (rent no-regression)
 *
 * ROOT CAUSE this test pins: Savills /property-detail/ is a Next.js shell. The visible
 * stats render as NON-ADJACENT spans (<span>4</span><span>Bedrooms</span>) and there is no
 * <main>/[role=main]/.property-details — the first <article> is a cookie banner — so the
 * old main-scoped pageText regex omits the beds entirely. The full listing blob is in
 * __NEXT_DATA__.props.initialReduxState.propertyDetail.property (present on BOTH rent and
 * sale; pageProps is empty). The fix reads beds/baths/sqft/price/address from that blob.
 *
 * REAL captured SALE values (verified against the live blob):
 *   Bedrooms=4, Bathrooms=3, SizeSqFt=2765, Price=800000,
 *   InvariantFullPriceText="Guide price £800,000", AddressLine2 tail = "CW3 0DA".
 * REAL captured RENT values: Bedrooms=0 (studio), RentBasis=1, "Guide price £1,907".
 *
 * Run: node chrome-extension/savills_sale_extract_test.mjs   (exit 0 = pass)
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

// ---------------------------------------------------------------------------
// SALE — the headline beds-extraction fix.
// ---------------------------------------------------------------------------
const saleHtml = readFileSync(
  join(ROOT, 'tests/fixtures/for_sale/extension/savills_for_sale_detail.html'), 'utf8');
const SALE_PATH = '/property-detail/gbkgrsknu250108';
const SALE_URL = 'https://search.savills.com' + SALE_PATH;
const ex = loadExtractors(saleHtml, {
  hostname: 'search.savills.com', pathname: SALE_PATH, href: SALE_URL });

check('shim sanity: site detected as savills', ex.currentSite === 'savills', `got ${ex.currentSite}`);

const data = ex.extractPropertyDataSavills();
check('Savills-sale: extractor returns data (NOT null)', data && typeof data === 'object', `got ${data}`);

// Beds render OUTSIDE the old pageText scope; they MUST come from __NEXT_DATA__.
check('Savills-sale: bedrooms === 4 (from __NEXT_DATA__ initialReduxState, NOT span regex)',
  data && data.bedrooms === 4, `got ${data && data.bedrooms}`);
check('Savills-sale: bathrooms === 3', data && data.bathrooms === 3, `got ${data && data.bathrooms}`);
check('Savills-sale: sqft === 2765',
  data && ((data.sizings?.[0]?.minimumSize === 2765) || ex.extractSqftFromPage(data) === 2765),
  `got sizings=${JSON.stringify(data && data.sizings)} extractSqftFromPage=${data && ex.extractSqftFromPage(data)}`);
check('Savills-sale: parsePrice === 800000 (Guide price lump sum, no ×52/12)',
  data && ex.parsePrice(data.prices?.primaryPrice) === 800000,
  `raw="${data && data.prices?.primaryPrice}" → ${data && ex.parsePrice(data.prices?.primaryPrice)}`);
check("Savills-sale: detectTenure === 'sale' (reads og:title / no-rental-frequency-token)",
  ex.detectTenure(SALE_URL, data) === 'sale', `got ${ex.detectTenure(SALE_URL, data)}`);
check("Savills-sale: extractPostcode === 'CW3 0DA' (tail of AddressLine2)",
  data && ex.extractPostcode(data) === 'CW3 0DA', `got ${data && ex.extractPostcode(data)}`);
check('Savills-sale: primaryPrice carries no rental-frequency token (sale lump sum)',
  data && data.prices && !/pcm|pw|per\s*(?:week|month)|weekly|monthly/i.test(String(data.prices.primaryPrice)),
  `got "${data && data.prices?.primaryPrice}"`);

// ---------------------------------------------------------------------------
// RENT — no-regression: the blob read must NOT break rent classification.
// rent blob => Bedrooms=0 (studio, VALID), RentBasis=1, "Guide price £1,907";
// detectTenure reads og:title ("Property to rent") => 'rent' (og:title path dominates).
// ---------------------------------------------------------------------------
const rentHtml = readFileSync(
  join(ROOT, 'tests/fixtures/for_sale/extension/savills_to_rent_detail.html'), 'utf8');
const RENT_PATH = '/property-detail/gbljylcwq250040';
const RENT_URL = 'https://search.savills.com' + RENT_PATH;
const exr = loadExtractors(rentHtml, {
  hostname: 'search.savills.com', pathname: RENT_PATH, href: RENT_URL });

const rentData = exr.extractPropertyDataSavills();
check('Savills-rent: extractor returns data (NOT null)', rentData && typeof rentData === 'object', `got ${rentData}`);
check('Savills-rent NO-REGRESSION: bedrooms === 0 (studio; blob read keeps a valid 0, not null)',
  rentData && rentData.bedrooms === 0, `got ${rentData && rentData.bedrooms}`);
check("Savills-rent NO-REGRESSION: detectTenure === 'rent' (og:title path dominates)",
  exr.detectTenure(RENT_URL, rentData) === 'rent', `got ${exr.detectTenure(RENT_URL, rentData)}`);
check('Savills-rent NO-REGRESSION: parsePrice(primaryPrice) ≈ 1907 (lump-sum Guide price)',
  rentData && Math.abs(exr.parsePrice(rentData.prices?.primaryPrice) - 1907) <= 1,
  `raw="${rentData && rentData.prices?.primaryPrice}" → ${rentData && exr.parsePrice(rentData.prices?.primaryPrice)}`);

console.log('');
if (failures === 0) {
  console.log('=== PASS: Savills for-sale beds extraction + rent no-regression green ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) RED ===`);
  process.exit(1);
}
