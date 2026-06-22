/**
 * INC4 — SAVILLS for-SALE detection + extraction test (TDD).
 *
 * Spec: /tmp/inc4_extension_fix.md  §"Savills (BROKEN — detectTenure returns 'rent' for
 * every /property-detail/)".
 *
 * Fixture: tests/fixtures/for_sale/extension/savills_for_sale_detail.json
 *   REAL captured 2026-06-22 from https://search.savills.com/property-detail/gbkgrsknu250108.
 *   Savills /property-detail/ is a Next.js SSG shell: __NEXT_DATA__.props.pageProps is EMPTY
 *   (listing data hydrates client-side). The ONLY reliable STATIC sale marker is the <head>
 *   og:title: "... | Property for sale | Savills"  (rental: "... | Property to rent | Savills").
 *   In the live hydrated DOM the price renders "Guide Price £X" with NO pcm/pw token.
 *
 * This test builds the page the extension actually sees in-browser: the real captured
 * og:title/meta in <head> + the hydrated DOM price ("Guide Price £1,750,000", no frequency
 * token) + beds/baths/sqft text in <main>. The detail URL is IDENTICAL for rent and sale,
 * so the URL can't decide — detectTenure MUST read the og:title (or the price's absence of
 * a rental-frequency token).
 *
 * THE BUG (RED pre-fix): the current detectTenure Savills branch only checks
 * transactionType/tenure/price-frequency — none of which extractPropertyDataSavills sets —
 * so it returns 'rent' for EVERY Savills page, sale included. The fix adds an og:title /
 * "no rental-frequency token on a /property-detail/" sale signal.
 *
 * NOTE: Savills sale-detail listing data hydrates from api.savills and is NOT in static
 * HTML, so the field-extraction asserts use the hydrated-DOM price/beds/sqft the capture
 * documents (DERIVED hydrated DOM around the REAL captured og:title marker — flagged).
 *
 * EXPECTED STATE NOW: detectTenure → 'rent' (RED; the spec's headline Savills bug); the
 * price/beds asserts pass (extractor + parsePrice already handle a lump sum).
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

const fx = JSON.parse(
  readFileSync(join(ROOT, 'tests/fixtures/for_sale/extension/savills_for_sale_detail.json'), 'utf8'));

// The page the in-browser content script sees: REAL captured og:title/meta in <head>
// (the static, pre-hydration SALE marker) + the hydrated DOM (price/beds/baths/sqft).
const ogTitle = fx.head_meta['og:title'];
const salePrice = fx.hydrated_dom_price_example_for_sale; // "Guide Price £1,750,000"
const html = `<!DOCTYPE html><html><head>` +
  `<meta property="og:title" content="${ogTitle}">` +
  `<meta name="description" content="${fx.head_meta.description}">` +
  `<title>${ogTitle}</title>` +
  `</head><body><main role="main">` +
  `<h1>Wheelwright Cottage, Woore Road, Buerton, Cheshire, CW3 0DA</h1>` +
  `<div class="sv-price">${salePrice}</div>` +
  `<div class="sv-stats">3 bedrooms 2 bathrooms 1,750 sq ft</div>` +
  `</main></body></html>`;

const PATH = '/property-detail/gbkgrsknu250108';
const URL = 'https://search.savills.com' + PATH;
const ex = loadExtractors(html, { hostname: 'search.savills.com', pathname: PATH, href: URL });

check('shim sanity: site detected as savills', ex.currentSite === 'savills', `got ${ex.currentSite}`);

// Marker present in the captured head (sanity that the fixture carries the real signal).
check("Savills-sale: captured og:title carries 'Property for sale'",
  /property for sale/i.test(ogTitle), `og:title="${ogTitle}"`);

const data = ex.extractPropertyDataSavills();
check('Savills-sale: extractor returns data (NOT null)', data && typeof data === 'object', `got ${data}`);

// --- SALE DETECTION (the headline Savills bug) -----------------------------------
// The URL is identical rent/sale; detectTenure must read the og:title (or the price's
// absence of a rental-frequency token). Current branch returns 'rent' → RED.
check("Savills-sale: detectTenure → 'sale' (reads og:title / no-rental-frequency-token)",
  ex.detectTenure(URL, data) === 'sale',
  `got ${ex.detectTenure(URL, data)} — current Savills branch only checks transactionType/tenure/` +
  `price-frequency (none set by extractPropertyDataSavills), so it returns 'rent' on a sale page`);

if (data) {
  // --- PRICE: lump sum, NO ×52/12 (already handled — locks the sale price path) -----
  check("Savills-sale: primaryPrice carries no rental-frequency token (sale lump sum)",
    data.prices && !/pcm|pw|per\s*(?:week|month)|weekly|monthly/i.test(String(data.prices.primaryPrice)),
    `got "${data.prices && data.prices.primaryPrice}"`);
  check('Savills-sale: parsePrice(primaryPrice) === 1750000 (Guide Price lump sum)',
    ex.parsePrice(data.prices && data.prices.primaryPrice) === 1750000,
    `raw="${data.prices && data.prices.primaryPrice}" → ${ex.parsePrice(data.prices && data.prices.primaryPrice)}`);
  // --- MODEL-INPUT FIELDS resolve from the hydrated DOM -----------------------------
  check('Savills-sale: extractSqftFromPage / page sqft === 1750',
    ex.extractSqftFromPage(data) === 1750 || (data.sizings && data.sizings[0] && data.sizings[0].minimumSize === 1750),
    `got sqft=${ex.extractSqftFromPage(data)} sizings=${JSON.stringify(data.sizings)}`);
  check("Savills-sale: extractPostcode === 'CW3 0DA' (from the address)",
    ex.extractPostcode(data) === 'CW3 0DA', `got ${ex.extractPostcode(data)}`);
}

console.log('');
if (failures === 0) {
  console.log('=== PASS: Savills for-sale detection + extraction green ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) RED (detectTenure returns 'rent' for every /property-detail/) ===`);
  process.exit(1);
}
