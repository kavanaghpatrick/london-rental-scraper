/**
 * headless_extraction_smoke.mjs — LIVE in-browser extension extraction smoke.
 *
 * ============================================================================
 * WHY THIS EXISTS (the blind spot PR CI structurally CANNOT cover)
 * ----------------------------------------------------------------------------
 * The committed extension harnesses (rightmove_extract_test.mjs,
 * *_sale_extract_test.mjs, …) run content.js's extractors against FROZEN,
 * COMMITTED HTML fixtures under tests/fixtures/**. That is exactly what makes
 * them deterministic, network-free, and safe to gate a PR on — and it is also
 * the precise reason they keep passing GREEN forever after a live site silently
 * migrates its DOM / JSON shape out from under our selectors. The whole
 * platform's "CI false-green" failure mode is this: the fixture still says
 * 2019's Rightmove, the LIVE site moved to window.__PAGE_MODEL (devalue-encoded),
 * the content.js regex misses it -> no popup, and NO pull-request test ever
 * fetches the live page to notice. (Real, documented postmortems: the Rightmove
 * __PAGE_MODEL migration "LIKELY BROKE" the extension with zero CI failure, and
 * floorplan extraction went to zero at a clean cliff while CI stayed green.)
 *
 * This harness is the OPPOSITE trade-off on purpose. For each site x {rent,sale}
 * it loads ONE known-live DETAIL url in a real headless browser, lets the page
 * HYDRATE (so window.__PAGE_MODEL / self.__next_f / JSON-LD are populated exactly
 * as a user's browser sees them), then runs the *real* content.js extractors
 * against that live DOM and asserts the extraction still produces a non-null,
 * plausible price / beds / sqft / postcode AND that the Fair-Value popup would
 * render. A failure here is the live-DOM-migration signal the frozen fixtures
 * can never raise.
 *
 * It is invoked ONLY by the weekly SCHEDULED extraction-drift-smoke workflow,
 * never by ci.yml — a live site flaking must never red-gate an unrelated PR.
 *
 * ============================================================================
 * HOW THE HEADLESS INJECTION WORKS (and why this approach, not unpacked MV3)
 * ----------------------------------------------------------------------------
 * Loading the full unpacked MV3 extension into headless Chrome (persistent
 * context + --load-extension, wait for the service worker, scrape the injected
 * shadow-DOM popup) is brittle at CI scale: MV3 service-worker lifecycle in
 * headless is flaky, the popup fetches our live Vercel /api/predict + GitHub-raw
 * model JSON (more live deps to flake), and asserting on injected shadow DOM is
 * fragile. The task explicitly sanctions the more tractable approach when full
 * MV3 loading is too brittle — and that is what we do:
 *
 *   1. PYTHON PLAYWRIGHT fetches the live page (it is already installed +
 *      `playwright install chromium`-ed in the workflow for the SPA spiders).
 *      A tiny inline Python driver (run as a subprocess) navigates to the url,
 *      waits for network-idle + hydration, and emits a JSON envelope on stdout:
 *        { ok, html, pageModel, status, finalUrl, err }
 *      where `html` is the fully-rendered document.documentElement.outerHTML
 *      (so embedded <script>window.__PAGE_MODEL=…</script>, the RSC __next_f
 *      chunks, JSON-LD and the hydrated innerText are all present), and
 *      `pageModel` is the *serialized runtime global* window.__PAGE_MODEL
 *      (Rightmove sets this post-hydration; content.js Strategy 3a reads it as a
 *      runtime global, so we must reproduce it — see below).
 *
 *   2. The EXISTING export-trampoline (extract_test_shim.mjs :: loadExtractors)
 *      — the same zero-dependency injector every committed extractor test uses —
 *      runs the *unmodified* content.js IIFE against that live HTML, with
 *      window.location.hostname set so the script's own detectSite() picks the
 *      site. We monkey-patch the shim's window shim to also expose the captured
 *      runtime window.__PAGE_MODEL, so the Rightmove runtime-global path
 *      (Strategy 3a) is exercised exactly as in a real browser, not only the
 *      embedded-<script> path (3b).
 *
 *   3. We call the real extractors + shared helpers
 *      (extractPropertyData<Site>, extractPostcode, extractSqftFromPage,
 *       extractPropertyType, parsePrice, detectTenure) and assert plausibility.
 *      "Popup would render" is asserted structurally: the model inputs the popup
 *      builder consumes (price + beds resolve, tenure detected) are all present —
 *      i.e. analyzeProperty/analyzeSale would reach the rfv-container render
 *      rather than bail on a null extraction.
 *
 * Net: we reuse the repo's proven injection convention (no node_modules added to
 * the extension, no MV3-load flake) and get a TRUE live-DOM signal: Playwright
 * supplies the live hydrated DOM, the trampoline supplies the real in-page code.
 *
 * ============================================================================
 * ROBUSTNESS
 * ----------------------------------------------------------------------------
 *  - Per (site,mode): independent try/catch. One site's failure NEVER aborts the
 *    others — we collect results and print a single PASS/FAIL matrix at the end.
 *  - Network retries: each live fetch is retried (default 3x) with backoff; only
 *    after exhausting retries is the cell marked a hard FAIL. A fetch that never
 *    yields a page (timeout/blocked) is distinguished from a page that loaded but
 *    extracted nothing (the real drift signal).
 *  - A captured-DOM excerpt + a per-field diff are emitted for any failing cell
 *    so the issue body can show WHAT the live page now looks like vs expectation.
 *
 * EXIT CODE: 0 = every required cell PASS; 1 = at least one hard drift FAIL.
 *   (Soft-skips — e.g. a url that 404s because the listing sold — are reported
 *    but do NOT fail the run unless EVERY cell for a site soft-skips.)
 *
 * Run locally:   node chrome-extension/headless_extraction_smoke.mjs
 *   optional:    SMOKE_SITES=rightmove,savills  SMOKE_RETRIES=2  SMOKE_JSON=1
 * ============================================================================
 */

import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadExtractors, SRC } from './extract_test_shim.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// KNOWN-LIVE DETAIL URLS — site x {rent, sale}.  THESE DRIFT (listings sell /
// get withdrawn), so they live in ONE reviewable place. Seeded from the exact
// real pages the committed fixtures were captured from (so they are known-good
// shapes). When a cell SOFT-SKIPs (404/gone) week after week, swap in a fresh
// live url of the same site+mode here — that is the only maintenance this needs.
//
// `host` is the hostname content.js's detectSite() keys on. `path` is the detail
// path. We never hardcode query strings the site may reject.
// ---------------------------------------------------------------------------
const LIVE_URLS = {
  rightmove: {
    host: 'www.rightmove.co.uk',
    // rent: a real to-rent detail; sale: the captured for-sale detail (W4, 5-bed).
    rent: '/properties/169944029',
    sale: '/properties/88970700',
  },
  knightfrank: {
    host: 'www.knightfrank.co.uk',
    rent: '/properties/residential/to-let/alexandra-mansions-333-kings-road-chelsea-london-sw3/chq012691683lg',
    sale: '/properties/residential/for-sale/alexandra-mansions-333-kings-road-chelsea-london-sw3/chq012691683',
  },
  chestertons: {
    host: 'www.chestertons.co.uk',
    rent: '/properties/21844781/lettings/CJL260150',
    sale: '/properties/21855578/sales/FUL250188',
  },
  savills: {
    host: 'search.savills.com',
    // Savills detail pages are under /property-detail/<id>. We only seed the REAL
    // captured FOR-SALE detail (gbkgrsknu250108). A distinct to-let detail id has
    // not been captured, so `rent` is intentionally null -> that cell SKIPs with a
    // "needs a real url" note rather than reusing the sale id and raising a bogus
    // sale/rent-fork FAIL. Seed a live to-let /property-detail/<id> here to enable it.
    rent: null,
    sale: '/property-detail/gbkgrsknu250108',
  },
  foxtons: {
    host: 'www.foxtons.co.uk',
    rent: '/properties-to-rent/SW1X/chpk0327321',
    sale: '/properties-for-sale/sw7/chpk2514513',
  },
};

// Per-site extractor name + plausibility bands. Sale/rent share an extractor
// (the extractor returns the raw blob; detectTenure + parsePrice interpret it).
const SITE_EXTRACTOR = {
  rightmove: 'extractPropertyDataRightmove',
  knightfrank: 'extractPropertyDataKnightFrank',
  chestertons: 'extractPropertyDataChestertons',
  savills: 'extractPropertyDataSavills',
  foxtons: 'extractPropertyDataFoxtons',
};

// Plausible London magnitude bands per mode. Deliberately WIDE — we are catching
// a COLLAPSE (null / wrong-vertical) not asserting a specific listing's price.
const BANDS = {
  rent: { priceLo: 400, priceHi: 80_000, label: 'pcm' },         // £/month
  sale: { priceLo: 80_000, priceHi: 80_000_000, label: 'lump' }, // £ asking
};
const BEDS_LO = 0;   // studio == 0 beds is valid
const BEDS_HI = 12;
const SQFT_LO = 80;  // a real flat is > ~80 sqft
const SQFT_HI = 60_000;

const RETRIES = Math.max(1, parseInt(process.env.SMOKE_RETRIES || '3', 10));
const NAV_TIMEOUT_MS = parseInt(process.env.SMOKE_NAV_TIMEOUT_MS || '45000', 10);
const EMIT_JSON = process.env.SMOKE_JSON === '1';

const wantSites = (process.env.SMOKE_SITES || 'all').trim();
const SITES =
  wantSites === '' || wantSites === 'all'
    ? Object.keys(LIVE_URLS)
    : wantSites.split(',').map((s) => s.trim()).filter(Boolean);

// ---------------------------------------------------------------------------
// LIVE FETCH via Python Playwright (subprocess). Returns
//   { ok, status, finalUrl, html, pageModel, err }
// `pageModel` is the JSON.stringify of the runtime window.__PAGE_MODEL (or null).
// We pass the script on stdin so there is no temp file to clean up.
// ---------------------------------------------------------------------------
const PY_DRIVER = String.raw`
import sys, json
URL = sys.argv[1]
TIMEOUT = int(sys.argv[2])
out = {"ok": False, "status": None, "finalUrl": None, "html": None, "pageModel": None, "err": None}
try:
    from playwright.sync_api import sync_playwright
except Exception as e:
    out["err"] = "playwright import failed: %r" % (e,)
    print(json.dumps(out)); sys.exit(0)
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"),
            locale="en-GB",
            viewport={"width": 1366, "height": 900},
        )
        page = ctx.new_page()
        resp = None
        try:
            resp = page.goto(URL, wait_until="domcontentloaded", timeout=TIMEOUT)
        except Exception as e:
            out["err"] = "goto failed: %r" % (e,)
            browser.close(); print(json.dumps(out)); sys.exit(0)
        # Let client JS hydrate: network-idle is best-effort (some sites long-poll).
        try:
            page.wait_for_load_state("networkidle", timeout=min(TIMEOUT, 15000))
        except Exception:
            pass
        # Best-effort: dismiss a cookie wall that can gate hydrated content.
        for sel in ["button#onetrust-accept-btn-handler",
                    "button:has-text('Accept all')",
                    "button:has-text('Accept All')",
                    "button:has-text('I Accept')"]:
            try:
                el = page.query_selector(sel)
                if el:
                    el.click(timeout=2000); page.wait_for_timeout(500); break
            except Exception:
                pass
        out["status"] = resp.status if resp else None
        out["finalUrl"] = page.url
        # Walk the redirect chain so a 404/410 that the browser then 30x-redirected to a
        # generic 200 landing page is still reported as "the listing is gone" (the seed
        # url is stale), not mistaken for live-DOM drift. resp is the FINAL response;
        # request.redirected_from chains back to the original.
        try:
            statuses = []
            req = resp.request if resp else None
            seen = 0
            while req is not None and seen < 12:
                rr = req.response()
                if rr is not None:
                    statuses.append(rr.status)
                req = req.redirected_from
                seen += 1
            out["redirectStatuses"] = statuses
        except Exception:
            out["redirectStatuses"] = []
        try:
            out["html"] = page.content()
        except Exception as e:
            out["err"] = "content() failed: %r" % (e,); browser.close(); print(json.dumps(out)); sys.exit(0)
        # Serialize the RUNTIME window.__PAGE_MODEL (Rightmove sets it post-hydration;
        # content.js Strategy 3a reads it as a runtime global). May be absent.
        try:
            pm = page.evaluate("() => (typeof window.__PAGE_MODEL !== 'undefined') ? JSON.stringify(window.__PAGE_MODEL) : null")
            out["pageModel"] = pm
        except Exception:
            out["pageModel"] = None
        out["ok"] = bool(out["html"])
        browser.close()
except Exception as e:
    out["err"] = "driver crashed: %r" % (e,)
print(json.dumps(out))
`;

function pyBin() {
  // Mirror the global rule / workflow: python3 first, then python.
  for (const b of [process.env.PYTHON, 'python3', 'python']) {
    if (!b) continue;
    const r = spawnSync(b, ['--version'], { encoding: 'utf8' });
    if (r.status === 0 || (r.stdout || r.stderr || '').toLowerCase().includes('python')) return b;
  }
  return 'python3';
}
const PYTHON = pyBin();

function fetchLive(url, timeoutMs) {
  const r = spawnSync(PYTHON, ['-', url, String(timeoutMs)], {
    input: PY_DRIVER,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024, // pages can be large
    timeout: timeoutMs + 30000,
  });
  if (r.error) return { ok: false, err: `spawn: ${r.error.message}` };
  if (r.status !== 0 && !r.stdout) {
    return { ok: false, err: `python exit ${r.status}: ${(r.stderr || '').slice(-400)}` };
  }
  // The driver prints exactly one JSON line on stdout (last line wins if logs leak).
  const line = (r.stdout || '').trim().split('\n').filter(Boolean).pop() || '';
  try {
    return JSON.parse(line);
  } catch (e) {
    return { ok: false, err: `unparseable driver output: ${(r.stdout || r.stderr || '').slice(-400)}` };
  }
}

function fetchLiveWithRetry(url, label) {
  let last = null;
  for (let attempt = 1; attempt <= RETRIES; attempt++) {
    const res = fetchLive(url, NAV_TIMEOUT_MS);
    last = res;
    if (res && res.ok && res.html) return res;
    // brief backoff between attempts (busy-wait is fine — short, CI-only)
    const ms = 1500 * attempt;
    const end = Date.now() + ms;
    while (Date.now() < end) { /* backoff */ }
    process.stderr.write(`    [${label}] live fetch attempt ${attempt}/${RETRIES} failed` +
      `${res && res.err ? ' (' + res.err + ')' : ''}; retrying…\n`);
  }
  return last || { ok: false, err: 'no result' };
}

// ---------------------------------------------------------------------------
// Run the real extractors against the live HTML via the export-trampoline,
// injecting the captured runtime window.__PAGE_MODEL so Strategy 3a fires.
// ---------------------------------------------------------------------------
function extractFromLive(html, { host, path, url, pageModelJson }) {
  // page.content() already includes the embedded <script>window.__PAGE_MODEL=…</script>
  // and the __next_f / __NEXT_DATA__ / JSON-LD blobs, so the trampoline's
  // serialized-<script> strategies (Rightmove 3b, Chestertons flight, Savills
  // JSON-LD) fire against the LIVE shape directly.
  //
  // BUT Rightmove's __PAGE_MODEL is sometimes ONLY a post-hydration RUNTIME global
  // (Strategy 3a) with no embedded mirror. The captured `pageModelJson` is exactly
  // that runtime global. To exercise 3a faithfully without modifying the shim, we
  // synthesize an embedded <script>window.__PAGE_MODEL=<captured JSON></script> and
  // inject it into the live HTML so the trampoline's 3a/3b paths both have the data
  // a real browser saw. This is a no-op shape-wise (it is the same object the live
  // runtime held) and only ADDS the marker when the live DOM lacked the embedded
  // copy — it never overrides a richer embedded blob, since we append, and 3a is
  // tried before 3b inside content.js anyway.
  let liveHtml = html;
  if (pageModelJson && !/window\.__PAGE_MODEL\s*=/.test(html)) {
    const inject =
      `<script>window.__PAGE_MODEL=${pageModelJson};</script>`;
    liveHtml = /<\/body>/i.test(html)
      ? html.replace(/<\/body>/i, `${inject}</body>`)
      : html + inject;
  }
  const ex = loadExtractors(liveHtml, { hostname: host, pathname: path, href: url });
  return { ex, pageModelJson };
}

// Decide the verdict for one (site,mode) cell.
function evaluateCell(site, mode, ex, pageModelJson, diag, url) {
  const band = BANDS[mode];
  const out = { site, mode, status: 'PASS', reasons: [], fields: { url } };

  const extractorName = SITE_EXTRACTOR[site];
  const extractor = ex[extractorName];
  if (typeof extractor !== 'function') {
    out.status = 'FAIL';
    out.reasons.push(`extractor ${extractorName} missing from content.js (export-trampoline returned ${typeof extractor})`);
    return out;
  }
  if (ex.currentSite !== site) {
    out.status = 'FAIL';
    out.reasons.push(`detectSite mismatch: content.js saw '${ex.currentSite}', expected '${site}' (hostname routing drift)`);
    return out;
  }

  let pd = null;
  try {
    pd = extractor();
  } catch (e) {
    out.status = 'FAIL';
    out.reasons.push(`${extractorName}() threw: ${e && e.message}`);
    return out;
  }

  // NULL extraction == the headline drift signal (site moved its data shape).
  if (!pd || typeof pd !== 'object') {
    out.status = 'FAIL';
    out.reasons.push(
      `${extractorName}() returned ${pd === null ? 'null' : typeof pd} on the LIVE page — ` +
      `the data shape this extractor parses is GONE (DOM/JSON migration). ${diag}`);
    return out;
  }

  // --- PRICE: resolve via the same path the popup uses (parsePrice on the blob's
  //     primary price string), then magnitude-band it. ---------------------------
  const rawPrice =
    (pd.prices && (pd.prices.primaryPrice ?? pd.prices.displayPrices?.[0]?.displayPrice)) ??
    pd.price ?? pd.askingPrice ?? null;
  const price = typeof ex.parsePrice === 'function' ? ex.parsePrice(rawPrice) : null;
  out.fields.rawPrice = rawPrice;
  out.fields.price = price;
  if (price == null || price < band.priceLo || price > band.priceHi) {
    out.status = 'FAIL';
    out.reasons.push(
      `price ${price} (raw=${JSON.stringify(rawPrice)}) outside plausible ${mode} band ` +
      `[${band.priceLo},${band.priceHi}] — null/zero == price selector drift; ` +
      `out-of-band == wrong-vertical parse (${band.label})`);
  }

  // --- BEDS --------------------------------------------------------------------
  const beds = pd.bedrooms ?? pd.beds ?? null;
  out.fields.beds = beds;
  if (beds == null || beds < BEDS_LO || beds > BEDS_HI) {
    out.status = 'FAIL';
    out.reasons.push(`beds ${beds} missing/implausible (expected ${BEDS_LO}..${BEDS_HI}) — beds selector drift`);
  }

  // --- POSTCODE (best-effort: shared helper) -----------------------------------
  const postcode = typeof ex.extractPostcode === 'function' ? ex.extractPostcode(pd) : null;
  out.fields.postcode = postcode;
  // A UK outward+inward or at least a valid outward code. extractPostcode returns
  // null deliberately when there is no real postcode, so treat null as a WARN
  // (model still runs district-less) but a garbage string as a FAIL.
  if (postcode != null && !/^[A-Z]{1,2}\d[A-Z\d]?(\s*\d[A-Z]{2})?$/i.test(String(postcode).trim())) {
    out.status = 'FAIL';
    out.reasons.push(`postcode '${postcode}' is not a valid UK code — postcode parse drift`);
  } else if (postcode == null) {
    out.reasons.push(`WARN: postcode null (model runs district-less; not a hard fail)`);
  }

  // --- SQFT (best-effort: page-present sqft; many real pages legitimately lack it
  //     and rely on floorplan-OCR enrichment, so null is a WARN not a FAIL) ------
  const sqft = typeof ex.extractSqftFromPage === 'function' ? ex.extractSqftFromPage(pd) : null;
  out.fields.sqft = sqft;
  if (sqft != null && (sqft < SQFT_LO || sqft > SQFT_HI)) {
    out.status = 'FAIL';
    out.reasons.push(`sqft ${sqft} implausible (expected ${SQFT_LO}..${SQFT_HI}) — size parse drift`);
  } else if (sqft == null) {
    out.reasons.push(`WARN: sqft null on page (OCR-enrichment territory; not a hard fail)`);
  }

  // --- PROPERTY TYPE (best-effort) ---------------------------------------------
  const ptype = typeof ex.extractPropertyType === 'function' ? ex.extractPropertyType(pd) : null;
  out.fields.propertyType = ptype;

  // --- TENURE: detectTenure must classify the page; for the 'sale' cell it must
  //     say 'sale', for 'rent' it must NOT say 'sale'. This is the exact fork the
  //     popup uses to choose analyzeSale vs analyzeProperty. --------------------
  if (typeof ex.detectTenure === 'function') {
    let tenure = null;
    try { tenure = ex.detectTenure(url || '', pd); } catch (e) { tenure = `threw:${e && e.message}`; }
    out.fields.tenure = tenure;
    if (mode === 'sale' && tenure !== 'sale') {
      out.status = 'FAIL';
      out.reasons.push(`detectTenure='${tenure}' on a known FOR-SALE url (expected 'sale') — sale/rent fork drift`);
    }
    if (mode === 'rent' && tenure === 'sale') {
      out.status = 'FAIL';
      out.reasons.push(`detectTenure='sale' on a known TO-RENT url — sale/rent fork drift`);
    }
  }

  // --- POPUP-WOULD-RENDER (structural): the popup builder (analyzeProperty /
  //     analyzeSale -> displayResult/displaySaleResult, rfv-container) only runs
  //     if extraction yielded the model inputs. We assert the necessary-and-
  //     sufficient inputs are present: a banded price AND a valid bed count. If
  //     both resolve, analyze*() reaches the rfv-container render rather than
  //     bailing on a null extraction. (We do NOT actually fetch the live model —
  //     that would add live-API flake; this is the in-page render precondition.) -
  const popupWouldRender = price != null && price >= band.priceLo && price <= band.priceHi &&
    beds != null && beds >= BEDS_LO && beds <= BEDS_HI;
  out.fields.popupWouldRender = popupWouldRender;
  if (!popupWouldRender && out.status === 'PASS') {
    out.status = 'FAIL';
    out.reasons.push('popup would NOT render: model inputs (price+beds) did not both resolve');
  }

  return out;
}

// ---------------------------------------------------------------------------
// Main: iterate site x mode, fetch live, extract, evaluate, collect.
// ---------------------------------------------------------------------------
function domExcerpt(html, n = 600) {
  if (!html) return '(no html captured)';
  // Prefer the <head> + first chunk of body so the diff shows the markers we key on.
  const head = (html.match(/<head[\s\S]*?<\/head>/i) || [''])[0].slice(0, n);
  const scriptHints = (html.match(/window\.__?PAGE_MODEL|__NEXT_DATA__|__next_f|application\/ld\+json/gi) || [])
    .slice(0, 6).join(', ') || '(none of the known data markers present!)';
  return `markers: ${scriptHints}\nhead[0:${n}]: ${head.replace(/\s+/g, ' ').slice(0, n)}`;
}

function run() {
  console.log('============================================================');
  console.log('  HEADLESS EXTENSION EXTRACTION SMOKE (live, weekly)');
  console.log(`  python=${PYTHON}  retries=${RETRIES}  nav_timeout=${NAV_TIMEOUT_MS}ms`);
  console.log(`  content.js bytes=${SRC.length}  sites=${SITES.join(',')}`);
  console.log('============================================================\n');

  const results = [];

  for (const site of SITES) {
    const cfg = LIVE_URLS[site];
    if (!cfg) {
      results.push({ site, mode: '-', status: 'FAIL', reasons: [`no LIVE_URLS entry for site '${site}'`], fields: {} });
      continue;
    }
    for (const mode of ['rent', 'sale']) {
      const path = cfg[mode];
      const label = `${site}/${mode}`;
      // A mode with no seeded url is a deliberate SKIP (we have not captured a
      // known-live detail page for it) — NOT a drift FAIL. It is excluded from the
      // per-site collapse rule below via the `unseeded` flag.
      if (!path) {
        console.log(`--- ${label}  ->  (no seeded url)`);
        console.log('    SKIP  (no known-live url seeded for this mode)');
        results.push({
          site, mode, status: 'SKIP', unseeded: true,
          reasons: [`no known-live url seeded for ${label} in LIVE_URLS — add one to enable this cell`],
          fields: {},
        });
        continue;
      }
      const url = `https://${cfg.host}${path}`;
      console.log(`--- ${label}  ->  ${url}`);

      // The last path segment is the listing's unique id slug (rightmove numeric id,
      // foxtons chpk…, KF chq…). If the live final URL no longer contains it, the
      // listing 30x-redirected off the detail page == sold/withdrawn == stale seed.
      const idSlug = (path.split('/').filter(Boolean).pop() || '').toLowerCase();
      const isGone = (fetched) => {
        const st = fetched && fetched.status;
        const chain = (fetched && fetched.redirectStatuses) || [];
        if (st === 404 || st === 410) return true;
        if (chain.some((c) => c === 404 || c === 410)) return true;
        // Redirected off the detail path (final URL dropped the id slug).
        if (fetched && fetched.finalUrl && idSlug &&
            !fetched.finalUrl.toLowerCase().includes(idSlug)) return true;
        return false;
      };

      let cell;
      try {
        const fetched = fetchLiveWithRetry(url, label);
        const status = fetched && fetched.status;
        if (!fetched || !fetched.ok || !fetched.html) {
          // Could not get a page at all after retries. Distinguish "listing gone"
          // (soft-skip — 404/410/redirect off the detail path) from a hard fetch
          // failure (timeout / blocked) that still warrants a look.
          const gone = isGone(fetched);
          cell = {
            site, mode,
            status: gone ? 'SKIP' : 'FAIL',
            reasons: [
              gone
                ? `live url unreachable as a detail page (status ${status}, final '${fetched && fetched.finalUrl}') — ` +
                  `listing likely sold/withdrawn; REFRESH the ${label} url in LIVE_URLS`
                : `live fetch failed after ${RETRIES} retries: ${(fetched && fetched.err) || 'unknown'} (status ${status})`,
            ],
            fields: { url, status, finalUrl: fetched && fetched.finalUrl },
          };
        } else if (isGone(fetched)) {
          // We GOT html, but it is a generic landing page the site 30x-redirected us
          // to because the listing is sold/withdrawn (e.g. KF 404 -> /residential
          // landing, HTTP 200, no __NEXT_DATA__). This is a STALE SEED, not live-DOM
          // drift — SKIP it so it does not red the weekly run, and flag for url refresh.
          cell = {
            site, mode, status: 'SKIP',
            reasons: [
              `listing redirected off the detail page (status ${status}, redirect-chain ` +
              `${JSON.stringify((fetched.redirectStatuses) || [])}, final '${fetched.finalUrl}') — ` +
              `sold/withdrawn; REFRESH the ${label} url in LIVE_URLS`,
            ],
            fields: { url, status, finalUrl: fetched.finalUrl },
          };
        } else {
          const { ex } = extractFromLive(fetched.html, {
            host: cfg.host, path, url, pageModelJson: fetched.pageModel,
          });
          cell = evaluateCell(site, mode, ex, fetched.pageModel, domExcerpt(fetched.html), url);
          cell.fields.status = status;
          if (cell.status === 'FAIL') cell.fields.domExcerpt = domExcerpt(fetched.html);
        }
      } catch (e) {
        // Per-cell guard: an unexpected throw here must NOT abort the whole run.
        cell = { site, mode, status: 'FAIL', reasons: [`harness threw: ${e && e.stack ? e.stack.split('\n')[0] : e}`], fields: { url } };
      }

      const tag = cell.status === 'PASS' ? 'PASS' : cell.status === 'SKIP' ? 'SKIP' : 'FAIL';
      console.log(`    ${tag}  price=${cell.fields.price} beds=${cell.fields.beds} ` +
        `sqft=${cell.fields.sqft} postcode=${cell.fields.postcode} ` +
        `type=${cell.fields.propertyType} tenure=${cell.fields.tenure} ` +
        `popup=${cell.fields.popupWouldRender}`);
      for (const r of cell.reasons) console.log(`        - ${r}`);
      results.push(cell);
    }
  }

  // --- Per-site collapse rule: if BOTH modes of a site SKIP, that site produced
  //     no live signal at all — treat as a soft FAIL so the issue prompts a url
  //     refresh (a permanently-dead pair is itself drift in our url list). ------
  const bySite = {};
  for (const r of results) (bySite[r.site] = bySite[r.site] || []).push(r);
  for (const [site, cells] of Object.entries(bySite)) {
    // Only count cells that were actually SEEDED + fetched. An unseeded mode is an
    // intentional gap, not a live-signal failure, so it neither counts toward nor
    // triggers the collapse rule.
    const real = cells.filter((c) => c.mode !== '-' && !c.unseeded);
    if (real.length && real.every((c) => c.status === 'SKIP')) {
      for (const c of real) {
        c.status = 'FAIL';
        c.reasons.push(`ALL modes for ${site} soft-skipped — no live signal; refresh both LIVE_URLS entries`);
      }
    }
  }

  // --- Summary matrix ----------------------------------------------------------
  console.log('\n================ HEADLESS SMOKE SUMMARY ================');
  const fails = results.filter((r) => r.status === 'FAIL');
  const skips = results.filter((r) => r.status === 'SKIP');
  for (const r of results) {
    console.log(`  ${r.status.padEnd(4)} ${r.site}/${r.mode}` +
      (r.status !== 'PASS' ? `  — ${r.reasons[0] || ''}` : ''));
  }
  console.log(`\n  ${results.length} cells: ` +
    `${results.filter((r) => r.status === 'PASS').length} PASS, ` +
    `${fails.length} FAIL, ${skips.length} SKIP`);

  if (EMIT_JSON) {
    // Machine-readable block the workflow greps out for the issue body.
    console.log('\n<<<SMOKE_JSON>>>');
    console.log(JSON.stringify({
      generatedAt: new Date().toISOString(),
      contentBytes: SRC.length,
      results: results.map((r) => ({
        site: r.site, mode: r.mode, status: r.status,
        reasons: r.reasons, fields: r.fields,
      })),
    }));
    console.log('<<<END_SMOKE_JSON>>>');
  }

  if (fails.length) {
    console.log(`\n  ${fails.length} live-extraction drift FAIL(s) — the extension would mis-extract or`);
    console.log('  not render on at least one live site. See per-cell reasons above.');
    process.exit(1);
  }
  console.log('\n  All live cells extracted plausible data + would render. No extension drift.');
  process.exit(0);
}

run();
