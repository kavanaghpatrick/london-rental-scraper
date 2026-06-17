# Debug: Rent Fair Value plugin doesn't load on a Chestertons lettings page

**Repro URL:** https://www.chestertons.co.uk/properties/21844781/lettings/CJL260150
**Symptom (user):** the "Rent Fair Value" plugin doesn't load / show its UI on this page.

## What's already RULED OUT (lead scoped this — start from here, don't re-derive)
The extension SHOULD inject + activate on this URL — so this is NOT a "not injected" bug, it's a RUNTIME failure inside content.js. Confirmed:
- **manifest.json** content_scripts `matches: ['https://www.chestertons.co.uk/properties/*', ...]` → MATCHES this URL. js loaded: vendor/tesseract.min.js, xgboost.js, content.js.
- **host_permissions** include `https://*.chestertons.co.uk/*`, `https://*.homeflow-assets.co.uk/*` (chestertons asset host), `https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/*` (model artifacts), and the vercel dashboard.
- **content.js detectSite()**: `hostname.includes('chestertons.co.uk')` → SITES.CHESTERTONS ✓.
- **content.js isPropertyDetailPage()** for CHESTERTONS: `/\/properties\/\d+\/lettings\//.test(url)` → MATCHES `/properties/21844781/lettings/CJL260150` ✓.
- The page is a **React SPA** (initial HTML has react markers + price/PW/Price text present in the server payload).
- Chestertons extraction entrypoint: content.js `extractPropertyDataChestertons()` (~line 460); SPA handling via a URL-change watcher + mutation observer (~lines 75-152); floorplan tab-click logic for CHESTERTONS (~line 265).

## The 4 runtime-cause hypotheses to investigate (each owner takes one)
1. **DOM/selector drift (TOP suspect):** does `extractPropertyDataChestertons()`'s selectors match THIS page's actual rendered DOM? Chestertons may have changed markup, or this property/page-type differs. Fetch the live page, compare the selectors content.js queries (price, beds, baths, sqft, postcode/address, property type, floorplan) against what's actually present. If extraction returns null/throws, the UI never renders.
2. **Model/artifact load failure:** xgboost.js fetches model.json + features.json from raw.githubusercontent.com (and maybe similar_listings/predictions). If that fetch fails (404/availability/parse), the predictor never initializes → no UI. Check the fetch URLs, that the artifacts exist + parse, and the load/init path + error handling in xgboost.js + content.js.
3. **SPA timing / re-injection:** the content script runs at document_idle, but React may not have rendered the property data yet; the mutation observer / URL-watcher must catch it. Does the flow actually fire extraction after render on a FRESH page load of this URL (not just on in-app navigation)? Look for a race where extraction runs once, finds nothing, and never retries.
4. **Uncaught JS error:** an exception anywhere in the content.js init→detect→extract→predict→render chain (e.g. parseAmenities on empty description, a missing field, a chestertons selector throwing) aborts before the UI renders. Trace the chain for unguarded throws; check the browser-console error path.

## Constraints / notes
- Agents CANNOT run the real Chrome extension in a browser — investigate by reading content.js + xgboost.js, fetching the LIVE page DOM (curl/WebFetch), comparing selectors, and tracing the flow. The user will test the fix in their browser.
- Do NOT break the OTHER sites (rightmove/knightfrank/savills) — chestertons extraction/timing changes must be site-scoped.
- Keep JS↔Python parity intact if xgboost.js is touched (byte-identity guard + _fixture_diff) — but a load-fix likely lives in content.js (extraction/UI), not the predictor core.

## RESOLUTION (task #4 — fix in content.js only)
Root cause was **two stacked bugs**, both reproduced statically (`node _content_load_test.mjs`):

1. **PRIMARY — fatal, site-agnostic crash.** `log()`/`logError()` (lines 23-28) were
   infinitely self-recursive (`log` called `log`, `logError` called `logError`). The
   first top-level `log('Script loaded!')` — outside any try/catch — threw
   `RangeError: Maximum call stack size exceeded`, aborting the whole content script
   before `init()` ran. This broke the plugin on **every** site; Chestertons is just
   what was tested. Fixed: delegate to `console.log`/`console.error` with an `[RFV]` prefix.

2. **CONTRIBUTING — Chestertons SPA initial-render race.** Live HTML confirms
   price/beds/baths/type are NOT in the server payload (only photos + the "Long Let"
   badge are) — React renders them after `document_idle`. The first extraction comes
   back empty and the URL-watcher only re-fires on URL *change*, so a direct hit /
   refresh never retried. Fixed (site-scoped to Chestertons): on empty extraction,
   arm `scheduleSpaRetry()` — a debounced MutationObserver + poll backstop, bounded by
   20 attempts and a 15s deadline — that re-runs `init()` once the listing renders.

Robustness: `extractPropertyDataChestertons()` is now wrapped in try/catch (a single
throwing selector can't abort the UI) and gates on a rendered price ("not ready" →
return null → retry, instead of erroring on a missing price).

Not a cause: artifact/model load (`/api/predict` + GitHub-raw model.json) — it has
graceful fallback and only runs *after* extraction succeeds, so it was downstream of
both bugs. `xgboost.js` untouched; JS↔Python parity stays 0/0, vendored copy byte-identical.
