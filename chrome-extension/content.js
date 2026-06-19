/**
 * Rent Fair Value - Chrome Extension
 * Shows ML-powered fair rent estimates on London rental listings
 * Supports: Rightmove, Knight Frank, Chestertons, Savills
 *
 * Flow:
 * 1. Detect which site we're on
 * 2. Extract property data using site-specific logic
 * 3. Run XGBoost model locally (OCR floorplan if available)
 *
 * DISCLAIMER: This extension provides automated estimates for informational
 * purposes only. Estimates are NOT professional valuations and should not
 * be relied upon for financial decisions. See privacy policy for data handling.
 */

(function() {
  'use strict';

  // Set to false for production builds to reduce console noise
  const DEBUG = true;

  // Debug logging helper - only logs when DEBUG is true.
  // NOTE: these MUST delegate to console.*, not to themselves. A previous version
  // called log()/logError() recursively, which threw "Maximum call stack size
  // exceeded" on the very first log('Script loaded!') below — at top level in this
  // IIFE, outside any try/catch — aborting the entire content script before init()
  // ever ran. That made the plugin fail to load on every site.
  function log(...args) {
    if (DEBUG) console.log('[RentFairValue]', ...args);
  }
  function logError(...args) {
    console.error('[RentFairValue]', ...args); // Always log errors
  }

  log('Script loaded!');

  const CONFIG = {
    PREDICTIONS_URL: 'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/chrome-extension/api/predictions.json',
    SIMILAR_URL: 'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/chrome-extension/api/similar_listings.json',
    MODEL_URL: 'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/chrome-extension/api/model.json',
    FEATURES_URL: 'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/chrome-extension/api/features.json',
    // PRIMARY estimate source: our backend running the canonical v20 model (#28/#29).
    // The in-browser XGBoost path is a LABELED-approximate FALLBACK (it diverges from
    // canonical — see PREDICT_API_CONTRACT.md). On error/timeout we fall back.
    PREDICT_URL: 'https://dashboard-fawn-nu-59.vercel.app/api/predict',
    PREDICT_TIMEOUT: 6000,
    OCR_TIMEOUT: 60000,
  };

  // Site detection
  const SITES = {
    RIGHTMOVE: 'rightmove',
    KNIGHTFRANK: 'knightfrank',
    CHESTERTONS: 'chestertons',
    SAVILLS: 'savills',
    FOXTONS: 'foxtons',
  };

  function detectSite() {
    const hostname = window.location.hostname;
    if (hostname.includes('rightmove.co.uk')) return SITES.RIGHTMOVE;
    if (hostname.includes('knightfrank.co.uk')) return SITES.KNIGHTFRANK;
    if (hostname.includes('chestertons.co.uk')) return SITES.CHESTERTONS;
    if (hostname.includes('savills.com')) return SITES.SAVILLS;
    if (hostname.includes('foxtons.co.uk')) return SITES.FOXTONS;
    return null;
  }

  const currentSite = detectSite();
  log(' Detected site:', currentSite);

  // Prevent duplicate execution
  if (window.__rentFairValueLoaded) return;
  window.__rentFairValueLoaded = true;

  // Caches
  let predictionsCache = null;
  let similarListingsCache = null;
  let xgbPredictor = null;

  // Track current URL for SPA navigation detection
  let lastUrl = window.location.href;
  let isRunning = false;
  // Guards the SPA initial-render retry loop (see scheduleSpaRetry). Declared up
  // here — alongside the other state — because init() runs synchronously below,
  // BEFORE the scheduleSpaRetry definition is reached; a `let` declared next to
  // that function would be in its temporal-dead-zone when init() first calls it.
  let spaRetryActive = false;

  // Main execution
  init();

  // ============================================
  // SPA NAVIGATION DETECTION
  // ============================================
  // Property listing sites (Rightmove, Chestertons, etc.) are Single Page Applications
  // that don't trigger full page reloads when navigating between listings.
  // This requires intercepting History API calls to detect navigation and re-run
  // the valuation for the new property. This is a standard SPA detection technique
  // used by many Chrome extensions and does not modify page behavior beyond
  // triggering our own URL change handler.

  function setupNavigationDetection() {
    // 1. Listen for popstate (back/forward navigation)
    window.addEventListener('popstate', handleUrlChange);

    // 2. Wrap pushState and replaceState to detect programmatic navigation
    // This is necessary because SPAs use these methods to change URLs without page reload
    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;

    history.pushState = function(...args) {
      originalPushState.apply(this, args);
      handleUrlChange();
    };

    history.replaceState = function(...args) {
      originalReplaceState.apply(this, args);
      handleUrlChange();
    };

    // 3. Fallback: Poll for URL changes (catches edge cases)
    setInterval(() => {
      if (window.location.href !== lastUrl) {
        handleUrlChange();
      }
    }, 1000);

    log(' SPA navigation detection enabled');
  }

  function handleUrlChange() {
    const newUrl = window.location.href;
    if (newUrl === lastUrl) return;

    log(' URL changed:', lastUrl, '->', newUrl);
    lastUrl = newUrl;

    // Check if this is a property detail page (not search results)
    if (!isPropertyPage(newUrl)) {
      log(' Not a property page, skipping');
      removeExisting(); // Remove sidebar on non-property pages
      return;
    }

    // Debounce: wait for page content to update
    setTimeout(() => {
      if (!isRunning) {
        log(' Re-running for new property...');
        init();
      }
    }, 1500);
  }

  function isPropertyPage(url) {
    // Check if URL matches property detail page patterns
    switch (currentSite) {
      case SITES.RIGHTMOVE:
        return /\/properties\/\d+/.test(url);
      case SITES.KNIGHTFRANK:
        return /\/properties\//.test(url) && !/\/search/.test(url);
      case SITES.CHESTERTONS:
        return /\/properties\/\d+\/lettings\//.test(url);
      case SITES.SAVILLS:
        return /\/property-detail\//.test(url);
      case SITES.FOXTONS:
        // Detail URLs are /properties-to-rent/{postcode-or-area-slug}/{ref}, where
        // the ref segment is letters+digits (e.g. b2rc5264686, btrc0001663,
        // chpk0478367). Search/area pages are a SINGLE segment (e.g.
        // /properties-to-rent/kensington/) with no digit-bearing ref, so requiring a
        // digit in the final segment excludes them. This self-gating mirrors the other
        // sites (a search page's __NEXT_DATA__ has no propertyDetail key, so the
        // extractor also returns null there — but gating the SPA re-run/retry on the
        // URL keeps us from arming retries on listing-grid pages).
        return /\/properties-to-rent\/[^/]+\/[a-z]*\d[a-z\d]*\/?(?:[?#].*)?$/i.test(url);
      default:
        return false;
    }
  }

  // ============================================
  // SPA INITIAL-RENDER RETRY (safe for all sites)
  // ============================================
  // On a FRESH page load of an agent SPA (Chestertons is the known case), the content
  // script fires at document_idle BEFORE React has rendered the price/beds/etc., so the
  // first extraction comes back empty. The URL-watcher only re-fires on URL *changes*,
  // so without this the plugin would silently never render on a direct hit/refresh.
  // This watches the DOM for the listing to render and re-runs init() once data is
  // found, bounded by both a retry cap and a wall-clock deadline so it can't spin
  // forever. It calls the SAME per-site extractors, so it's a no-op for sites that
  // already succeeded (they never reach the null branch that arms this) and harmless
  // for a site that genuinely has no data (it just times out and disconnects).
  // (spaRetryActive is declared with the other state vars above, to avoid a TDZ
  // error when init() calls this synchronously during initial load.)
  function scheduleSpaRetry() {
    if (spaRetryActive) return; // one retry loop at a time
    spaRetryActive = true;

    const MAX_RETRIES = 20;          // hard cap on re-extraction attempts
    const DEADLINE_MS = 15000;       // give up ~15s after load
    const startedAt = Date.now();
    let attempts = 0;
    let settleTimer = null;
    let poll = null;

    const stop = () => {
      spaRetryActive = false;
      if (settleTimer) clearTimeout(settleTimer);
      if (poll) clearInterval(poll);
      try { observer.disconnect(); } catch (e) {}
    };

    const tryAgain = () => {
      if (!spaRetryActive) return;
      attempts++;
      if (attempts > MAX_RETRIES || Date.now() - startedAt > DEADLINE_MS) {
        log(' SPA retry giving up after', attempts, 'attempts');
        stop();
        return;
      }
      // Only re-run when not already running and the page still looks right.
      if (isRunning || !isPropertyPage(window.location.href)) return;
      if (extractPropertyData()) {
        log(' SPA retry: data is ready, re-running init()');
        stop();
        init();
      }
    };

    // React often renders in bursts; debounce mutations so we extract once it settles.
    const observer = new MutationObserver(() => {
      if (settleTimer) clearTimeout(settleTimer);
      settleTimer = setTimeout(tryAgain, 400);
    });
    observer.observe(document.body, { childList: true, subtree: true });

    // Also poll on a timer as a backstop (covers renders that don't mutate <body>
    // subtree in a way the observer catches, e.g. text-node-only updates).
    poll = setInterval(() => {
      if (!spaRetryActive) { clearInterval(poll); return; }
      tryAgain();
    }, 750);

    log(' SPA initial-render retry armed for', currentSite);
  }

  // Start navigation detection
  setupNavigationDetection();

  async function init() {
    // Prevent concurrent runs
    if (isRunning) {
      log(' Already running, skipping');
      return;
    }
    isRunning = true;

    try {
      // Remove any existing sidebar before starting fresh
      removeExisting();

      // 1. Extract property data from page
      const propertyData = extractPropertyData();
      if (!propertyData) {
        log(' No property data found');
        isRunning = false;
        // SPA sites (Chestertons especially) render the listing AFTER this content
        // script fires at document_idle, so a fresh-load extraction can legitimately
        // come back empty. Schedule a bounded, observer-driven retry instead of
        // giving up — once the listing renders this re-enters init() and renders.
        //
        // SAFE FOR ALL SITES: the retry just re-calls the same per-site extractor.
        // Rightmove/Knight Frank/Savills that already succeeded never reach here;
        // if one legitimately has no data (e.g. a static Rightmove page truly
        // missing __NEXT_DATA__) the retry simply times out within its budget and
        // disconnects — no behavior change, no per-site extractor touched.
        if (isPropertyPage(window.location.href)) {
          scheduleSpaRetry();
        }
        return;
      }

      const propertyId = extractPropertyId();
      log(' Property ID:', propertyId);

      // 2. Check if short-term let - show warning instead of valuation
      const letType = extractLetType(propertyData);
      log(' Let type:', letType);

      if (letType === 'short') {
        log(' Short-term let detected - showing warning');
        injectShortLetWarning(propertyData.prices?.primaryPrice);
        isRunning = false;
        return;
      }

      // 3. Show loading
      injectLoadingState('Loading estimate...');

      // 4. Parse asking price
      const askingPrice = parsePrice(propertyData.prices?.primaryPrice);
      if (!askingPrice) {
        injectError('Could not parse price');
        isRunning = false;
        return;
      }

      // 4. Try cache first (instant) - DISABLED FOR TESTING v0.6.0 fixes
      // const cached = await getCachedPrediction(propertyId);
      // if (cached) {
      //   log(' Cache hit!');
      //   displayResult({
      //     asking_price: askingPrice,
      //     fair_value: cached.fv,
      //     range_low: cached.lo,
      //     range_high: cached.hi,
      //     premium_pct: cached.pct,
      //     size_sqft: cached.sq,
      //     amenities_detected: [],
      //   }, 'cached');
      //   return;
      // }
      log(' Cache disabled - running live prediction');

      // 5. Not cached - run full analysis
      log(' Cache miss, running local model...');
      injectLoadingState('Analyzing property...');

      const result = await analyzeProperty(propertyData, askingPrice);
      if (!result) {
        // analyzeProperty already injected a specific error (e.g. model not loaded).
        isRunning = false;
        return;
      }
      displayResult(result, result.size_source);

      isRunning = false;
    } catch (error) {
      logError(' Error:', error);
      injectError('Something went wrong');
      isRunning = false;
    }
  }

  async function analyzeProperty(propertyData, askingPrice) {
    // FIX C: the shared feature builder (xgboost.js -> window.XGBFeatures) is required
    // by BOTH the server-request path (extractFloors/parseAmenities) and the in-browser
    // fallback (buildFeatures + XGBoostPredictor). If xgboost.js failed to load/parse,
    // these globals are undefined and the first deref below would throw a bare
    // TypeError. Fail fast with a clear message and return null (caller stops cleanly).
    if (!window.XGBFeatures || !window.XGBoostPredictor) {
      logError(' XGBFeatures/XGBoostPredictor not available — xgboost.js failed to load');
      injectError('Model failed to load');
      return null;
    }

    // Extract all available data
    const beds = propertyData.bedrooms || 1;
    const baths = propertyData.bathrooms || 1;
    const postcode = extractPostcode(propertyData);
    const propertyType = extractPropertyType(propertyData);
    const agentName = extractAgentName(propertyData);
    const lat = propertyData.location?.latitude;
    const lon = propertyData.location?.longitude;
    const address = propertyData.address?.displayAddress || '';  // V16: for garden square/prime street detection
    const description = (propertyData.text?.description || '') + ' ' +
                       (propertyData.text?.propertyPhrase || '') +
                       ' ' + (propertyData.keyFeatures || []).join(' ');
    console.log(`[RFV] Extracted: type=${propertyType}, agent=${agentName}, address=${address}`);

    // Get sqft - from page JSON or OCR
    let sizeSqft = extractSqftFromPage(propertyData);
    let sizeSource = sizeSqft ? 'page' : null;
    let ocrText = ''; // Store raw OCR text for floor extraction

    // Run OCR on the floorplan to extract floor flags (and sqft when the page
    // doesn't expose it). For SPA sites (Chestertons, Savills) the floorplan lives
    // in a tab that must be clicked first; retry if needed.
    //
    // EXCEPTION — Foxtons: its floorplan CDN (web-prod-page-assets.foxtons.co.uk,
    // Cloudflare in front of CloudFront) actively blocks the extension's
    // service-worker fetch with HTTP 403 (bot/hotlink mitigation; see
    // scripts/ocr_enrich.py "Foxtons CDN blocks ... use Playwright"). When Foxtons
    // already gives us real sqft from the page (size_source==='page'), OCR can only
    // add floor-flag granularity — not worth a guaranteed-failing fetch that floods
    // the console with 403s. So we skip floorplan OCR on Foxtons when sqft is known.
    // All other hosts (incl. Chestertons, which depends on OCR for sqft) are
    // unchanged. The skip is keyed on the known-hostile host, NOT on size_source
    // generally, so we never weaken OCR where it works.
    let floorplanUrl = getFloorplanUrl(propertyData);
    log(' Initial floorplan URL:', floorplanUrl || 'NOT FOUND');

    // For agent sites, floorplans may be in a tab that needs clicking
    if (!floorplanUrl && (currentSite === SITES.CHESTERTONS || currentSite === SITES.SAVILLS)) {
      log(' Trying to click floorplan tab...');
      injectLoadingState('Looking for floorplan...');

      // Try to click the floorplan tab
      const clicked = await clickFloorplanTab();
      if (clicked) {
        log(' Clicked floorplan tab, waiting for content...');
        // Wait for lazy content to load
        await new Promise(r => setTimeout(r, 2500));
        // Re-extract property data to get floorplan
        const updatedData = extractPropertyData();
        floorplanUrl = getFloorplanUrl(updatedData || propertyData);
        log(' After tab click, floorplan URL:', floorplanUrl || 'STILL NOT FOUND');
      }

      // Retry: wait a bit more and try again (content might still be loading)
      if (!floorplanUrl) {
        log(' Retrying floorplan search after delay...');
        await new Promise(r => setTimeout(r, 2000));
        floorplanUrl = getFloorplanUrl(extractPropertyData() || propertyData);
        log(' Retry result:', floorplanUrl || 'NOT FOUND');
      }
    }

    // Skip floorplan OCR on Foxtons when we already have sqft from the page: the
    // Foxtons CDN 403s the extension fetch, so OCR would only spam errors while
    // adding (at most) floor flags. Only floor-flag granularity is lost, which is
    // acceptable vs. a guaranteed-failing fetch on every analysis.
    const sqftKnownFromPage = sizeSource === 'page';
    const hostBlocksFloorplanFetch = currentSite === SITES.FOXTONS;
    const skipFloorplanOcr = floorplanUrl && hostBlocksFloorplanFetch && sqftKnownFromPage;

    if (floorplanUrl && !skipFloorplanOcr) {
      injectLoadingState('Reading floorplan...');
      // BUG2 GUARD: OCR is best-effort enrichment only — it must NEVER abort the
      // analysis or the comps render. ocrFloorplan already catches its own errors
      // and returns {sqft:null,text:''}, but we wrap the await here too so that even
      // an unexpected rejection that escapes that catch degrades to empty OCR text
      // instead of bubbling to init()'s outer catch (which would injectError and wipe
      // the sidebar + Similar Properties). Comps render independent of OCR outcome.
      let ocrResult = { sqft: null, text: '' };
      try {
        ocrResult = await ocrFloorplan(floorplanUrl);
      } catch (ocrErr) {
        log(' OCR step skipped (unexpected error, analysis continues):',
          (ocrErr && ocrErr.message) || 'unknown');
        ocrResult = { sqft: null, text: '' };
      }
      ocrText = ocrResult.text || '';
      // Only use OCR sqft if we don't have it from page
      if (!sizeSqft && ocrResult.sqft) {
        sizeSqft = ocrResult.sqft;
        sizeSource = 'ocr';
      }
      log(' OCR result: sqft=' + (ocrResult.sqft || 'none') + ', text length=' + ocrText.length);
    } else if (skipFloorplanOcr) {
      // ocrText stays '' -> extractFloors('') returns all-zero flags (safe).
      log(' Skipping floorplan OCR (sqft known from page; this host blocks the floorplan CDN fetch)');
    } else {
      log(' No floorplan found in property data');
    }

    if (!sizeSqft) {
      // Estimate from beds
      sizeSqft = estimateSqft(beds);
      sizeSource = 'estimated';
    }

    // Derive floor flags from OCR floorplan text (same extractor the model uses)
    // so the server gets explicit floors instead of having to re-OCR.
    const floors = window.XGBFeatures.extractFloors(ocrText);

    // RAW property fields = the /api/predict request contract
    // (see PREDICT_API_CONTRACT.md). These are the fields canonical v20 FE reads;
    // property_type_std + source + the floor flags are load-bearing for parity, so
    // send them explicitly. The server passes this straight to buildFeatures().
    const rawFields = {
      bedrooms: beds,
      bathrooms: baths,
      size_sqft: sizeSqft,
      postcode: postcode,
      postcode_normalized: postcode,
      area: '',
      property_type: propertyType,          // raw type (Python field name)
      property_type_std: propertyType,      // canonical type signal (drives FE)
      address: address,
      latitude: lat,
      longitude: lon,
      source: currentSite || '',            // rightmove|knightfrank|chestertons|savills -> source_quality
      agent_brand: agentName,               // premium-agent detection (Python field name)
      let_type: 'long',  // short lets are intercepted in init() before this runs
      features: '',
      description: description,
      ocrText: ocrText,
      pageUrl: window.location.href,
      // explicit floor flags (model consumes these directly)
      floor_count: floors.floor_count,
      has_basement: floors.has_basement,
      has_ground: floors.has_ground,
      has_first_floor: floors.has_first_floor,
      has_second_floor: floors.has_second_floor,
      has_third_floor: floors.has_third_floor,
      has_fourth_plus: floors.has_fourth_plus,
      has_roof_terrace: floors.has_roof_terrace,
    };

    // PRIMARY: ask our backend (canonical v20). On error/timeout, fall back to the
    // in-browser model (labeled approximate).
    injectLoadingState('Calculating fair value...');
    const serverResult = await fetchServerEstimate(rawFields);
    if (serverResult && serverResult.estimate_pcm > 0) {
      const fairValue = Math.round(serverResult.estimate_pcm);
      const premiumPct = Math.round((askingPrice / fairValue - 1) * 100 * 10) / 10;
      const amenities = window.XGBFeatures.parseAmenities(description);
      const amenitiesDetected = Object.entries(amenities).filter(([k, v]) => v).map(([k]) => k.replace('has_', ''));
      return {
        asking_price: askingPrice,
        fair_value: fairValue,
        range_low: Math.round(serverResult.range_low ?? fairValue * 0.79),
        range_high: Math.round(serverResult.range_high ?? fairValue * 1.21),
        premium_pct: premiumPct,
        size_sqft: sizeSqft,
        size_source: sizeSource,
        amenities_detected: amenitiesDetected,
        postcode_district: normalizeDistrict(postcode),  // '' when no real postcode -> comps suppressed
        beds: beds,
        baths: baths,
        predictor_source: 'server',  // drives the UI "v20 · Live" label
      };
    }
    log(' Server estimate unavailable — falling back to in-browser model (approximate)');

    // FALLBACK: in-browser XGBoost (approximate — diverges from canonical v20)
    // Load XGBoost model if needed
    if (!xgbPredictor) {
      injectLoadingState('Loading model...');
      xgbPredictor = new window.XGBoostPredictor();
      await xgbPredictor.load(CONFIG.MODEL_URL, CONFIG.FEATURES_URL);
    }

    // Build features and predict — reuse the SAME rawFields the server gets, so the
    // in-browser fallback now matches canonical v20 too (the JS FE was made byte-equal
    // to Python). Identical inputs => identical features.
    injectLoadingState('Calculating fair value...');
    console.log(`[RFV] Building features with: beds=${beds}, baths=${baths}, sqft=${sizeSqft}, postcode=${postcode}, propertyType=${propertyType}, agent=${agentName}`);
    const features = window.XGBFeatures.buildFeatures(rawFields);
    console.log(`[RFV] Key features: tube_dist=${features.tube_distance_km?.toFixed(3)}, center_dist=${features.center_distance_km?.toFixed(3)}, center_inv=${features.center_distance_inv?.toFixed(4)}, is_prime=${features.is_prime_postcode}`);

    const predLog = xgbPredictor.predict(features);
    const fairValue = Math.round(Math.expm1(predLog));

    const premiumPct = Math.round((askingPrice / fairValue - 1) * 100 * 10) / 10;
    const amenities = window.XGBFeatures.parseAmenities(description);
    const amenitiesDetected = Object.entries(amenities)
      .filter(([k, v]) => v)
      .map(([k]) => k.replace('has_', ''));

    // Extract postcode district for similar properties search ('' when there is
    // no real postcode, which suppresses comps rather than mislocating them).
    const postcodeDistrict = normalizeDistrict(postcode);

    return {
      asking_price: askingPrice,
      fair_value: fairValue,
      range_low: Math.round(fairValue * 0.79),
      range_high: Math.round(fairValue * 1.21),
      premium_pct: premiumPct,
      size_sqft: sizeSqft,
      size_source: sizeSource,
      amenities_detected: amenitiesDetected,
      postcode_district: postcodeDistrict,
      beds: beds,
      baths: baths,
      predictor_source: 'fallback',  // in-browser model; approximate vs canonical v20
    };
  }

  // Ask our backend (canonical v20) for the estimate. Returns the parsed JSON
  // ({estimate_pcm, range_low, range_high, model_version, ...}) or null on any
  // error/timeout so analyzeProperty falls back to the in-browser model.
  async function fetchServerEstimate(rawFields) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CONFIG.PREDICT_TIMEOUT);
    try {
      const res = await fetch(CONFIG.PREDICT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rawFields),
        signal: controller.signal,
      });
      if (!res.ok) {
        log(' /api/predict HTTP ' + res.status);
        return null;
      }
      const data = await res.json();
      if (!data || !(data.estimate_pcm > 0)) {
        log(' /api/predict returned no estimate');
        return null;
      }
      return data;
    } catch (e) {
      log(' /api/predict failed: ' + (e.name === 'AbortError' ? 'timeout' : e.message));
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  // ============================================
  // DATA EXTRACTION - Site-specific routing
  // ============================================

  function extractPropertyData() {
    switch (currentSite) {
      case SITES.RIGHTMOVE:
        return extractPropertyDataRightmove();
      case SITES.KNIGHTFRANK:
        return extractPropertyDataKnightFrank();
      case SITES.CHESTERTONS:
        return extractPropertyDataChestertons();
      case SITES.SAVILLS:
        return extractPropertyDataSavills();
      case SITES.FOXTONS:
        return extractPropertyDataFoxtons();
      default:
        log(' Unknown site, trying Rightmove extraction');
        return extractPropertyDataRightmove();
    }
  }

  // ============================================
  // RIGHTMOVE EXTRACTION
  // ============================================

  function extractPropertyDataRightmove() {
    // Strategy 1: __NEXT_DATA__
    const nextDataScript = document.getElementById('__NEXT_DATA__');
    if (nextDataScript) {
      try {
        const data = JSON.parse(nextDataScript.textContent);
        const propertyData = data?.props?.pageProps?.propertyData;
        if (propertyData) {
          log(' Found via __NEXT_DATA__');
          return propertyData;
        }
      } catch (e) {}
    }

    // Strategy 2: window.PAGE_MODEL (legacy single-underscore embedded blob).
    // Kept for backward-compat: an older Rightmove build (and some cached pages)
    // shipped `window.PAGE_MODEL = { propertyData: {...} }` directly in a <script>.
    for (const script of document.querySelectorAll('script')) {
      const text = script.textContent || '';
      const match = text.match(/window\.PAGE_MODEL\s*=\s*/);
      if (match) {
        try {
          const obj = parseScriptObjectLiteral(text, match.index + match[0].length);
          if (obj && obj.propertyData) {
            log(' Found via PAGE_MODEL');
            return obj.propertyData;
          }
        } catch (e) {}
      }
    }

    // Strategy 3: window.__PAGE_MODEL (CURRENT live Rightmove — DOUBLE underscore).
    // Rightmove now ships a devalue/flatten-encoded blob:
    //   window.__PAGE_MODEL = {"data":"<escaped JSON>","encoding":"on"}
    // where `data` is a STRING of escaped JSON that parses to a FLAT reference-indexed
    // array `arr`. arr[0] is the root index-map ({propertyData: <idx>, ...}); every
    // value inside an object/array node is an INTEGER INDEX into arr. We rebuild the
    // real object via decodeRightmovePageModel() (see its comment for the load-bearing
    // scalar-vs-container subtlety) and return propertyData in the canonical shape the
    // shared helpers already consume.
    //
    // Two sources, in order:
    //   (a) a live runtime global window.__PAGE_MODEL (set post-hydration by client JS);
    //   (b) the static escaped blob embedded in a page <script> (what a fresh load /
    //       a non-browser fetch sees before hydration).
    // (a) is tried first so a future runtime-only format still works; (b) is the path
    // that fires on a normally-rendered page.
    try {
      const runtime = (typeof window !== 'undefined') ? window.__PAGE_MODEL : null;
      if (runtime && typeof runtime === 'object') {
        const pd = decodeRightmovePageModel(runtime);
        if (pd) {
          log(' Found via __PAGE_MODEL (runtime global)');
          return pd;
        }
      }
    } catch (e) {}

    for (const script of document.querySelectorAll('script')) {
      const text = script.textContent || '';
      const match = text.match(/window\.__PAGE_MODEL\s*=\s*/);
      if (match) {
        try {
          const obj = parseScriptObjectLiteral(text, match.index + match[0].length);
          const pd = decodeRightmovePageModel(obj);
          if (pd) {
            log(' Found via __PAGE_MODEL (embedded script)');
            return pd;
          }
        } catch (e) {}
      }
    }

    log(' No Rightmove property data found');
    return null;
  }

  // Brace-match a top-level `{ ... }` object literal starting at `start` in `text`,
  // honouring string quotes + escapes so a `{`/`}` inside a JSON string value (or an
  // escaped quote) does not throw off the depth count, then JSON.parse it.
  // Used by the PAGE_MODEL / __PAGE_MODEL embedded-<script> strategies.
  function parseScriptObjectLiteral(text, start) {
    let depth = 0, i = start, inStr = false, esc = false, quote = '';
    // skip leading whitespace to the opening brace
    while (i < text.length && text[i] !== '{') i++;
    for (; i < text.length; i++) {
      const ch = text[i];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === '\\') esc = true;
        else if (ch === quote) inStr = false;
        continue;
      }
      if (ch === '"' || ch === "'") { inStr = true; quote = ch; continue; }
      if (ch === '{') depth++;
      else if (ch === '}') { if (--depth === 0) { i++; break; } }
    }
    const slice = text.slice(start, i);
    const objStart = slice.indexOf('{');
    return JSON.parse(objStart === -1 ? slice : slice.slice(objStart));
  }

  // Decode Rightmove's devalue/flatten-encoded __PAGE_MODEL into the canonical
  // propertyData object. `model` is {data:"<escaped JSON>", encoding:...} (the live
  // shape) OR an already-decoded {propertyData:{...}} (defensive). Returns the
  // propertyData object or null.
  //
  // FLATTEN FORMAT: model.data parses to a flat array `arr`. arr[0] is the root
  // index-map. Inside any object/array node, a numeric value is an INDEX into arr.
  //
  // THE LOAD-BEARING SUBTLETY: dereferencing is SINGLE-HOP per slot and TYPE-AWARE.
  // resolveSlot(idx) inspects arr[idx]: if it is a CONTAINER (object/array) we rebuild
  // it, resolving each child value as an index; if it is a SCALAR (string/number/bool/
  // null) we return it LITERALLY and DO NOT index again. This is essential because
  // scalar leaves are themselves small integers — e.g. bedrooms references slot 258 and
  // arr[258] === 2; naive "every number is an index" recursion would then read arr[2]
  // (the property id) and corrupt the value. sizings carries multiple unit rows
  // (ha/sqft/sqm/ac) with their sizes likewise hoisted to slots.
  function decodeRightmovePageModel(model) {
    if (!model || typeof model !== 'object') return null;

    // Already-decoded shape (legacy / defensive): {propertyData:{...}} not flattened.
    if (model.propertyData && typeof model.propertyData === 'object'
        && !Array.isArray(model.propertyData) && model.data === undefined) {
      return model.propertyData;
    }

    let arr;
    try {
      arr = typeof model.data === 'string' ? JSON.parse(model.data) : model.data;
    } catch (e) {
      return null;
    }
    if (!Array.isArray(arr) || arr.length === 0) return null;

    const root = arr[0];
    if (!root || typeof root !== 'object' || typeof root.propertyData !== 'number') {
      return null;
    }

    const cache = new Array(arr.length);

    function resolveValue(v) {
      if (typeof v === 'number' && Number.isInteger(v) && v >= 0 && v < arr.length) {
        return resolveSlot(v);
      }
      if (Array.isArray(v)) return v.map(resolveValue);
      if (v && typeof v === 'object') {
        const o = {};
        for (const k of Object.keys(v)) o[k] = resolveValue(v[k]);
        return o;
      }
      return v;
    }

    function resolveSlot(idx) {
      if (cache[idx] !== undefined) return cache[idx];
      const node = arr[idx];
      let out;
      if (Array.isArray(node)) {
        out = [];
        cache[idx] = out;          // set before recursing => cycle-safe
        for (const child of node) out.push(resolveValue(child));
      } else if (node && typeof node === 'object') {
        out = {};
        cache[idx] = out;
        for (const k of Object.keys(node)) out[k] = resolveValue(node[k]);
      } else {
        out = node;                // scalar leaf — literal, never re-indexed
        cache[idx] = out;
      }
      return out;
    }

    const pd = resolveSlot(root.propertyData);
    return (pd && typeof pd === 'object') ? pd : null;
  }

  // ============================================
  // FOXTONS EXTRACTION
  // ============================================
  // Foxtons detail pages are server-rendered Next.js with a clean single-property JSON
  // blob at <script id="__NEXT_DATA__"> -> props.pageProps.propertyDetail (a FLAT
  // object). This mirrors the Rightmove embedded-__NEXT_DATA__ path; we read the blob
  // and BUILD the canonical propertyData-shaped object that the shared normalization
  // helpers (extractPostcode / extractPropertyType / extractAgentName /
  // extractSqftFromPage / parsePrice / extractLetType) already consume.
  //
  // SELF-GATING: on a Foxtons search/area page (which the manifest glob also matches)
  // there is NO propertyDetail key, so this returns null and no popup shows — same
  // gating the other sites rely on. Delisted refs 301-redirect to the area search page
  // (URL gains a #gone fragment) where propertyDetail is likewise absent -> null.
  //
  // PRICE (verified live): pd.pricePcm is the correct per-calendar-month figure
  // (rendered "£12,740 pcm"); pd.priceFrom/pd.priceTo are the WEEKLY figure
  // (2,940 -> "£2,940 pw"). So when pricePcm > 0 we surface "£{pricePcm} pcm". On the
  // rare price-RANGE listing pricePcm is 0 — there we fall back to the weekly priceFrom
  // and surface "£{priceFrom} pw" so parsePrice() applies the ×52/12 pw->pcm conversion
  // (e.g. 3,000 pw -> £13,000 pcm, matching the rendered figure). FLOORAREA is ALREADY
  // square feet (the foxtons spider uses it verbatim as size_sqft; jsonLd unitCode
  // "MTK" is mislabeled — ignore it).
  function extractPropertyDataFoxtons() {
    const nextDataScript = document.getElementById('__NEXT_DATA__');
    if (!nextDataScript) {
      log(' Foxtons: no __NEXT_DATA__ script');
      return null;
    }

    let pd;
    try {
      const json = JSON.parse(nextDataScript.textContent);
      pd = json?.props?.pageProps?.propertyDetail;
    } catch (e) {
      logError(' Foxtons __NEXT_DATA__ parse failed:', e);
      return null;
    }

    // No propertyDetail => search/area page or delisted-redirect. No popup.
    if (!pd) {
      log(' Foxtons: no propertyDetail (search page or delisted)');
      return null;
    }

    const data = { _source: 'foxtons' };

    // Bedrooms / bathrooms / receptions (ints; 0/undefined left to caller defaults)
    if (typeof pd.bedrooms === 'number') data.bedrooms = pd.bedrooms;
    if (typeof pd.bathrooms === 'number') data.bathrooms = pd.bathrooms;
    if (typeof pd.receptions === 'number') data.receptions = pd.receptions;

    // Size — floorArea is ALREADY sqft. set size_source='page' via the sizings array
    // (extractSqftFromPage reads sizings; analyzeProperty marks sizeSource='page' when
    // sqft is present). Guard against null/0/garbage so we fall back to estimateSqft.
    if (typeof pd.floorArea === 'number' && pd.floorArea > 0) {
      const sqft = Math.round(pd.floorArea);
      if (sqft >= 100 && sqft <= 50000) {
        data.sizings = [{ minimumSize: sqft, unit: 'sqft' }];
      }
    }

    // Price — build the visible-style string parsePrice() expects.
    if (typeof pd.pricePcm === 'number' && pd.pricePcm > 0) {
      data.prices = { primaryPrice: `£${pd.pricePcm.toLocaleString('en-GB')} pcm` };
    } else if (typeof pd.priceFrom === 'number' && pd.priceFrom > 0) {
      // pricePcm unreliable (0) on price-range listings; priceFrom is the WEEKLY figure.
      // Surface as "£X pw" -> parsePrice does ×52/12 to recover the pcm value.
      data.prices = { primaryPrice: `£${pd.priceFrom.toLocaleString('en-GB')} pw` };
    }

    // Postcode -> outcode/incode for extractPostcode(). Prefer the full postcode.
    const fullPostcode = (pd.postcode && typeof pd.postcode === 'object' && pd.postcode.name)
      ? pd.postcode.name
      : (typeof pd.postcode === 'string' ? pd.postcode : (pd.postcodeShort || ''));

    // Address — build displayAddress from the structured fields.
    const addrObj = (pd.address && typeof pd.address === 'object') ? pd.address : {};
    const line1 = pd.addressLine1 || addrObj.addressLine1 || '';
    const line2 = pd.addressLine2 || addrObj.addressLine2 || addrObj.streetName || pd.street || '';
    const town = pd.town || 'London';
    const displayParts = [line1, line2, town].filter(p => p && String(p).trim());
    data.address = { displayAddress: displayParts.join(', ') };

    if (fullPostcode) {
      const pc = String(fullPostcode).trim();
      const parts = pc.split(/\s+/);
      data.address.outcode = parts[0] || '';
      data.address.incode = parts[1] || '';
      // Ensure the postcode is also discoverable in the free-text address as a fallback.
      if (data.address.displayAddress && !data.address.displayAddress.toUpperCase().includes(parts[0].toUpperCase())) {
        data.address.displayAddress = `${data.address.displayAddress}, ${pc}`;
      }
    }

    // Property type — pd.type is the specific lowercase string ('flat'/'penthouse'/
    // 'house'); '' falls through to extractPropertyType()'s default ('flat').
    if (typeof pd.type === 'string' && pd.type.trim()) {
      data.propertyType = pd.type.trim().toLowerCase();
    }

    // Agent — always Foxtons.
    data.customer = { companyName: 'Foxtons' };

    // Lat/lng for the predictor's location features (pd.lat/pd.lng preferred).
    const lat = (typeof pd.lat === 'number') ? pd.lat : (typeof pd.latitude === 'number' ? pd.latitude : undefined);
    const lng = (typeof pd.lng === 'number') ? pd.lng : (typeof pd.longitude === 'number' ? pd.longitude : undefined);
    if (typeof lat === 'number' && typeof lng === 'number') {
      data.location = { latitude: lat, longitude: lng };
    }

    // Floorplan — pd.floorplan.src lives on web-prod-page-assets.foxtons.co.uk; often
    // "" (no floorplan). Guard so getFloorplanUrl() doesn't return an empty string.
    const fpSrc = (pd.floorplan && typeof pd.floorplan === 'object') ? pd.floorplan.src : '';
    if (fpSrc && String(fpSrc).trim()) {
      data.floorplans = [{ url: String(fpSrc).trim() }];
    }

    // Description / key features for short-let detection + amenity parsing.
    if (typeof pd.description === 'string' && pd.description.trim()) {
      data.text = { description: pd.description.trim() };
    }
    if (Array.isArray(pd.features) && pd.features.length) {
      data.keyFeatures = pd.features.filter(f => typeof f === 'string');
    } else if (Array.isArray(pd.keyFeatures) && pd.keyFeatures.length) {
      data.keyFeatures = pd.keyFeatures.filter(f => typeof f === 'string');
    }

    log(' Foxtons extracted:', data);
    // Require at least a price OR beds so a half-empty blob doesn't render garbage.
    return (data.prices?.primaryPrice || data.bedrooms) ? data : null;
  }

  // ============================================
  // KNIGHT FRANK EXTRACTION
  // ============================================

  function extractPropertyDataKnightFrank() {
    log(' Extracting Knight Frank data from DOM');
    const data = { _source: 'knightfrank' };

    // Get main content text for regex extraction
    const mainContent = document.querySelector('main, [role="main"], .property-details, article') || document.body;
    const pageText = mainContent.innerText || '';

    // Address - from page title or h1
    const titleEl = document.querySelector('h1.kf-pdp-hero__title, h1[class*="title"], .property-address h1, h1');
    if (titleEl) {
      data.address = { displayAddress: titleEl.textContent.trim() };
    } else {
      const pageTitle = document.title.replace(/\s*\|.*$/, '').trim();
      data.address = { displayAddress: pageTitle };
    }

    // Price - use regex on page text (more reliable)
    const priceMatch = pageText.match(/£([\d,]+)\s*(?:pcm|pw|per\s*(?:calendar\s*)?month|per\s*week|monthly|weekly)?/i);
    if (priceMatch) {
      data.prices = { primaryPrice: priceMatch[0] };
      log(' Knight Frank price found:', priceMatch[0]);
    } else {
      // Fallback to DOM selector
      const priceEl = document.querySelector('.kf-pdp-hero__price, .property-price, [class*="price"]');
      if (priceEl) {
        data.prices = { primaryPrice: priceEl.textContent.trim() };
      }
    }

    // Bedrooms/Bathrooms - regex on page text
    const bedsMatch = pageText.match(/(\d+)\s*(?:bed(?:room)?s?)/i);
    if (bedsMatch) {
      data.bedrooms = parseInt(bedsMatch[1], 10);
    }
    const bathsMatch = pageText.match(/(\d+)\s*(?:bath(?:room)?s?)/i);
    if (bathsMatch) {
      data.bathrooms = parseInt(bathsMatch[1], 10);
    }
    const receptionsMatch = pageText.match(/(\d+)\s*(?:reception)/i);
    if (receptionsMatch) {
      data.receptions = parseInt(receptionsMatch[1], 10);
    }

    // Size in sqft - search in main content only
    const sizeMatch = pageText.match(/(\d{1,5}(?:,\d{3})?)\s*(?:sq\.?\s*ft|sqft|square\s*feet)/i);
    if (sizeMatch) {
      const sqft = parseInt(sizeMatch[1].replace(/,/g, ''), 10);
      if (sqft >= 100 && sqft <= 50000) {
        data.sizings = [{ minimumSize: sqft, unit: 'sqft' }];
      }
    }

    // Postcode from address - FIXED split bug
    // KF addresses are OUTCODE-ONLY ("...Chelsea, London, SW3"): they carry no incode.
    // First try a FULL postcode (outcode + incode); if none is present, fall back to an
    // outcode-only match anchored at the END of the address so we still set
    // data.address.outcode (the spider's own parse_card_data also stores outcode-only
    // for KF, so this matches canonical behaviour). Without this fallback the full-only
    // regex never matched KF's "...London, SW3" and outcode was left unset.
    if (data.address?.displayAddress) {
      const addr = data.address.displayAddress;
      const pcMatch = addr.match(/([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})/i);
      if (pcMatch) {
        const parts = pcMatch[1].split(/\s+/);
        data.address.outcode = parts[0] || '';
        data.address.incode = parts[1] || '';
      } else {
        const ocMatch = addr.match(/,\s*([A-Z]{1,2}\d{1,2}[A-Z]?)\s*$/i)
          || addr.match(/\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*$/i);
        if (ocMatch) {
          data.address.outcode = ocMatch[1].toUpperCase();
          data.address.incode = '';
        }
      }
    }

    // Property type from page text
    const textLower = pageText.toLowerCase();
    if (textLower.includes('penthouse')) data.propertyType = 'penthouse';
    else if (textLower.includes('studio')) data.propertyType = 'studio';
    else if (textLower.includes('house')) data.propertyType = 'house';
    else if (textLower.includes('maisonette')) data.propertyType = 'maisonette';
    else if (textLower.includes('apartment')) data.propertyType = 'apartment';
    else data.propertyType = 'flat';

    // Agent name
    data.customer = { companyName: 'Knight Frank' };

    // Floorplan URL - Knight Frank CDN (content.knightfrank.com)
    // Spider checks: 1) anchor tags with "Floorplan" text, 2) images, 3) data-src for lazy loading

    // 1. Check anchor tags first (spider pattern)
    const floorplanLinks = document.querySelectorAll('a[href*="floorplan"], a[href*="Floorplan"]');
    for (const link of floorplanLinks) {
      const href = link.href;
      if (href && (href.includes('.jpg') || href.includes('.png') || href.includes('.jpeg') || href.includes('content.knightfrank.com'))) {
        data.floorplans = [{ url: href }];
        break;
      }
    }

    // 2. Check images with src or data-src
    if (!data.floorplans) {
      const floorplanImg = document.querySelector(
        'img[src*="content.knightfrank.com"][src*="floorplan"], ' +
        'img[data-src*="content.knightfrank.com"][data-src*="floorplan"], ' +
        'img[src*="floorplan"], img[data-src*="floorplan"]'
      );
      if (floorplanImg) {
        const url = floorplanImg.src || floorplanImg.dataset.src || floorplanImg.getAttribute('data-src');
        if (url) data.floorplans = [{ url }];
      }
    }

    // 3. Regex fallback on page HTML
    if (!data.floorplans) {
      const floorplanMatch = document.body.innerHTML.match(/https:\/\/content\.knightfrank\.com\/[^"'\s]*(?:floorplan|floor-plan)[^"'\s]*\.(?:jpg|png|jpeg)/i);
      if (floorplanMatch) {
        data.floorplans = [{ url: floorplanMatch[0] }];
      }
    }

    // Description
    const descEl = document.querySelector('.kf-pdp-description, .property-description, [class*="description"]');
    if (descEl) {
      data.text = { description: descEl.textContent.trim() };
    }

    // Key features
    const keyFeatures = [];
    document.querySelectorAll('.kf-pdp-features li, .key-feature, [class*="feature"] li').forEach(li => {
      keyFeatures.push(li.textContent.trim());
    });
    if (keyFeatures.length > 0) data.keyFeatures = keyFeatures;

    log(' Knight Frank extracted:', data);
    return Object.keys(data).length > 2 ? data : null;
  }

  // ============================================
  // CHESTERTONS EXTRACTION
  // ============================================

  // FIX #4: Subject-anchored extraction for Chestertons.
  // Chestertons is a Next.js app; the listing data lives in the RSC "flight" stream
  // (self.__next_f.push([...]) chunks inside <script> tags). The page often contains
  // MANY property objects (the similar-listings carousel), so a first-match-wins
  // innerText regex (e.g. /\d+\s*bed/) can latch onto the WRONG unit — that is exactly
  // how the trigger listing (a 3-bed) was mis-read as a 1-bed.
  //
  // We instead select the SUBJECT property object by the NUMERIC property id from the
  // URL (/properties/<NUMID>/lettings/<REF>). The stream carries the subject as
  // "property":{"id":<NUMID>,...} and that same object holds bedrooms / bathrooms /
  // propertyType / postcode / squareFeetInternal. NOTE (from verification): the URL
  // <REF> is NOT a usable anchor — "reference":"<REF>" is not present as a field, and
  // <REF> appears 100s of times (in photo URLs etc.). Only the numeric id is reliable.
  //
  // Content scripts run in an ISOLATED world and cannot read the page's `self.__next_f`
  // directly, so we scan the serialized <script> tag text instead. The flight payload is
  // string-escaped (\" / \\), so we unescape a copy before searching.
  // Returns { bedrooms, bathrooms, propertyType, postcode, squareFeetInternal } (any
  // field may be undefined/null) or null if the subject object can't be found.
  function extractChestertonsSubjectFromFlight() {
    try {
      // Numeric property id from the URL: /properties/<NUMID>/lettings/<REF>
      const idMatch = window.location.pathname.match(/\/properties\/(\d+)\/lettings\//);
      if (!idMatch) {
        log(' Chestertons flight: no numeric property id in URL');
        return null;
      }
      const numId = idMatch[1];

      // Gather all flight-stream script text. self.__next_f.push(...) chunks only.
      let combined = '';
      for (const script of document.querySelectorAll('script')) {
        const t = script.textContent || '';
        if (t.indexOf('__next_f') !== -1) combined += t + '\n';
      }
      if (!combined) {
        log(' Chestertons flight: no __next_f script chunks found');
        return null;
      }

      // The payload is string-escaped inside the push() argument. Unescape a copy so
      // the JSON object delimiters ({ } " :) are real. Order matters: \\" -> " first,
      // then \\\\ -> \  (do NOT reorder, or escaped backslashes corrupt the quotes).
      const text = combined.replace(/\\"/g, '"').replace(/\\\\/g, '\\');

      // Anchor on the SUBJECT: "property":{"id":<NUMID>
      const anchor = '"property":{"id":' + numId;
      const at = text.indexOf(anchor);
      if (at === -1) {
        log(' Chestertons flight: subject anchor not found for id ' + numId);
        return null;
      }

      // Extract the balanced { ... } object that starts right after "property":
      const objStart = at + '"property":'.length; // points at '{'
      let depth = 0, inStr = false, esc = false, end = -1;
      for (let k = objStart; k < text.length; k++) {
        const c = text[k];
        if (inStr) {
          if (esc) esc = false;
          else if (c === '\\') esc = true;
          else if (c === '"') inStr = false;
        } else {
          if (c === '"') inStr = true;
          else if (c === '{') depth++;
          else if (c === '}') {
            depth--;
            if (depth === 0) { end = k + 1; break; }
          }
        }
      }
      if (end === -1) {
        log(' Chestertons flight: unbalanced subject object');
        return null;
      }

      const objStr = text.slice(objStart, end);
      let obj;
      try {
        obj = JSON.parse(objStr);
      } catch (e) {
        // Fall back to field-level regex on the (subject-scoped) object string so a
        // single malformed nested value doesn't lose the whole extraction.
        log(' Chestertons flight: JSON.parse failed, using field regex: ' + e);
        const num = (f) => {
          const m = objStr.match(new RegExp('"' + f + '":\\s*(\\d+)'));
          return m ? parseInt(m[1], 10) : undefined;
        };
        const str = (f) => {
          const m = objStr.match(new RegExp('"' + f + '":\\s*"([^"]*)"'));
          return m ? m[1] : undefined;
        };
        // First URL inside the subject-scoped objStr's floorplans:[...] array. objStr is
        // already bounded to THIS subject, so the first match is the subject's own plan
        // (never a carousel comp). The capture [^"\\] stops at the first quote OR
        // backslash, so a residual trailing escape from the stream is excluded rather
        // than captured (a backslash-suffixed URL would 404 on fetch); the closing-quote
        // anchor is dropped so an optional trailing backslash before the quote can't make
        // the whole match fail.
        const arr = (f) => {
          const m = objStr.match(new RegExp('"' + f + '":\\s*\\[\\s*"([^"\\\\]+)'));
          return m ? m[1].replace(/\\+$/, '') : undefined;
        };
        return {
          bedrooms: num('bedrooms'),
          bathrooms: num('bathrooms'),
          propertyType: str('propertyType'),
          postcode: str('postcode'),
          squareFeetInternal: num('squareFeetInternal'),
          floorplanUrl: arr('floorplans'),
        };
      }

      // The subject object also carries a floorplans:[<url>,...] array of direct
      // homeflow-assets.co.uk /files/floorplan/ URLs. For the SIZE-UNKNOWN listings the
      // subject's squareFeetInternal is null, so OCR of this plan is the ONLY way to
      // recover a real sqft. Surface floorplans[0] here (already unescaped + anchored to
      // the subject id, so it is guaranteed the subject's own plan, not a carousel comp).
      let floorplanUrl;
      if (Array.isArray(obj.floorplans)) {
        const first = obj.floorplans.find(
          (u) => typeof u === 'string' && u.trim().toLowerCase().startsWith('http'));
        if (first) floorplanUrl = first.trim().replace(/\\+$/, '');
      }

      return {
        bedrooms: typeof obj.bedrooms === 'number' ? obj.bedrooms : undefined,
        bathrooms: typeof obj.bathrooms === 'number' ? obj.bathrooms : undefined,
        propertyType: typeof obj.propertyType === 'string' ? obj.propertyType : undefined,
        postcode: typeof obj.postcode === 'string' ? obj.postcode : undefined,
        // squareFeetInternal is often null for the subject (only the carousel comps
        // tend to carry it); read it ONLY from this subject object, never globally.
        squareFeetInternal: (typeof obj.squareFeetInternal === 'number' && obj.squareFeetInternal > 0)
          ? obj.squareFeetInternal : undefined,
        floorplanUrl,
      };
    } catch (e) {
      logError(' Chestertons flight extraction threw:', e);
      return null;
    }
  }

  function extractPropertyDataChestertons() {
    // Chestertons is a React SPA: the price/beds/baths/type are NOT in the initial
    // server HTML — they render client-side after the content script fires at
    // document_idle. So (a) guard the whole extraction in try/catch — a single
    // throwing selector must not abort the UI render — and (b) treat "no price" as
    // "not rendered yet" and return null, so the SPA retry logic re-fires extraction
    // once React has populated the DOM (rather than emitting a half-baked object).
    try {
      return extractPropertyDataChestertonsImpl();
    } catch (e) {
      logError(' Chestertons extraction threw:', e);
      return null;
    }
  }

  function extractPropertyDataChestertonsImpl() {
    log(' Extracting Chestertons data from DOM');
    const data = { _source: 'chestertons' };

    // Get main content text for regex extraction (like spider does)
    const mainContent = document.querySelector('main, [role="main"], .property-details, article') || document.body;
    const pageText = mainContent.innerText || '';

    // Try to find JSON data in scripts first
    for (const script of document.querySelectorAll('script[type="application/ld+json"]')) {
      try {
        const json = JSON.parse(script.textContent);
        if (json['@type'] === 'RealEstateListing' || json['@type'] === 'Apartment' || json['@type'] === 'House') {
          if (json.name) data.address = { displayAddress: json.name };
          if (json.address?.streetAddress) data.address = { displayAddress: json.address.streetAddress };
          break;
        }
      } catch (e) {}
    }

    // Address from page - try h1 first (most reliable)
    if (!data.address) {
      const h1El = document.querySelector('h1');
      if (h1El) {
        data.address = { displayAddress: h1El.textContent.trim() };
      }
    }
    // Fallback to specific selectors
    if (!data.address) {
      const addressEl = document.querySelector('.property-details__address, .property-address, [class*="address"]');
      if (addressEl) {
        data.address = { displayAddress: addressEl.textContent.trim() };
      }
    }
    // Final fallback to title
    if (!data.address) {
      const pageTitle = document.title.replace(/\s*-.*$/, '').replace(/\s*\|.*$/, '').trim();
      data.address = { displayAddress: pageTitle };
    }

    // Price - use regex on page text (like spider does) - CRITICAL FIX
    // Look for £X,XXX pattern followed by pcm/pw/month/week
    const priceMatch = pageText.match(/£([\d,]+)\s*(?:pcm|pw|per\s*(?:calendar\s*)?month|per\s*week|monthly|weekly)?/i);
    if (priceMatch) {
      const priceText = priceMatch[0];
      data.prices = { primaryPrice: priceText };
      log(' Chestertons price found:', priceText);
    } else {
      // Fallback: try any £X,XXX pattern
      const anyPriceMatch = pageText.match(/£([\d,]+)/);
      if (anyPriceMatch) {
        // Check context for period indicator
        const priceIndex = pageText.indexOf(anyPriceMatch[0]);
        const context = pageText.substring(priceIndex, priceIndex + 50).toLowerCase();
        const period = context.includes('pw') || context.includes('week') ? 'pw' : 'pcm';
        data.prices = { primaryPrice: `${anyPriceMatch[0]} ${period}` };
        log(' Chestertons price fallback:', data.prices.primaryPrice);
      }
    }

    // FIX #4: Prefer the SUBJECT property object from the RSC flight stream, anchored on
    // the numeric URL id. This avoids the first-match-wins innerText regexes below
    // latching onto the WRONG unit on multi-unit / similar-listings carousel pages
    // (the trigger 3-bed was mis-read as a 1-bed that way). Falls back to the regexes
    // when the flight stream is unavailable (older render path / structure change).
    const subject = extractChestertonsSubjectFromFlight();
    if (subject) {
      log(' Chestertons subject (flight): beds=' + subject.bedrooms +
          ', baths=' + subject.bathrooms + ', type=' + subject.propertyType +
          ', sqft=' + subject.squareFeetInternal + ', postcode=' + subject.postcode);
    }

    // Bedrooms/Bathrooms - subject object first, else regex on page text.
    if (subject && Number.isFinite(subject.bedrooms)) {
      data.bedrooms = subject.bedrooms;
    } else {
      const bedsMatch = pageText.match(/(\d+)\s*(?:bed(?:room)?s?)/i);
      if (bedsMatch) {
        data.bedrooms = parseInt(bedsMatch[1], 10);
      }
    }
    if (subject && Number.isFinite(subject.bathrooms)) {
      data.bathrooms = subject.bathrooms;
    } else {
      const bathsMatch = pageText.match(/(\d+)\s*(?:bath(?:room)?s?)/i);
      if (bathsMatch) {
        data.bathrooms = parseInt(bathsMatch[1], 10);
      }
    }

    // Size - read sqft ONLY from the subject object (its squareFeetInternal). Do NOT
    // scrape squareFeetInternal globally: the page carries ~18 of them and only ~1 is
    // the subject's (the rest are carousel comps) — wrong-property hazard. The subject's
    // value is often null; when it is, leave data.sizings unset so analyzeProperty falls
    // back to estimateSqft and flags size_source='estimated' (FIX #1 then neutralizes the
    // verdict). The old innerText /sqft/ regex is kept ONLY as a last resort when the
    // flight stream is entirely unavailable.
    if (subject && Number.isFinite(subject.squareFeetInternal) && subject.squareFeetInternal > 0) {
      data.sizings = [{ minimumSize: subject.squareFeetInternal, unit: 'sqft' }];
    } else if (!subject) {
      const sizeMatch = pageText.match(/(\d{1,5}(?:,\d{3})?)\s*(?:sq\.?\s*ft|sqft|square\s*feet)/i);
      if (sizeMatch) {
        const sqft = parseInt(sizeMatch[1].replace(/,/g, ''), 10);
        if (sqft >= 100 && sqft <= 50000) { // Validate reasonable range
          data.sizings = [{ minimumSize: sqft, unit: 'sqft' }];
        }
      }
    }

    // Postcode - subject object first (full postcode incl. incode), else parse from address.
    if (subject && subject.postcode) {
      data.address = data.address || {};
      const parts = subject.postcode.trim().split(/\s+/);
      data.address.outcode = parts[0] || '';
      data.address.incode = parts[1] || '';
      data.address.postcode = subject.postcode.trim();
    } else if (data.address?.displayAddress) {
      const pcMatch = data.address.displayAddress.match(/([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})/i);
      if (pcMatch) {
        const parts = pcMatch[1].split(/\s+/);
        data.address.outcode = parts[0] || '';
        data.address.incode = parts[1] || '';
      }
    }

    // Agent name
    data.customer = { companyName: 'Chestertons' };

    // Property type - subject object first (FIX #4), else infer from page text.
    if (subject && subject.propertyType) {
      data.propertyType = subject.propertyType;
    } else {
      const textLower = pageText.toLowerCase();
      if (textLower.includes('penthouse')) data.propertyType = 'penthouse';
      else if (textLower.includes('studio')) data.propertyType = 'studio';
      else if (textLower.includes('house')) data.propertyType = 'house';
      else if (textLower.includes('maisonette')) data.propertyType = 'maisonette';
      else if (textLower.includes('apartment')) data.propertyType = 'apartment';
      else data.propertyType = 'flat';
    }

    // Floorplan - Chestertons uses homeflow-assets CDN with /files/floorplan/ path.
    // PREFER the SUBJECT object's floorplans[0] from the RSC flight stream. At
    // document_idle the live PDP has ZERO rendered floorplan <img> tags (the Floorplans
    // tab is a lazy React skeleton), and the only place the subject plan URL exists is
    // inside the escaped __next_f stream — so the DOM selectors / innerHTML regex below
    // either miss it or (the regex) capture it with a trailing backslash that 404s on
    // fetch. The subject URL is already unescaped + anchored to the numeric URL id, so it
    // is guaranteed THIS listing's own plan (not a carousel comp). This is what lets the
    // existing ocrFloorplan() path fire and recover real sqft (size_source='ocr') instead
    // of falling back to estimateSqft -> "SIZE UNKNOWN".
    if (subject && subject.floorplanUrl) {
      data.floorplans = [{ url: subject.floorplanUrl }];
    }
    // DOM fallbacks (older render path / structure change) — only when the flight stream
    // didn't yield a URL. Match spider pattern (look for /files/floorplan/ path, not just
    // domain) and check data-src for lazy-loaded images.
    if (!data.floorplans) {
      const floorplanImg = document.querySelector(
        'img[src*="/files/floorplan/"], img[data-src*="/files/floorplan/"], ' +
        'img[src*="floorplan"], img[data-src*="floorplan"]'
      );
      if (floorplanImg) {
        const url = floorplanImg.src || floorplanImg.dataset.src || floorplanImg.getAttribute('data-src');
        if (url) data.floorplans = [{ url }];
      }
    }
    if (!data.floorplans) {
      // Try to find in page HTML - use spider's exact pattern. [^"\s\\] excludes the
      // escaped-stream's trailing backslash so the URL doesn't 404 on fetch.
      const floorplanMatch = document.body.innerHTML.match(/https:\/\/[^"\s\\]+\/files\/floorplan\/[^"\s\\]+/i);
      if (floorplanMatch) {
        data.floorplans = [{ url: floorplanMatch[0] }];
      }
    }

    // Description - find largest text block
    const descEl = document.querySelector('.property-description, [class*="description"], .overview, article p');
    if (descEl) {
      data.text = { description: descEl.textContent.trim() };
    }

    log(' Chestertons extracted:', data);
    // Readiness gate: the React SPA renders price last. Require a price before we
    // consider the page "ready" — otherwise return null so the caller's bounded
    // retry waits for the next render tick instead of erroring on a missing price.
    if (!data.prices?.primaryPrice) {
      log(' Chestertons price not rendered yet — treating as not-ready');
      return null;
    }
    return data;
  }

  // ============================================
  // SAVILLS EXTRACTION
  // ============================================

  function extractPropertyDataSavills() {
    log(' Extracting Savills data from DOM');
    const data = { _source: 'savills' };

    // Get main content text for regex extraction
    const mainContent = document.querySelector('main, [role="main"], .property-details, article') || document.body;
    const pageText = mainContent.innerText || '';

    // Try JSON-LD first
    for (const script of document.querySelectorAll('script[type="application/ld+json"]')) {
      try {
        const json = JSON.parse(script.textContent);
        if (json['@type'] === 'RealEstateListing' || json['@type'] === 'Apartment') {
          if (json.name) data.address = { displayAddress: json.name };
          if (json.address?.streetAddress) data.address = { displayAddress: json.address.streetAddress };
          break;
        }
      } catch (e) {}
    }

    // Address - try h1 first
    if (!data.address) {
      const h1El = document.querySelector('h1');
      if (h1El) {
        data.address = { displayAddress: h1El.textContent.trim() };
      }
    }
    if (!data.address) {
      const addressEl = document.querySelector('.sv-property-header__address, .property-address, [class*="address"]');
      if (addressEl) {
        data.address = { displayAddress: addressEl.textContent.trim() };
      }
    }
    if (!data.address) {
      const pageTitle = document.title.replace(/\s*-.*$/, '').replace(/\s*\|.*$/, '').trim();
      data.address = { displayAddress: pageTitle };
    }

    // Price - use regex on page text (more reliable)
    const priceMatch = pageText.match(/£([\d,]+)\s*(?:pcm|pw|per\s*(?:calendar\s*)?month|per\s*week|monthly|weekly)?/i);
    if (priceMatch) {
      data.prices = { primaryPrice: priceMatch[0] };
      log(' Savills price found:', priceMatch[0]);
    } else {
      // Fallback to DOM selector
      const priceEl = document.querySelector('.sv-property-header__price, .sv-pdp-hero__price, .property-price, [class*="price"]');
      if (priceEl) {
        data.prices = { primaryPrice: priceEl.textContent.trim() };
      }
    }

    // Bedrooms/Bathrooms - regex on page text
    const bedsMatch = pageText.match(/(\d+)\s*(?:bed(?:room)?s?)/i);
    if (bedsMatch) {
      data.bedrooms = parseInt(bedsMatch[1], 10);
    }
    const bathsMatch = pageText.match(/(\d+)\s*(?:bath(?:room)?s?)/i);
    if (bathsMatch) {
      data.bathrooms = parseInt(bathsMatch[1], 10);
    }
    const receptionsMatch = pageText.match(/(\d+)\s*(?:reception)/i);
    if (receptionsMatch) {
      data.receptions = parseInt(receptionsMatch[1], 10);
    }

    // Size - Savills usually has good sqft data
    const sizeMatch = pageText.match(/(\d{1,5}(?:,\d{3})?)\s*(?:sq\.?\s*ft|sqft|square\s*feet)/i);
    if (sizeMatch) {
      const sqft = parseInt(sizeMatch[1].replace(/,/g, ''), 10);
      if (sqft >= 100 && sqft <= 50000) {
        data.sizings = [{ minimumSize: sqft, unit: 'sqft' }];
      }
    }
    // Also try sqm
    if (!data.sizings) {
      const sqmMatch = pageText.match(/(\d{1,5}(?:,\d{3})?)\s*(?:sq\.?\s*m|sqm|m²)/i);
      if (sqmMatch) {
        const sqm = parseInt(sqmMatch[1].replace(/,/g, ''), 10);
        if (sqm >= 10 && sqm <= 5000) {
          data.sizings = [{ minimumSize: Math.round(sqm * 10.764), unit: 'sqft' }];
        }
      }
    }

    // Postcode from address - FIXED split bug
    if (data.address?.displayAddress) {
      const pcMatch = data.address.displayAddress.match(/([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})/i);
      if (pcMatch) {
        const parts = pcMatch[1].split(/\s+/);
        data.address.outcode = parts[0] || '';
        data.address.incode = parts[1] || '';
      }
    }

    // Agent name
    data.customer = { companyName: 'Savills' };

    // Property type from page text (scoped to main content)
    const textLower = pageText.toLowerCase();
    if (textLower.includes('penthouse')) data.propertyType = 'penthouse';
    else if (textLower.includes('studio')) data.propertyType = 'studio';
    else if (textLower.includes('house')) data.propertyType = 'house';
    else if (textLower.includes('maisonette')) data.propertyType = 'maisonette';
    else if (textLower.includes('apartment')) data.propertyType = 'apartment';
    else data.propertyType = 'flat';

    // Floorplan - Savills uses CDN images
    // Spider clicks "Plans" tab first - we can't easily do that, so check multiple selectors
    // Also check data-src for lazy-loaded images

    // 1. Check tabs container for Plans content (if already visible)
    const plansTab = document.querySelector('[data-tab="plans"], [data-type="floorplan"], .sv-pdp-floorplan');
    if (plansTab) {
      const img = plansTab.querySelector('img[src], img[data-src]');
      if (img) {
        const url = img.src || img.dataset.src || img.getAttribute('data-src');
        if (url) data.floorplans = [{ url }];
      }
    }

    // 2. Check for floorplan images with src or data-src
    if (!data.floorplans) {
      const floorplanImg = document.querySelector(
        'img[src*="floorplan"], img[data-src*="floorplan"], ' +
        'img[src*="floor-plan"], img[data-src*="floor-plan"], ' +
        'img[alt*="floorplan" i], img[alt*="floor plan" i]'
      );
      if (floorplanImg) {
        const url = floorplanImg.src || floorplanImg.dataset.src || floorplanImg.getAttribute('data-src');
        if (url) data.floorplans = [{ url }];
      }
    }

    // 3. Check anchor tags
    if (!data.floorplans) {
      const floorplanLink = document.querySelector('a[href*="floorplan"], a[href*="floor-plan"]');
      if (floorplanLink?.href) {
        data.floorplans = [{ url: floorplanLink.href }];
      }
    }

    // 4. Regex fallback - Savills CDN pattern
    if (!data.floorplans) {
      const floorplanMatch = document.body.innerHTML.match(/https:\/\/[^"'\s]*savills[^"'\s]*(?:floorplan|floor-plan|_fp)[^"'\s]*\.(?:jpg|png|jpeg)/i);
      if (floorplanMatch) {
        data.floorplans = [{ url: floorplanMatch[0] }];
      }
    }

    // Description
    const descEl = document.querySelector('.sv-property-description, .sv-pdp-description, [class*="description"]');
    if (descEl) {
      data.text = { description: descEl.textContent.trim() };
    }

    // Key features
    const keyFeatures = [];
    document.querySelectorAll('.sv-property-features li, .sv-pdp-features li, [class*="feature"] li').forEach(li => {
      keyFeatures.push(li.textContent.trim());
    });
    if (keyFeatures.length > 0) data.keyFeatures = keyFeatures;

    log(' Savills extracted:', data);
    return Object.keys(data).length > 1 ? data : null;
  }

  function extractPropertyId() {
    const url = window.location.href;
    const pathname = window.location.pathname;

    switch (currentSite) {
      case SITES.RIGHTMOVE: {
        const match = pathname.match(/\/properties\/(\d+)/);
        return match ? match[1] : null;
      }
      case SITES.KNIGHTFRANK: {
        // URL like: /properties/residential/to-let/london/abc123xyz
        const match = pathname.match(/\/properties\/.*\/([a-zA-Z0-9-]+)\/?$/);
        return match ? match[1] : null;
      }
      case SITES.CHESTERTONS: {
        // URL like: /properties/21142524/lettings/KNL220048
        const match = pathname.match(/\/properties\/(\d+)\/lettings\/([a-zA-Z0-9]+)/);
        return match ? `${match[1]}_${match[2]}` : null;
      }
      case SITES.SAVILLS: {
        // URL like: /property-detail/abc123xyz
        const match = pathname.match(/\/property-detail\/([a-zA-Z0-9-]+)/);
        return match ? match[1] : null;
      }
      case SITES.FOXTONS: {
        // URL like: /properties-to-rent/SW7/b2rc5264686 — the final ref segment is
        // the stable id (letters+digits, e.g. b2rc5264686 / btrc0001663 / chpk0478367).
        const match = pathname.match(/\/properties-to-rent\/[^/]+\/([a-zA-Z]*\d[a-zA-Z\d]*)\/?$/);
        return match ? match[1] : null;
      }
      default:
        return null;
    }
  }

  function extractPostcode(data) {
    if (data.address?.outcode) {
      return data.address.outcode + (data.address.incode ? ' ' + data.address.incode : '');
    }
    // Pull a postcode out of the free-text address. UK postcodes sit at the END
    // of an address, so prefer the LAST match and anchor to a valid outward code
    // (district) so "A12 Building" doesn't read as a postcode. Match an optional
    // inward code too (digit + 2 letters), e.g. "SW10 9HD".
    const addr = data.address?.displayAddress || '';
    const re = /\b([A-Z]{1,2}[0-9][0-9A-Z]?)(\s*[0-9][A-Z]{2})?\b/gi;
    let last = null, m;
    while ((m = re.exec(addr)) !== null) last = m;
    if (last) {
      const outward = last[1].toUpperCase();
      return outward + (last[2] ? ' ' + last[2].trim().toUpperCase() : '');
    }
    // No real postcode found. Return null (NOT a silent 'SW3' default) so similar
    // comps are SUPPRESSED rather than mislocated to Chelsea — the call site at
    // displayResult guards `if (r.postcode_district && r.beds)`. The model's own
    // missing-postcode handling lives in xgboost.js (which defaults to SW3 for the
    // prediction only); that is intentionally left untouched.
    return null;
  }

  function extractLetType(data) {
    // Extract let type from propertyData
    // Returns 'short' for short-term lets, 'long' otherwise
    // Works across all supported sites

    // 1. Rightmove-specific: Check lettings.letType field
    if (data.lettings?.letType) {
      const letType = data.lettings.letType.toLowerCase();
      if (letType.includes('short')) return 'short';
    }

    // 2. Check channel field (sometimes indicates short let)
    if (data.channel?.toLowerCase().includes('short')) return 'short';

    // 3. Check description/property phrase for short let keywords
    // Works for all sites
    const textToCheck = [
      data.text?.description || '',
      data.text?.propertyPhrase || '',
      data.listingUpdate?.listingUpdateReason || '',
      ...(data.keyFeatures || [])
    ].join(' ').toLowerCase();

    if (textToCheck.includes('short let') ||
        textToCheck.includes('short-let') ||
        textToCheck.includes('short term') ||
        textToCheck.includes('short-term') ||
        textToCheck.includes('serviced apartment') ||
        textToCheck.includes('serviced accommodation') ||
        textToCheck.includes('holiday let') ||
        textToCheck.includes('corporate let') ||
        textToCheck.includes('minimum 1 month') ||
        textToCheck.includes('minimum one month') ||
        textToCheck.includes('min 1 month')) {
      return 'short';
    }

    // 4. Check site-specific let type indicators (NOT full page text - causes false positives)
    if (currentSite === SITES.CHESTERTONS) {
      // Chestertons uses a .bg-primary badge with "Long Let" or "Short Let" text
      const letBadge = document.querySelector('.bg-primary, [class*="let-type"], [class*="lettings-type"]');
      if (letBadge) {
        const badgeText = letBadge.textContent.toLowerCase();
        if (badgeText.includes('short')) return 'short';
      }
    } else if (currentSite === SITES.KNIGHTFRANK || currentSite === SITES.SAVILLS) {
      // For other agent sites, check only listing description elements (not full page)
      const descriptionEls = document.querySelectorAll(
        '.property-description, [class*="description"], .kf-pdp-description, .sv-property-description'
      );
      for (const el of descriptionEls) {
        const descText = el.textContent.toLowerCase();
        if (descText.includes('short let') ||
            descText.includes('short-term') ||
            descText.includes('serviced apartment') ||
            descText.includes('corporate let')) {
          return 'short';
        }
      }
    }

    // 5. Check URL (but only for explicit short-let paths, not navigation)
    const urlPath = window.location.pathname.toLowerCase();
    if (urlPath.includes('short-let') || urlPath.includes('short_let')) {
      return 'short';
    }

    return 'long';
  }

  async function clickFloorplanTab() {
    // Click the floorplan/floor plans tab to reveal lazy-loaded floorplan content
    // This is required for Chestertons and Savills which hide floorplans in tabs
    // Matches spider pattern: click tab with text containing "floor" + "plan"

    try {
      // Find and click tab/button with floorplan text
      const tabSelectors = 'button, [role="tab"], a, .tab, [class*="tab"], nav a, .nav-link';
      const tabs = document.querySelectorAll(tabSelectors);

      for (const tab of tabs) {
        const text = (tab.innerText || tab.textContent || '').toLowerCase().trim();
        // Match "Floorplans", "Floor Plans", "Floorplan", "Floor Plan"
        if ((text.includes('floor') && text.includes('plan')) || text === 'floorplan' || text === 'floorplans') {
          log(' Found floorplan tab:', text);
          tab.click();
          return true;
        }
      }

      // Also try clicking by aria-label or data attributes
      const ariaTab = document.querySelector(
        '[aria-label*="floorplan" i], [aria-label*="floor plan" i], ' +
        '[data-tab*="floorplan" i], [data-tab*="floor" i], ' +
        '[data-target*="floorplan" i]'
      );
      if (ariaTab) {
        log(' Found floorplan tab via aria/data attribute');
        ariaTab.click();
        return true;
      }

      log(' No floorplan tab found');
      return false;
    } catch (e) {
      logError(' Error clicking floorplan tab:', e);
      return false;
    }
  }

  function extractPropertyType(data) {
    // Try multiple sources for property type
    // 1. Direct propertySubType field (most specific)
    if (data.propertySubType) {
      return data.propertySubType.toLowerCase();
    }
    // 2. propertyType field
    if (data.propertyType) {
      return data.propertyType.toLowerCase();
    }
    // 3. From text/propertyPhrase
    if (data.text?.propertyPhrase) {
      const phrase = data.text.propertyPhrase.toLowerCase();
      if (phrase.includes('penthouse')) return 'penthouse';
      if (phrase.includes('studio')) return 'studio';
      if (phrase.includes('maisonette')) return 'maisonette';
      if (phrase.includes('house')) return 'house';
      if (phrase.includes('apartment')) return 'apartment';
      if (phrase.includes('flat')) return 'flat';
    }
    // 4. From listing update reason (Rightmove)
    if (data.listingUpdate?.listingUpdateReason) {
      const reason = data.listingUpdate.listingUpdateReason.toLowerCase();
      if (reason.includes('penthouse')) return 'penthouse';
      if (reason.includes('studio')) return 'studio';
    }
    // 5. For agent sites, try page title or description
    if (currentSite !== SITES.RIGHTMOVE) {
      const checkText = (document.title + ' ' + (data.text?.description || '')).toLowerCase();
      if (checkText.includes('penthouse')) return 'penthouse';
      if (checkText.includes('studio')) return 'studio';
      if (checkText.includes('maisonette')) return 'maisonette';
      if (checkText.includes('house')) return 'house';
      if (checkText.includes('mews')) return 'house';
      if (checkText.includes('apartment')) return 'apartment';
    }
    // Default to flat
    return 'flat';
  }

  function extractAgentName(data) {
    // Try multiple sources for agent name
    // 1. From customer/branchDisplayName
    if (data.customer?.branchDisplayName) {
      return data.customer.branchDisplayName;
    }
    // 2. From customer/companyName
    if (data.customer?.companyName) {
      return data.customer.companyName;
    }
    // 3. From contactInfo
    if (data.contactInfo?.companyName) {
      return data.contactInfo.companyName;
    }
    // 4. From lettingInformation/agentName
    if (data.lettingInformation?.agentName) {
      return data.lettingInformation.agentName;
    }
    // 5. Fallback based on detected site
    switch (currentSite) {
      case SITES.KNIGHTFRANK: return 'Knight Frank';
      case SITES.CHESTERTONS: return 'Chestertons';
      case SITES.SAVILLS: return 'Savills';
      case SITES.FOXTONS: return 'Foxtons';
      default: return '';
    }
  }

  function extractSqftFromPage(data) {
    // Check sizings array
    const sizings = data.sizings || [];
    for (const s of sizings) {
      if (s.unit === 'sqft') {
        return parseInt(s.minimumSize || s.maximumSize, 10);
      }
      if (s.unit === 'sqm') {
        return Math.round(parseInt(s.minimumSize || s.maximumSize, 10) * 10.764);
      }
    }
    return null;
  }

  function getFloorplanUrl(data) {
    // Helper to extract best URL from img element, preferring data-src for lazy loading
    // and avoiding placeholder/data URLs
    function getBestImgUrl(img) {
      if (!img) return null;
      const dataSrc = img.dataset?.src || img.getAttribute('data-src');
      const src = img.getAttribute('src') || img.src;

      // Prefer data-src if it contains floorplan pattern (lazy-loaded actual URL)
      if (dataSrc && (dataSrc.includes('floorplan') || dataSrc.includes('/files/'))) {
        return dataSrc;
      }

      // Use src only if it's a real URL (not placeholder/data URI)
      if (src && !src.startsWith('data:') && !src.includes('placeholder') && !src.includes('loading')) {
        // Check if src has floorplan pattern
        if (src.includes('floorplan') || src.includes('/files/')) {
          return src;
        }
      }

      // Fallback: return whichever exists
      return dataSrc || (src && !src.startsWith('data:') ? src : null);
    }

    // Check floorplans array (works for all sites)
    const floorplans = data.floorplans || [];
    if (floorplans.length > 0) {
      return floorplans[0].url || floorplans[0].srcUrl;
    }

    // Check media array (Rightmove)
    const media = data.media || [];
    for (const m of media) {
      if (m.type === 'floorplan' || (m.url && m.url.includes('_FLP_'))) {
        return m.url || m.srcUrl;
      }
    }

    // For agent sites, try to find floorplan in DOM if not in data
    if (currentSite !== SITES.RIGHTMOVE) {
      // Site-specific patterns first
      let imgSelector = '';
      let linkSelector = '';

      switch (currentSite) {
        case SITES.CHESTERTONS:
          // Also check for homeflow-assets domain
          imgSelector = 'img[src*="/files/floorplan/"], img[data-src*="/files/floorplan/"], ' +
                       'img[src*="homeflow-assets"][src*="floorplan"], img[data-src*="homeflow-assets"]';
          break;
        case SITES.KNIGHTFRANK:
          imgSelector = 'img[src*="content.knightfrank.com"][src*="floorplan"], img[data-src*="content.knightfrank.com"]';
          linkSelector = 'a[href*="floorplan"]';
          break;
        case SITES.SAVILLS:
          imgSelector = '.sv-pdp-floorplan img, [data-type="floorplan"] img, [data-tab="plans"] img';
          break;
        case SITES.FOXTONS:
          // Foxtons floorplan (when present) is in pd.floorplan.src on
          // web-prod-page-assets.foxtons.co.uk and is already surfaced via
          // data.floorplans above; this DOM fallback covers the rare case where the
          // blob carried it but our array build missed it.
          imgSelector = 'img[src*="web-prod-page-assets.foxtons.co.uk"][src*="floorplan"], ' +
                       'img[src*="foxtons.co.uk"][src*="floorplan"], ' +
                       'img[data-src*="web-prod-page-assets.foxtons.co.uk"]';
          break;
      }

      // Check site-specific selectors first
      if (imgSelector) {
        const siteImg = document.querySelector(imgSelector);
        const url = getBestImgUrl(siteImg);
        if (url) return url;
      }

      if (linkSelector) {
        const siteLink = document.querySelector(linkSelector);
        if (siteLink?.href) return siteLink.href;
      }

      // Generic fallback - check for floorplan images (including data-src for lazy loading)
      const floorplanImg = document.querySelector(
        'img[src*="floorplan"], img[src*="floor-plan"], img[src*="Floorplan"], ' +
        'img[data-src*="floorplan"], img[data-src*="floor-plan"], ' +
        'img[alt*="floorplan" i], img[alt*="floor plan" i], ' +
        '.floorplan img, [class*="floorplan"] img, [data-type="floorplan"] img'
      );
      if (floorplanImg) {
        const url = getBestImgUrl(floorplanImg);
        if (url) return url;
      }

      // Check for floorplan links
      const floorplanLink = document.querySelector(
        'a[href*="floorplan"], a[href*="floor-plan"], ' +
        '[class*="floorplan"] a, [data-type="floorplan"] a'
      );
      if (floorplanLink && floorplanLink.href) {
        return floorplanLink.href;
      }

      // Final fallback - regex search in page HTML for common CDN patterns.
      // [^"\s\\] (not [^"\s]) so a URL sitting inside an escaped RSC flight chunk isn't
      // captured with its trailing backslash — that corrupted URL 404s on fetch.
      const htmlContent = document.body.innerHTML;
      const patterns = [
        /https:\/\/[^"\s\\]+\/files\/floorplan\/[^"\s\\]+/i,  // Chestertons
        /https:\/\/content\.knightfrank\.com\/[^"\s\\]*floorplan[^"\s\\]*\.(?:jpg|png|jpeg)/i,  // Knight Frank
        /https:\/\/[^"\s\\]*savills[^"\s\\]*(?:floorplan|floor-plan|_fp)[^"\s\\]*\.(?:jpg|png|jpeg)/i,  // Savills
      ];
      for (const pattern of patterns) {
        const match = htmlContent.match(pattern);
        if (match) return match[0];
      }
    }

    return null;
  }

  function estimateSqft(beds) {
    const sizes = { 0: 350, 1: 500, 2: 750, 3: 1000, 4: 1300, 5: 1600 };
    return sizes[Math.min(beds, 5)] || 500;
  }

  // ============================================
  // OCR
  // ============================================

  async function ocrFloorplan(url) {
    // Returns { sqft: number|null, text: string } - text is used for floor extraction
    if (typeof Tesseract === 'undefined') {
      logError(' Tesseract not loaded! Check vendor/tesseract.min.js');
      return { sqft: null, text: '' };
    }
    log(' Tesseract available, starting OCR...');

    let worker = null;
    try {
      log(' Running OCR on:', url);

      // Fetch image via background service worker to bypass CORS
      injectLoadingState('Fetching floorplan...');
      const imgData = await new Promise((resolve, reject) => {
        chrome.runtime.sendMessage(
          { action: 'fetchImage', url: url },
          response => {
            if (chrome.runtime.lastError) {
              reject(new Error(chrome.runtime.lastError.message));
            } else if (response && response.success) {
              resolve(response.data);
            } else {
              reject(new Error(response?.error || 'Unknown fetch error'));
            }
          }
        );
      });
      log(' Image fetched via background worker, data length:', imgData.length);

      // Create worker explicitly to ensure proper cleanup (fixes memory leak)
      worker = await Tesseract.createWorker('eng', 1, {
        logger: m => {
          if (m.status === 'recognizing text') {
            injectLoadingState(`Reading floorplan... ${Math.round(m.progress * 100)}%`);
          }
        }
      });

      const result = await Promise.race([
        worker.recognize(imgData),
        new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), CONFIG.OCR_TIMEOUT))
      ]);

      const text = result.data.text;
      log(' OCR result:', text.substring(0, 200));

      // Extract sqft - try sqft patterns first
      const sqftPatterns = [
        /(\d{1,4}(?:,\d{3})?)\s*(?:sq\.?\s*ft|sqft|square\s*feet)/i,
        /(\d{1,4}(?:,\d{3})?)\s*ft²/i,
        /total[:\s]+(\d{1,4}(?:,\d{3})?)\s*(?:sq\s*ft|sqft)/i,
        /approx[:\s]+(\d{1,4}(?:,\d{3})?)\s*(?:sq|ft)/i,
      ];

      for (const p of sqftPatterns) {
        const match = text.match(p);
        if (match) {
          const sqft = parseInt(match[1].replace(',', ''), 10);
          if (sqft >= 100 && sqft <= 15000) {
            log(' Found sqft via OCR:', sqft);
            return { sqft, text };
          }
        }
      }

      // Try sqm patterns (convert to sqft)
      const sqmPatterns = [
        /(\d{1,4}(?:,\d{3})?)\s*(?:sq\.?\s*m|sqm|square\s*m|m²)/i,
        /(\d{1,4}(?:,\d{3})?)\s*m²/i,
        /total[:\s]+(\d{1,4}(?:,\d{3})?)\s*(?:sq\s*m|sqm|m)/i,
      ];

      for (const p of sqmPatterns) {
        const match = text.match(p);
        if (match) {
          const sqm = parseInt(match[1].replace(',', ''), 10);
          if (sqm >= 10 && sqm <= 1500) {
            const sqft = Math.round(sqm * 10.764);
            log(' Found sqm via OCR:', sqm, '-> sqft:', sqft);
            return { sqft, text };
          }
        }
      }

      log(' No size pattern found in OCR text');
      return { sqft: null, text };
    } catch (e) {
      // GRACEFUL + NON-SPAMMY: a blocked/failed floorplan fetch (e.g. Foxtons CDN
      // 403) is expected and recoverable — the page sqft / estimateSqft fallback
      // already covers size, and extractFloors('') is safe. Log a SINGLE debug-level
      // line (never console.error, so it can't flood the console as an error), and
      // guard against an undefined e.message tail ("OCR failed: undefined"). The
      // clean return below guarantees this never aborts analyzeProperty or the
      // comps render.
      log(' OCR unavailable (floorplan fetch/OCR failed):', (e && e.message) || 'fetch blocked');
      return { sqft: null, text: '' };
    } finally {
      // Always terminate worker to prevent memory leak
      if (worker) {
        try {
          await worker.terminate();
          log(' Tesseract worker terminated');
        } catch (termErr) {
          console.warn('[RFV] Worker termination failed:', termErr.message);
        }
      }
    }
  }

  // ============================================
  // CACHE
  // ============================================

  function parsePrice(text) {
    if (!text) return null;

    // CRITICAL FIX: Extract FIRST price only (handles "£500 pw (£2,166 pcm)" cases)
    // Look for £X,XXX pattern - extract first match only
    const priceMatch = text.match(/£([\d,]+(?:\.\d{2})?)/);
    if (!priceMatch) {
      // Fallback: try to find any number sequence
      const numMatch = text.match(/(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)/);
      if (!numMatch) return null;
      const num = parseFloat(numMatch[1].replace(/,/g, ''));
      if (isNaN(num) || num <= 0) return null;
      // Apply weekly conversion if needed
      if (/pw|per week|weekly/i.test(text)) {
        return Math.round(num * 52 / 12);
      }
      return Math.round(num);
    }

    // Parse the matched price (remove commas)
    const price = parseFloat(priceMatch[1].replace(/,/g, ''));
    if (isNaN(price) || price <= 0) return null;

    // Check if this is a weekly price - look at context around the matched price
    const priceIndex = text.indexOf(priceMatch[0]);
    const contextAfter = text.substring(priceIndex, priceIndex + 30).toLowerCase();

    if (/pw|per\s*week|weekly/i.test(contextAfter)) {
      // Convert weekly to monthly: price * 52 / 12
      return Math.round(price * 52 / 12);
    }

    return Math.round(price);
  }

  async function getCachedPrediction(propertyId) {
    if (!propertyId) return null;
    try {
      if (!predictionsCache) {
        const res = await fetch(CONFIG.PREDICTIONS_URL);
        if (res.ok) {
          predictionsCache = await res.json();
          log(' Cache loaded:', Object.keys(predictionsCache).length);
        }
      }
      // Use site-specific cache key
      const cacheKey = `${currentSite}:${propertyId}`;
      return predictionsCache?.[cacheKey] || null;
    } catch (e) {
      logError(' Cache load failed:', e);
      return null;
    }
  }

  // ============================================
  // SIMILAR PROPERTIES
  // ============================================

  async function loadSimilarListings() {
    if (similarListingsCache) return similarListingsCache;
    try {
      const res = await fetch(CONFIG.SIMILAR_URL);
      if (res.ok) {
        similarListingsCache = await res.json();
        log(' Similar listings loaded:', Object.keys(similarListingsCache).length);
      }
    } catch (e) {
      logError(' Similar listings load failed:', e);
    }
    return similarListingsCache || {};
  }

  // Normalize a postcode (or already-extracted district) down to its outward
  // district code, e.g. "SW10 9HD" -> "SW10", "NW16BU" (no space) -> "NW1",
  // "sw3" -> "SW3". Returns '' if nothing district-like is present. Kept in sync
  // with scripts/export_similar_listings.py so the current-property district and
  // the exported `p` field compare like-for-like.
  function normalizeDistrict(pc) {
    if (!pc) return '';
    const m = pc.toUpperCase().match(/^([A-Z]{1,2}[0-9][0-9A-Z]?)/);
    return m ? m[1] : '';
  }

  async function findSimilarProperties(fairValue, postcodeDistrict, beds, baths, amenities, limit = 3) {
    /**
     * Find similar comps for the current property.
     *
     * Location is a HARD GATE, not a soft bonus (see SIMILAR_PROPERTIES_DEBUG.md):
     *   tier 0 = same postcode district (e.g. SW10)
     *   tier 1 = same postcode area only (e.g. SW*) — used to backfill if there
     *            aren't enough same-district comps
     *   anything outside the area, or with an unknown/empty district, is DISCARDED.
     * Within each tier we rank by a price/beds/baths/amenities similarity score,
     * but a same-district comp ALWAYS outranks a same-area one (tier0 first),
     * so a wrong-location comp can never reach the top results. This trades
     * "always 3 comps" for "never a comp from the wrong place" — thin districts
     * return fewer, or zero, comps rather than garbage.
     */
    const listings = await loadSimilarListings();
    if (!listings || Object.keys(listings).length === 0) {
      log(' No similar listings available');
      return [];
    }

    const currentUrl = window.location.href;
    const targetDistrict = normalizeDistrict(postcodeDistrict);
    const targetArea = targetDistrict.match(/^([A-Z]+)/)?.[1] || '';
    const targetAmenities = new Set(amenities || []);

    // tier0 = same district, tier1 = same area (different district)
    const tier0 = [];
    const tier1 = [];

    for (const [id, listing] of Object.entries(listings)) {
      // Skip current property
      if (listing.u && currentUrl.includes(listing.u)) continue;
      if (!listing.pr) continue;

      const listingPrice = listing.pr;
      const listingDistrict = normalizeDistrict(listing.p);
      const listingBeds = listing.b || 0;
      const listingBaths = listing.ba || 1;
      const listingAmenities = new Set(listing.am || []);

      // --- HARD LOCATION GATE (must run before scoring) ---
      // tier 0: exact district. tier 1: same area-prefix (guard against an empty
      // listingDistrict, which would make startsWith('') match everything).
      let tier;
      if (targetDistrict && listingDistrict === targetDistrict) {
        tier = 0;
      } else if (targetArea && listingDistrict && listingDistrict.startsWith(targetArea)) {
        tier = 1;
      } else {
        continue; // outside the area, or unknown district -> not a comp
      }

      // Price similarity (0-40 points). >50% off is never a comp.
      const priceDiff = Math.abs(listingPrice - fairValue) / fairValue;
      let score = 0;
      if (priceDiff <= 0.1) score += 40;
      else if (priceDiff <= 0.2) score += 30;
      else if (priceDiff <= 0.3) score += 20;
      else if (priceDiff <= 0.5) score += 10;
      else continue;

      // Beds similarity (0-15 points)
      const bedsDiff = Math.abs(listingBeds - beds);
      if (bedsDiff === 0) score += 15;
      else if (bedsDiff === 1) score += 8;
      else if (bedsDiff === 2) score += 3;

      // Baths similarity (0-10 points). NOTE: exported `ba` is the default 1 for
      // ~half of listings, so this is a weak tie-breaker, not a gate.
      const bathsDiff = Math.abs(listingBaths - baths);
      if (bathsDiff === 0) score += 10;
      else if (bathsDiff === 1) score += 5;

      // Amenities similarity (0-5 points)
      if (targetAmenities.size > 0 && listingAmenities.size > 0) {
        const overlap = [...targetAmenities].filter(a => listingAmenities.has(a)).length;
        const total = new Set([...targetAmenities, ...listingAmenities]).size;
        if (total > 0) score += 5 * (overlap / total);
      }

      const candidate = {
        url: listing.u,
        address: listing.a || 'Property',
        price: listingPrice,
        beds: listingBeds,
        baths: listingBaths,
        postcode: listingDistrict,
        sqft: listing.s,
        score: Math.round(score * 10) / 10
      };
      (tier === 0 ? tier0 : tier1).push(candidate);
    }

    // Rank within each tier by score, then take same-district first, backfilling
    // with same-area only if we don't have `limit` same-district comps.
    tier0.sort((a, b) => b.score - a.score);
    tier1.sort((a, b) => b.score - a.score);
    const result = tier0.concat(tier1).slice(0, limit);
    console.log(`[RFV] Similar: ${tier0.length} same-district + ${tier1.length} same-area for ${targetDistrict || '?'}, returning top ${result.length}`);
    return result;
  }

  // ============================================
  // UI
  // ============================================

  function removeExisting() {
    document.getElementById('rent-fair-value')?.remove();
  }

  function createContainer() {
    removeExisting();
    const el = document.createElement('div');
    el.id = 'rent-fair-value';
    document.body.appendChild(el);
    return el;
  }

  function injectLoadingState(msg) {
    const el = createContainer();
    el.innerHTML = `
      <div class="rfv-container">
        <div class="rfv-header">RENT FAIR VALUE</div>
        <div class="rfv-loading">
          <div class="rfv-spinner"></div>
          <div class="rfv-loading-text">${escapeHtml(msg)}</div>
        </div>
      </div>
    `;
  }

  function injectError(msg) {
    const el = createContainer();
    el.innerHTML = `
      <div class="rfv-container">
        <div class="rfv-header">RENT FAIR VALUE</div>
        <div class="rfv-error">
          <div class="rfv-error-icon">⚠️</div>
          <div class="rfv-error-text">${escapeHtml(msg)}</div>
        </div>
      </div>
    `;
  }

  function injectShortLetWarning(priceText) {
    const el = createContainer();
    const askingPrice = parsePrice(priceText);
    const priceDisplay = askingPrice ? `£${formatNum(askingPrice)}/mo` : priceText || 'N/A';

    el.innerHTML = `
      <div class="rfv-container">
        <div class="rfv-header">RENT FAIR VALUE</div>

        <div class="rfv-label">Asking</div>
        <div class="rfv-price">${escapeHtml(priceDisplay)}</div>

        <hr class="rfv-divider">

        <div class="rfv-short-let-warning">
          <div class="rfv-warning-icon">⚠️</div>
          <div class="rfv-warning-title">Short-Term Let</div>
          <div class="rfv-warning-text">
            This is a short-term let. Our model is trained on long-term rentals and cannot accurately value short-term lets, which typically command 2-3x higher rents.
          </div>
        </div>

        <div class="rfv-footer">Short-term lets excluded from analysis</div>
      </div>
    `;
  }

  async function displayResult(r, source) {
    // FIX #1: When the sqft was NOT found on the page and we fell back to estimateSqft()
    // (source === 'estimated'), the fair value — and therefore the over/under-priced
    // verdict — rests on a guessed size. Estimated size flips the verdict on ~29% of
    // real listings and produces a FALSE "OVERPRICED" on ~12%. So in that state we must
    // NOT render a confident verdict: suppress the bold OVERPRICED / GOOD DEAL banner,
    // show a neutral "SIZE UNKNOWN — EST. ONLY" state, soften the % to a tilde estimate,
    // widen the displayed range (±40%), and surface the caveat ABOVE the number.
    // The normal path (size from the page / floorplan / cache) is UNCHANGED.
    const sizeEstimated = source === 'estimated';

    const assessment = r.premium_pct > 15 ? 'overpriced' : r.premium_pct < -10 ? 'good_deal' : 'fair';
    const colorClass = assessment === 'overpriced' ? 'rfv-overpriced' :
                       assessment === 'good_deal' ? 'rfv-good-deal' : 'rfv-fair';
    const label = assessment.replace('_', ' ').toUpperCase();
    const sign = r.premium_pct > 0 ? '+' : '';

    // When size is estimated, widen the shown range to ~±40% (fairValue×0.6..×1.6) to
    // reflect that the point estimate is soft, rather than the normal ~±20% band.
    const rangeLow = sizeEstimated ? Math.round(r.fair_value * 0.6) : r.range_low;
    const rangeHigh = sizeEstimated ? Math.round(r.fair_value * 1.6) : r.range_high;

    const sizeNote = source === 'ocr' ? `${r.size_sqft} sqft (from floorplan)` :
                     source === 'estimated' ? 'Size estimated from beds' :
                     source === 'cached' ? 'From daily analysis' :
                     `${r.size_sqft} sqft`;

    // The assessment block: confident verdict on the normal path; a neutral
    // "size unknown" state when the size was estimated.
    const assessmentHtml = sizeEstimated
      ? `<div class="rfv-assessment rfv-fair rfv-estimated">
           <div class="rfv-assessment-value rfv-estimated-pct">~${sign}${r.premium_pct}%</div>
           <div class="rfv-assessment-label">SIZE UNKNOWN — EST. ONLY</div>
         </div>`
      : `<div class="rfv-assessment ${colorClass}">
           <div class="rfv-assessment-value">${sign}${r.premium_pct}%</div>
           <div class="rfv-assessment-label">${label}</div>
         </div>`;

    // Caveat shown ABOVE the model number when the valuation rests on a guessed size.
    const sizeCaveatHtml = sizeEstimated
      ? `<div class="rfv-size-caveat">Size not found on page — valuation approximate</div>`
      : '';

    const amenitiesHtml = r.amenities_detected?.length > 0
      ? `<div class="rfv-amenities">${r.amenities_detected.map(a =>
          `<span class="rfv-amenity">${escapeHtml(a)}</span>`).join('')}</div>`
      : '';

    // Find similar properties (async, don't block initial render)
    let similarHtml = '';

    const el = createContainer();
    el.innerHTML = `
      <div class="rfv-container">
        <div class="rfv-header">RENT FAIR VALUE</div>

        <div class="rfv-label">Asking</div>
        <div class="rfv-price">£${formatNum(r.asking_price)}/mo</div>

        <hr class="rfv-divider">

        ${sizeCaveatHtml}

        <div class="rfv-label">Model Estimate</div>
        <div class="rfv-price">£${formatNum(r.fair_value)}/mo</div>
        <div class="rfv-range">Range: £${formatNum(rangeLow)} – £${formatNum(rangeHigh)}</div>

        ${assessmentHtml}

        ${amenitiesHtml}

        <div class="rfv-size-note">${escapeHtml(sizeNote)}</div>

        <div id="rfv-similar-placeholder"></div>

        <button class="rfv-compare-btn" id="rfv-compare-btn">
          Compare with Similar Properties
        </button>

        <div class="rfv-footer">${
          r.predictor_source === 'fallback'
            ? 'XGBoost V20 · ~estimate (offline)'
            : 'XGBoost V20 · Live'
        }</div>
      </div>
    `;

    // Add click handler for Compare button
    const compareBtn = document.getElementById('rfv-compare-btn');
    if (compareBtn) {
      compareBtn.addEventListener('click', () => {
        openComparePage(r);
      });
    }

    // Load similar properties in background
    if (r.postcode_district && r.beds) {
      findSimilarProperties(
        r.fair_value,
        r.postcode_district,
        r.beds,
        r.baths || 1,
        r.amenities_detected,
        3
      ).then(similar => {
        if (similar && similar.length > 0) {
          const placeholder = document.getElementById('rfv-similar-placeholder');
          if (placeholder) {
            placeholder.innerHTML = renderSimilarProperties(similar);
          }
        }
      }).catch(err => {
        log(' Similar properties error:', err);
      });
    }
  }

  function renderSimilarProperties(properties) {
    if (!properties || properties.length === 0) return '';

    const items = properties.map(p => {
      const shortAddress = truncateAddress(p.address, 35);
      const specs = `${p.beds}bed · ${p.baths}bath${p.sqft ? ` · ${p.sqft}sqft` : ''}`;
      return `
        <a href="${escapeHtml(p.url)}" class="rfv-similar-item" target="_blank" rel="noopener">
          <div class="rfv-similar-address">${escapeHtml(shortAddress)}</div>
          <div class="rfv-similar-details">
            <span class="rfv-similar-price">£${formatNum(p.price)}/mo</span>
            <span class="rfv-similar-specs">${escapeHtml(specs)}</span>
          </div>
        </a>
      `;
    }).join('');

    return `
      <div class="rfv-similar-section">
        <div class="rfv-similar-title">Similar Properties</div>
        <div class="rfv-similar-list">
          ${items}
        </div>
      </div>
    `;
  }

  function truncateAddress(address, maxLen) {
    if (!address) return 'Property';
    if (address.length <= maxLen) return address;
    return address.substring(0, maxLen - 3) + '...';
  }

  function formatNum(n) { return n.toLocaleString('en-GB'); }
  function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }

  function openComparePage(result) {
    // Build URL params from result data
    const propertyData = extractPropertyData();
    const address = propertyData?.address?.displayAddress || 'Unknown Property';
    const postcode = extractPostcode(propertyData);
    const propertyId = extractPropertyId();

    const params = new URLSearchParams({
      address: address,
      postcode: postcode || '',  // extractPostcode may return null (no real postcode)
      beds: result.beds || 2,
      sqft: result.size_sqft || 0,
      price: result.asking_price,
      fairValue: result.fair_value,
      type: extractPropertyType(propertyData),
      url: window.location.href,
      propertyId: propertyId || '',
    });

    // Open compare page in new tab
    const compareUrl = chrome.runtime.getURL('compare.html') + '?' + params.toString();
    window.open(compareUrl, '_blank');
  }

})();
