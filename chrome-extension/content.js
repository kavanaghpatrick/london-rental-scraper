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

  // Debug logging helper - only logs when DEBUG is true
  function log(...args) {
    if (DEBUG) log('', ...args);
  }
  function logError(...args) {
    logError('', ...args); // Always log errors
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
  };

  function detectSite() {
    const hostname = window.location.hostname;
    if (hostname.includes('rightmove.co.uk')) return SITES.RIGHTMOVE;
    if (hostname.includes('knightfrank.co.uk')) return SITES.KNIGHTFRANK;
    if (hostname.includes('chestertons.co.uk')) return SITES.CHESTERTONS;
    if (hostname.includes('savills.com')) return SITES.SAVILLS;
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
      default:
        return false;
    }
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
      displayResult(result, result.size_source);

      isRunning = false;
    } catch (error) {
      logError(' Error:', error);
      injectError('Something went wrong');
      isRunning = false;
    }
  }

  async function analyzeProperty(propertyData, askingPrice) {
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

    // ALWAYS run OCR if floorplan available - we need it for floor extraction even if sqft is known
    // For SPA sites (Chestertons, Savills), click the floorplan tab first and retry if needed
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

    if (floorplanUrl) {
      injectLoadingState('Reading floorplan...');
      const ocrResult = await ocrFloorplan(floorplanUrl);
      ocrText = ocrResult.text || '';
      // Only use OCR sqft if we don't have it from page
      if (!sizeSqft && ocrResult.sqft) {
        sizeSqft = ocrResult.sqft;
        sizeSource = 'ocr';
      }
      log(' OCR result: sqft=' + (ocrResult.sqft || 'none') + ', text length=' + ocrText.length);
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
        postcode_district: postcode.split(' ')[0],
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

    // Extract postcode district for similar properties search
    const postcodeDistrict = postcode.split(' ')[0];

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

    // Strategy 2: window.PAGE_MODEL
    for (const script of document.querySelectorAll('script')) {
      const text = script.textContent || '';
      const match = text.match(/window\.PAGE_MODEL\s*=\s*/);
      if (match) {
        try {
          const start = match.index + match[0].length;
          let braceCount = 0, i = start;
          while (i < text.length) {
            if (text[i] === '{') braceCount++;
            else if (text[i] === '}' && --braceCount === 0) break;
            i++;
          }
          const data = JSON.parse(text.slice(start, i + 1));
          if (data.propertyData) {
            log(' Found via PAGE_MODEL');
            return data.propertyData;
          }
        } catch (e) {}
      }
    }

    log(' No Rightmove property data found');
    return null;
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
    if (data.address?.displayAddress) {
      const pcMatch = data.address.displayAddress.match(/([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})/i);
      if (pcMatch) {
        const parts = pcMatch[1].split(/\s+/);
        data.address.outcode = parts[0] || '';
        data.address.incode = parts[1] || '';
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

  function extractPropertyDataChestertons() {
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

    // Bedrooms/Bathrooms - regex on page text (more reliable than DOM selectors)
    const bedsMatch = pageText.match(/(\d+)\s*(?:bed(?:room)?s?)/i);
    if (bedsMatch) {
      data.bedrooms = parseInt(bedsMatch[1], 10);
    }
    const bathsMatch = pageText.match(/(\d+)\s*(?:bath(?:room)?s?)/i);
    if (bathsMatch) {
      data.bathrooms = parseInt(bathsMatch[1], 10);
    }

    // Size - look for sqft pattern
    const sizeMatch = pageText.match(/(\d{1,5}(?:,\d{3})?)\s*(?:sq\.?\s*ft|sqft|square\s*feet)/i);
    if (sizeMatch) {
      const sqft = parseInt(sizeMatch[1].replace(/,/g, ''), 10);
      if (sqft >= 100 && sqft <= 50000) { // Validate reasonable range
        data.sizings = [{ minimumSize: sqft, unit: 'sqft' }];
      }
    }

    // Postcode from address
    if (data.address?.displayAddress) {
      const pcMatch = data.address.displayAddress.match(/([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})/i);
      if (pcMatch) {
        const parts = pcMatch[1].split(/\s+/);
        data.address.outcode = parts[0] || '';
        data.address.incode = parts[1] || '';
      }
    }

    // Agent name
    data.customer = { companyName: 'Chestertons' };

    // Property type from page text
    const textLower = pageText.toLowerCase();
    if (textLower.includes('penthouse')) data.propertyType = 'penthouse';
    else if (textLower.includes('studio')) data.propertyType = 'studio';
    else if (textLower.includes('house')) data.propertyType = 'house';
    else if (textLower.includes('maisonette')) data.propertyType = 'maisonette';
    else if (textLower.includes('apartment')) data.propertyType = 'apartment';
    else data.propertyType = 'flat';

    // Floorplan - Chestertons uses homeflow-assets CDN with /files/floorplan/ path
    // CRITICAL: Match spider pattern - look for /files/floorplan/ path, not just domain
    // Also check data-src for lazy-loaded images
    const floorplanImg = document.querySelector(
      'img[src*="/files/floorplan/"], img[data-src*="/files/floorplan/"], ' +
      'img[src*="floorplan"], img[data-src*="floorplan"]'
    );
    if (floorplanImg) {
      const url = floorplanImg.src || floorplanImg.dataset.src || floorplanImg.getAttribute('data-src');
      if (url) data.floorplans = [{ url }];
    }
    if (!data.floorplans) {
      // Try to find in page HTML - use spider's exact pattern
      const floorplanMatch = document.body.innerHTML.match(/https:\/\/[^"\s]+\/files\/floorplan\/[^"\s]+/i);
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
    return Object.keys(data).length > 2 ? data : null; // Need more than just _source and customer
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
      default:
        return null;
    }
  }

  function extractPostcode(data) {
    if (data.address?.outcode) {
      return data.address.outcode + (data.address.incode ? ' ' + data.address.incode : '');
    }
    const addr = data.address?.displayAddress || '';
    const match = addr.match(/([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d?[A-Z]{0,2})/i);
    return match ? match[1] : 'SW3';
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

      // Final fallback - regex search in page HTML for common CDN patterns
      const htmlContent = document.body.innerHTML;
      const patterns = [
        /https:\/\/[^"\s]+\/files\/floorplan\/[^"\s]+/i,  // Chestertons
        /https:\/\/content\.knightfrank\.com\/[^"\s]*floorplan[^"\s]*\.(?:jpg|png|jpeg)/i,  // Knight Frank
        /https:\/\/[^"\s]*savills[^"\s]*(?:floorplan|floor-plan|_fp)[^"\s]*\.(?:jpg|png|jpeg)/i,  // Savills
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
      logError(' OCR failed:', e.message);
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

  async function findSimilarProperties(fairValue, postcodeDistrict, beds, baths, amenities, limit = 3) {
    /**
     * Find similar properties based on model price, location, beds, baths.
     * Scoring: price (40pts), location (30pts), beds (15pts), baths (10pts), amenities (5pts)
     */
    const listings = await loadSimilarListings();
    if (!listings || Object.keys(listings).length === 0) {
      log(' No similar listings available');
      return [];
    }

    const currentUrl = window.location.href;
    const postcodeArea = postcodeDistrict.match(/^([A-Z]+)/i)?.[1]?.toUpperCase() || '';
    const targetAmenities = new Set(amenities || []);

    const candidates = [];

    for (const [id, listing] of Object.entries(listings)) {
      // Skip current property
      if (listing.u && currentUrl.includes(listing.u)) continue;
      if (!listing.pr) continue;

      let score = 0;
      const listingPrice = listing.pr;
      const listingDistrict = listing.p || '';
      const listingBeds = listing.b || 0;
      const listingBaths = listing.ba || 1;
      const listingAmenities = new Set(listing.am || []);

      // Price similarity (0-40 points)
      const priceDiff = Math.abs(listingPrice - fairValue) / fairValue;
      if (priceDiff <= 0.1) score += 40;
      else if (priceDiff <= 0.2) score += 30;
      else if (priceDiff <= 0.3) score += 20;
      else if (priceDiff <= 0.5) score += 10;
      else continue; // Skip if price > 50% different

      // Location similarity (0-30 points)
      if (listingDistrict === postcodeDistrict) {
        score += 30;
      } else if (listingDistrict && postcodeArea && listingDistrict.startsWith(postcodeArea)) {
        score += 15;
      }

      // Beds similarity (0-15 points)
      const bedsDiff = Math.abs(listingBeds - beds);
      if (bedsDiff === 0) score += 15;
      else if (bedsDiff === 1) score += 8;
      else if (bedsDiff === 2) score += 3;

      // Baths similarity (0-10 points)
      const bathsDiff = Math.abs(listingBaths - baths);
      if (bathsDiff === 0) score += 10;
      else if (bathsDiff === 1) score += 5;

      // Amenities similarity (0-5 points)
      if (targetAmenities.size > 0 && listingAmenities.size > 0) {
        const overlap = [...targetAmenities].filter(a => listingAmenities.has(a)).length;
        const total = new Set([...targetAmenities, ...listingAmenities]).size;
        if (total > 0) score += 5 * (overlap / total);
      }

      // Only include if minimum threshold met
      if (score >= 30) {
        candidates.push({
          url: listing.u,
          address: listing.a || 'Property',
          price: listingPrice,
          beds: listingBeds,
          baths: listingBaths,
          postcode: listingDistrict,
          sqft: listing.s,
          score: Math.round(score * 10) / 10
        });
      }
    }

    // Sort by score descending, return top matches
    candidates.sort((a, b) => b.score - a.score);
    console.log(`[RFV] Found ${candidates.length} similar properties, returning top ${limit}`);
    return candidates.slice(0, limit);
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
    const assessment = r.premium_pct > 15 ? 'overpriced' : r.premium_pct < -10 ? 'good_deal' : 'fair';
    const colorClass = assessment === 'overpriced' ? 'rfv-overpriced' :
                       assessment === 'good_deal' ? 'rfv-good-deal' : 'rfv-fair';
    const label = assessment.replace('_', ' ').toUpperCase();
    const sign = r.premium_pct > 0 ? '+' : '';

    const sizeNote = source === 'ocr' ? `${r.size_sqft} sqft (from floorplan)` :
                     source === 'estimated' ? 'Size estimated from beds' :
                     source === 'cached' ? 'From daily analysis' :
                     `${r.size_sqft} sqft`;

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

        <div class="rfv-label">Model Estimate</div>
        <div class="rfv-price">£${formatNum(r.fair_value)}/mo</div>
        <div class="rfv-range">Range: £${formatNum(r.range_low)} – £${formatNum(r.range_high)}</div>

        <div class="rfv-assessment ${colorClass}">
          <div class="rfv-assessment-value">${sign}${r.premium_pct}%</div>
          <div class="rfv-assessment-label">${label}</div>
        </div>

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
      postcode: postcode,
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
