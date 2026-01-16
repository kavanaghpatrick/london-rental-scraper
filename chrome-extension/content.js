/**
 * Rent Fair Value - Chrome Extension
 * Shows ML-powered fair rent estimates on Rightmove listings
 *
 * Flow:
 * 1. Check predictions cache (instant if found)
 * 2. If not cached: extract data from page, OCR floorplan, run XGBoost model locally
 */

console.log('[RFV] Script loaded!');

(function() {
  'use strict';

  const CONFIG = {
    PREDICTIONS_URL: 'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/chrome-extension/api/predictions.json',
    SIMILAR_URL: 'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/chrome-extension/api/similar_listings.json',
    MODEL_URL: chrome.runtime.getURL('api/model.json'),
    FEATURES_URL: chrome.runtime.getURL('api/features.json'),
    OCR_TIMEOUT: 60000,
  };

  // Prevent duplicate execution
  if (window.__rentFairValueLoaded) return;
  window.__rentFairValueLoaded = true;

  // Caches
  let predictionsCache = null;
  let similarListingsCache = null;
  let xgbPredictor = null;

  // Main execution
  init();

  async function init() {
    try {
      // 1. Extract property data from page
      const propertyData = extractPropertyData();
      if (!propertyData) {
        console.log('[RFV] No property data found');
        return;
      }

      const propertyId = extractPropertyId();
      console.log('[RFV] Property ID:', propertyId);

      // 2. Check if short-term let - show warning instead of valuation
      const letType = extractLetType(propertyData);
      console.log('[RFV] Let type:', letType);

      if (letType === 'short') {
        console.log('[RFV] Short-term let detected - showing warning');
        injectShortLetWarning(propertyData.prices?.primaryPrice);
        return;
      }

      // 3. Show loading
      injectLoadingState('Loading estimate...');

      // 4. Parse asking price
      const askingPrice = parsePrice(propertyData.prices?.primaryPrice);
      if (!askingPrice) {
        injectError('Could not parse price');
        return;
      }

      // 4. Try cache first (instant) - DISABLED FOR TESTING v0.6.0 fixes
      // const cached = await getCachedPrediction(propertyId);
      // if (cached) {
      //   console.log('[RFV] Cache hit!');
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
      console.log('[RFV] Cache disabled - running live prediction');

      // 5. Not cached - run full analysis
      console.log('[RFV] Cache miss, running local model...');
      injectLoadingState('Analyzing property...');

      const result = await analyzeProperty(propertyData, askingPrice);
      displayResult(result, result.size_source);

    } catch (error) {
      console.error('[RFV] Error:', error);
      injectError('Something went wrong');
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
    const floorplanUrl = getFloorplanUrl(propertyData);
    console.log('[RFV] Floorplan URL:', floorplanUrl || 'NOT FOUND');
    if (floorplanUrl) {
      injectLoadingState('Reading floorplan...');
      const ocrResult = await ocrFloorplan(floorplanUrl);
      ocrText = ocrResult.text || '';
      // Only use OCR sqft if we don't have it from page
      if (!sizeSqft && ocrResult.sqft) {
        sizeSqft = ocrResult.sqft;
        sizeSource = 'ocr';
      }
      console.log('[RFV] OCR result: sqft=' + (ocrResult.sqft || 'none') + ', text length=' + ocrText.length);
    } else {
      console.log('[RFV] No floorplan found in property data');
    }

    if (!sizeSqft) {
      // Estimate from beds
      sizeSqft = estimateSqft(beds);
      sizeSource = 'estimated';
    }

    // Load XGBoost model if needed
    if (!xgbPredictor) {
      injectLoadingState('Loading model...');
      xgbPredictor = new window.XGBoostPredictor();
      await xgbPredictor.load(CONFIG.MODEL_URL, CONFIG.FEATURES_URL);
    }

    // Build features and predict
    injectLoadingState('Calculating fair value...');
    console.log(`[RFV] Building features with: beds=${beds}, baths=${baths}, sqft=${sizeSqft}, postcode=${postcode}, propertyType=${propertyType}, agent=${agentName}`);
    const features = window.XGBFeatures.buildFeatures({
      bedrooms: beds,
      bathrooms: baths,
      size_sqft: sizeSqft,
      postcode: postcode,
      propertyType: propertyType,
      latitude: lat,
      longitude: lon,
      address: address,  // V16: for garden square/prime street detection
      description: description,
      ocrText: ocrText, // Pass OCR text for floor extraction
      agentName: agentName, // For premium agent detection
      pageUrl: window.location.href, // For source quality detection
    });
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
    };
  }

  // ============================================
  // DATA EXTRACTION
  // ============================================

  function extractPropertyData() {
    // Strategy 1: __NEXT_DATA__
    const nextDataScript = document.getElementById('__NEXT_DATA__');
    if (nextDataScript) {
      try {
        const data = JSON.parse(nextDataScript.textContent);
        const propertyData = data?.props?.pageProps?.propertyData;
        if (propertyData) {
          console.log('[RFV] Found via __NEXT_DATA__');
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
            console.log('[RFV] Found via PAGE_MODEL');
            return data.propertyData;
          }
        } catch (e) {}
      }
    }

    console.log('[RFV] No property data found');
    return null;
  }

  function extractPropertyId() {
    const match = window.location.pathname.match(/\/properties\/(\d+)/);
    return match ? match[1] : null;
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
    // Extract let type from Rightmove propertyData
    // Returns 'short' for short-term lets, 'long' otherwise

    // 1. Check lettings.letType field
    if (data.lettings?.letType) {
      const letType = data.lettings.letType.toLowerCase();
      if (letType.includes('short')) return 'short';
    }

    // 2. Check channel field (sometimes indicates short let)
    if (data.channel?.toLowerCase().includes('short')) return 'short';

    // 3. Check description/property phrase for short let keywords
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
        textToCheck.includes('holiday let')) {
      return 'short';
    }

    // 4. Check URL
    if (window.location.href.toLowerCase().includes('short')) {
      return 'short';
    }

    return 'long';
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
      // Check for specific types
      if (phrase.includes('penthouse')) return 'penthouse';
      if (phrase.includes('studio')) return 'studio';
      if (phrase.includes('maisonette')) return 'maisonette';
      if (phrase.includes('house')) return 'house';
      if (phrase.includes('apartment')) return 'apartment';
      if (phrase.includes('flat')) return 'flat';
    }
    // 4. From listing update reason
    if (data.listingUpdate?.listingUpdateReason) {
      const reason = data.listingUpdate.listingUpdateReason.toLowerCase();
      if (reason.includes('penthouse')) return 'penthouse';
      if (reason.includes('studio')) return 'studio';
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
    // Default empty
    return '';
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
    // Check floorplans array
    const floorplans = data.floorplans || [];
    if (floorplans.length > 0) {
      return floorplans[0].url || floorplans[0].srcUrl;
    }
    // Check media array
    const media = data.media || [];
    for (const m of media) {
      if (m.type === 'floorplan' || (m.url && m.url.includes('_FLP_'))) {
        return m.url || m.srcUrl;
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
      console.error('[RFV] Tesseract not loaded! Check vendor/tesseract.min.js');
      return { sqft: null, text: '' };
    }
    console.log('[RFV] Tesseract available, starting OCR...');

    let worker = null;
    try {
      console.log('[RFV] Running OCR on:', url);

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
      console.log('[RFV] Image fetched via background worker, data length:', imgData.length);

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
      console.log('[RFV] OCR result:', text.substring(0, 200));

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
            console.log('[RFV] Found sqft via OCR:', sqft);
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
            console.log('[RFV] Found sqm via OCR:', sqm, '-> sqft:', sqft);
            return { sqft, text };
          }
        }
      }

      console.log('[RFV] No size pattern found in OCR text');
      return { sqft: null, text };
    } catch (e) {
      console.error('[RFV] OCR failed:', e.message);
      return { sqft: null, text: '' };
    } finally {
      // Always terminate worker to prevent memory leak
      if (worker) {
        try {
          await worker.terminate();
          console.log('[RFV] Tesseract worker terminated');
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
    const num = text.replace(/[^\d.]/g, '');
    if (!num) return null;
    let price = parseFloat(num);
    if (/pw|per week|weekly/i.test(text)) {
      price = price * 52 / 12;
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
          console.log('[RFV] Cache loaded:', Object.keys(predictionsCache).length);
        }
      }
      return predictionsCache?.[`rightmove:${propertyId}`] || null;
    } catch (e) {
      console.error('[RFV] Cache load failed:', e);
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
        console.log('[RFV] Similar listings loaded:', Object.keys(similarListingsCache).length);
      }
    } catch (e) {
      console.error('[RFV] Similar listings load failed:', e);
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
      console.log('[RFV] No similar listings available');
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

        <div class="rfv-footer">XGBoost V20 · ${source === 'cached' ? 'Cached' : 'Live'}</div>
      </div>
    `;

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
        console.log('[RFV] Similar properties error:', err);
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

})();
