/**
 * Rent Fair Value - Chrome Extension
 * Shows ML-powered fair rent estimates on Rightmove listings
 */

console.log('[RFV] Script loaded!');

(function() {
  'use strict';

  console.log('[RFV] IIFE starting...');

  // Configuration
  const CONFIG = {
    // Predictions cache hosted on GitHub (updated daily)
    PREDICTIONS_URL: 'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/chrome-extension/api/predictions.json',
    // Model-based lookup table for cache misses (postcode_beds_sqft -> fair_value)
    LOOKUP_URL: 'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/chrome-extension/api/lookup.json',
    OCR_TIMEOUT: 60000,  // 60s max for OCR
    // Sqft buckets used in lookup table
    SQFT_BUCKETS: [300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500, 3000],
  };

  // Prevent duplicate execution
  if (window.__rentFairValueLoaded) return;
  window.__rentFairValueLoaded = true;

  // Caches (loaded once)
  let predictionsCache = null;
  let lookupCache = null;

  // Main execution
  init();

  async function init() {
    try {
      // 1. Extract property data from page
      const propertyData = extractPropertyData();
      if (!propertyData) {
        console.log('[RFV] No property data found on this page');
        return;
      }

      // Get property ID from URL
      const propertyId = extractPropertyId();
      console.log('[RFV] Property ID:', propertyId);

      // 2. Show loading state
      injectLoadingState('Loading estimate...');

      // 3. Parse asking price
      const askingPrice = parsePrice(propertyData.prices?.primaryPrice);
      if (!askingPrice) {
        injectError('Could not parse price');
        return;
      }

      // 4. Try to get prediction from cache (instant!)
      const cached = await getCachedPrediction(propertyId);

      if (cached) {
        console.log('[RFV] Found in cache!');
        const result = {
          asking_price: askingPrice,
          fair_value: cached.fv,
          range_low: cached.lo,
          range_high: cached.hi,
          premium_pct: cached.pct,
          assessment: cached.pct > 15 ? 'overpriced' : cached.pct < -10 ? 'good_deal' : 'fair',
          size_sqft: cached.sq,
          amenities_detected: [],
        };
        injectSidebar(result, cached.sq ? 'cached' : 'estimated');
        return;
      }

      // 5. Not in cache - calculate client-side estimate
      console.log('[RFV] Not in cache, calculating locally...');
      injectLoadingState('Calculating estimate...');

      const estimate = await calculateLocalEstimate(propertyData, askingPrice);
      injectSidebar(estimate, estimate.size_source);

    } catch (error) {
      console.error('[RFV] Error:', error);
      injectError('Something went wrong');
    }
  }

  function extractPropertyId() {
    // Extract from URL: /properties/123456789
    const match = window.location.pathname.match(/\/properties\/(\d+)/);
    return match ? match[1] : null;
  }

  function parsePrice(priceText) {
    if (!priceText) return null;
    const priceStr = priceText.replace(/[^\d.]/g, '');
    if (!priceStr) return null;
    let price = parseFloat(priceStr);
    // Convert weekly to monthly
    if (/pw|per week|weekly/i.test(priceText)) {
      price = price * 52 / 12;
    }
    return Math.round(price);
  }

  async function getCachedPrediction(propertyId) {
    if (!propertyId) return null;

    try {
      // Load cache if not loaded
      if (!predictionsCache) {
        const response = await fetch(CONFIG.PREDICTIONS_URL);
        if (response.ok) {
          predictionsCache = await response.json();
          console.log('[RFV] Loaded predictions cache:', Object.keys(predictionsCache).length, 'entries');
        }
      }

      // Look up by rightmove:propertyId
      const key = `rightmove:${propertyId}`;
      return predictionsCache?.[key] || null;

    } catch (e) {
      console.error('[RFV] Failed to load cache:', e);
      return null;
    }
  }

  async function calculateLocalEstimate(propertyData, askingPrice) {
    // Extract fields
    const beds = Math.min(propertyData.bedrooms || 1, 5);  // Cap at 5 for lookup
    const outcode = extractPostcodeDistrict(propertyData);

    // Get sqft from page or estimate
    let sizeSqft = extractSqftFromSizings(propertyData.sizings);
    let sizeSource = 'page';

    if (!sizeSqft) {
      // Try OCR if floorplan available
      const floorplanUrl = propertyData.floorplans?.[0]?.url;
      if (floorplanUrl && typeof Tesseract !== 'undefined') {
        injectLoadingState('Analyzing floorplan...');
        sizeSqft = await ocrFloorplan(floorplanUrl);
        if (sizeSqft) {
          sizeSource = 'ocr';
        }
      }
    }

    if (!sizeSqft) {
      // Estimate from beds using typical sizes
      sizeSqft = estimateSizeFromBeds(beds);
      sizeSource = 'estimated';
    }

    // Use model-based lookup table for fair value
    const fairValue = await lookupFairValue(outcode, beds, sizeSqft);
    const premiumPct = Math.round((askingPrice / fairValue - 1) * 100 * 10) / 10;

    return {
      asking_price: askingPrice,
      fair_value: fairValue,
      range_low: Math.round(fairValue * 0.79),
      range_high: Math.round(fairValue * 1.21),
      premium_pct: premiumPct,
      assessment: premiumPct > 15 ? 'overpriced' : premiumPct < -10 ? 'good_deal' : 'fair',
      size_sqft: sizeSqft,
      size_source: sizeSource,
      amenities_detected: [],
    };
  }

  function extractPostcodeDistrict(propertyData) {
    // Try to get postcode district (e.g., "SW3" from "SW3 4TX")
    const address = propertyData.address?.displayAddress || '';
    const outcode = propertyData.address?.outcode;

    if (outcode) return outcode;

    // Try to extract from address
    const match = address.match(/\b([A-Z]{1,2}\d{1,2}[A-Z]?)\b/i);
    return match ? match[1].toUpperCase() : 'SW3';  // Default to SW3
  }

  function estimateSizeFromBeds(beds) {
    // Typical sizes by bedroom count
    const baseSizes = { 0: 350, 1: 500, 2: 750, 3: 1000, 4: 1300, 5: 1600 };
    return baseSizes[Math.min(beds, 5)] || 500;
  }

  function findClosestSqftBucket(sqft) {
    // Find the closest sqft bucket from the lookup table
    let closest = CONFIG.SQFT_BUCKETS[0];
    let minDiff = Math.abs(sqft - closest);

    for (const bucket of CONFIG.SQFT_BUCKETS) {
      const diff = Math.abs(sqft - bucket);
      if (diff < minDiff) {
        minDiff = diff;
        closest = bucket;
      }
    }
    return closest;
  }

  async function lookupFairValue(postcode, beds, sqft) {
    // Load lookup table if not loaded
    if (!lookupCache) {
      try {
        const response = await fetch(CONFIG.LOOKUP_URL);
        if (response.ok) {
          lookupCache = await response.json();
          console.log('[RFV] Loaded lookup table:', Object.keys(lookupCache).length, 'entries');
        }
      } catch (e) {
        console.error('[RFV] Failed to load lookup table:', e);
      }
    }

    // Find closest sqft bucket
    const sqftBucket = findClosestSqftBucket(sqft);

    // Look up fair value: "PC_beds_sqft" -> fair_value
    const key = `${postcode}_${beds}_${sqftBucket}`;
    let fairValue = lookupCache?.[key];

    if (fairValue) {
      console.log(`[RFV] Lookup hit: ${key} -> £${fairValue}`);
      // Interpolate if actual sqft differs from bucket
      if (sqft !== sqftBucket) {
        // Simple linear interpolation based on sqft ratio
        fairValue = Math.round(fairValue * (sqft / sqftBucket));
      }
      return fairValue;
    }

    // Fallback: try without postcode-specific lookup (use SW3 as default)
    const fallbackKey = `SW3_${beds}_${sqftBucket}`;
    fairValue = lookupCache?.[fallbackKey];
    if (fairValue) {
      console.log(`[RFV] Lookup fallback: ${fallbackKey} -> £${fairValue}`);
      return Math.round(fairValue * (sqft / sqftBucket));
    }

    // Ultimate fallback: simple calculation (should rarely happen)
    console.log('[RFV] Lookup miss, using fallback calculation');
    return Math.round(sqft * 4.5 + beds * 200);
  }

  // ============================================
  // DATA EXTRACTION
  // ============================================

  function extractPropertyData() {
    // Strategy 1: Try __NEXT_DATA__ (used on some pages)
    const nextDataScript = document.getElementById('__NEXT_DATA__');
    if (nextDataScript) {
      try {
        const nextData = JSON.parse(nextDataScript.textContent);
        const propertyData = nextData?.props?.pageProps?.propertyData;
        if (propertyData) {
          console.log('[RFV] Found propertyData via __NEXT_DATA__');
          return propertyData;
        }
      } catch (e) {
        console.log('[RFV] __NEXT_DATA__ parse failed:', e.message);
      }
    }

    // Strategy 2: Try window.PAGE_MODEL (Rightmove's current format for detail pages)
    const scripts = document.querySelectorAll('script');
    for (const script of scripts) {
      const text = script.textContent || '';
      const match = text.match(/window\.PAGE_MODEL\s*=\s*/);
      if (match) {
        try {
          const start = match.index + match[0].length;
          // Find matching closing brace
          let braceCount = 0;
          let i = start;
          while (i < text.length) {
            if (text[i] === '{') braceCount++;
            else if (text[i] === '}') {
              braceCount--;
              if (braceCount === 0) break;
            }
            i++;
          }
          const jsonStr = text.slice(start, i + 1);
          const data = JSON.parse(jsonStr);
          const propertyData = data.propertyData;
          if (propertyData) {
            console.log('[RFV] Found propertyData via PAGE_MODEL');
            return propertyData;
          }
        } catch (e) {
          console.log('[RFV] PAGE_MODEL parse failed:', e.message);
        }
      }
    }

    // Strategy 3: Check for inline JSON with propertyData
    for (const script of scripts) {
      const text = script.textContent || '';
      if (text.includes('"propertyData"') && text.includes('"bedrooms"')) {
        try {
          // Try to find JSON object containing propertyData
          const jsonMatch = text.match(/\{[^{}]*"propertyData"[^]*\}/);
          if (jsonMatch) {
            const data = JSON.parse(jsonMatch[0]);
            if (data.propertyData) {
              console.log('[RFV] Found propertyData via inline JSON');
              return data.propertyData;
            }
          }
        } catch (e) {
          // Continue to next script
        }
      }
    }

    console.log('[RFV] No property data found after trying all strategies');
    return null;
  }

  function extractSqftFromSizings(sizings) {
    if (!sizings || !Array.isArray(sizings) || sizings.length === 0) {
      return null;
    }

    // Look for sqft entry
    const sqftEntry = sizings.find(s => s.unit === 'sqft');
    if (sqftEntry) {
      const value = sqftEntry.minimumSize || sqftEntry.maximumSize;
      if (value) {
        return parseInt(value, 10);
      }
    }

    // Try sqm and convert
    const sqmEntry = sizings.find(s => s.unit === 'sqm');
    if (sqmEntry) {
      const value = sqmEntry.minimumSize || sqmEntry.maximumSize;
      if (value) {
        return Math.round(parseInt(value, 10) * 10.764);  // sqm to sqft
      }
    }

    return null;
  }

  // ============================================
  // SQFT RESOLUTION (JSON → OCR → null)
  // ============================================

  async function getSqft(propertyData) {
    // 1. Try page JSON first (instant, most reliable)
    const jsonSqft = extractSqftFromSizings(propertyData.sizings);
    if (jsonSqft) {
      console.log('[RFV] Sqft from page JSON:', jsonSqft);
      return { sqft: jsonSqft, source: 'page' };
    }

    // 2. Try OCR on floorplan
    const floorplanUrl = propertyData.floorplans?.[0]?.url;
    if (floorplanUrl && typeof Tesseract !== 'undefined') {
      console.log('[RFV] Attempting OCR on floorplan...');
      injectLoadingState('Analyzing floorplan...');

      const ocrSqft = await ocrFloorplan(floorplanUrl);
      if (ocrSqft) {
        console.log('[RFV] Sqft from OCR:', ocrSqft);
        return { sqft: ocrSqft, source: 'ocr' };
      }
    }

    // 3. No sqft found - API will estimate
    console.log('[RFV] No sqft found, API will estimate');
    return { sqft: null, source: 'estimated' };
  }

  async function ocrFloorplan(url) {
    try {
      const result = await Promise.race([
        Tesseract.recognize(url, 'eng', {
          logger: m => {
            if (m.status === 'recognizing text') {
              const pct = Math.round(m.progress * 100);
              injectLoadingState(`Analyzing floorplan... ${pct}%`);
            }
          }
        }),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('OCR timeout')), CONFIG.OCR_TIMEOUT)
        )
      ]);

      const text = result.data.text;
      console.log('[RFV] OCR text:', text.substring(0, 200));

      // Extract sqft using regex (same patterns as server-side)
      const patterns = [
        /(\d{2,4})\s*(?:sq\.?\s*ft|sqft|square\s*feet)/i,
        /(\d{2,4})\s*ft²/i,
        /total[:\s]+(\d{2,4})\s*(?:sq|ft)/i,
        /(\d{3,4})\s*(?=.*(?:bedroom|flat|apartment))/i,
      ];

      for (const pattern of patterns) {
        const match = text.match(pattern);
        if (match) {
          const sqft = parseInt(match[1].replace(',', ''), 10);
          // Sanity check: 100-10000 sqft is reasonable for London
          if (sqft >= 100 && sqft <= 10000) {
            return sqft;
          }
        }
      }

      return null;
    } catch (e) {
      console.error('[RFV] OCR failed:', e.message);
      return null;
    }
  }

  // ============================================
  // API CALL
  // ============================================

  async function callValuationAPI(propertyData, sizeSqft) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), CONFIG.API_TIMEOUT);

      const response = await fetch(CONFIG.API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': CONFIG.API_KEY,
        },
        body: JSON.stringify({
          source: 'rightmove',
          property: propertyData,
          size_sqft: sizeSqft,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        if (response.status === 401) {
          return { error: 'Invalid API key' };
        }
        return { error: `API error: ${response.status}` };
      }

      return await response.json();
    } catch (e) {
      if (e.name === 'AbortError') {
        return { error: 'Request timed out' };
      }
      console.error('[RFV] API call failed:', e);
      return { error: 'Failed to connect to API' };
    }
  }

  // ============================================
  // UI INJECTION
  // ============================================

  function removeExistingSidebar() {
    const existing = document.getElementById('rent-fair-value');
    if (existing) {
      existing.remove();
    }
  }

  function createSidebarContainer() {
    removeExistingSidebar();
    const sidebar = document.createElement('div');
    sidebar.id = 'rent-fair-value';
    document.body.appendChild(sidebar);
    return sidebar;
  }

  function injectLoadingState(message) {
    const sidebar = createSidebarContainer();
    sidebar.innerHTML = `
      <div class="rfv-container">
        <div class="rfv-header">RENT FAIR VALUE</div>
        <div class="rfv-loading">
          <div class="rfv-spinner"></div>
          <div class="rfv-loading-text">${escapeHtml(message)}</div>
        </div>
      </div>
    `;
  }

  function injectError(message) {
    const sidebar = createSidebarContainer();
    sidebar.innerHTML = `
      <div class="rfv-container">
        <div class="rfv-header">RENT FAIR VALUE</div>
        <div class="rfv-error">
          <div class="rfv-error-icon">⚠️</div>
          <div class="rfv-error-text">${escapeHtml(message)}</div>
        </div>
      </div>
    `;
  }

  function injectSidebar(result, sizeSource) {
    const {
      asking_price,
      fair_value,
      range_low,
      range_high,
      premium_pct,
      assessment,
      size_sqft,
      amenities_detected = [],
    } = result;

    // Determine color class
    const colorClass = assessment === 'overpriced' ? 'rfv-overpriced' :
                       assessment === 'good_deal' ? 'rfv-good-deal' : 'rfv-fair';

    const label = assessment.replace('_', ' ').toUpperCase();
    const sign = premium_pct > 0 ? '+' : '';

    // Size source note
    const sizeNote = sizeSource === 'estimated'
      ? 'Size estimated from beds/location'
      : sizeSource === 'ocr'
        ? `${size_sqft} sqft (from floorplan)`
        : `${size_sqft} sqft`;

    // Amenities HTML
    const amenitiesHtml = amenities_detected.length > 0
      ? `<div class="rfv-amenities">
          ${amenities_detected.map(a => `<span class="rfv-amenity">${escapeHtml(a)}</span>`).join('')}
         </div>`
      : '';

    const sidebar = createSidebarContainer();
    sidebar.innerHTML = `
      <div class="rfv-container">
        <div class="rfv-header">RENT FAIR VALUE</div>

        <div class="rfv-label">Asking</div>
        <div class="rfv-price">£${formatNumber(asking_price)}/mo</div>

        <hr class="rfv-divider">

        <div class="rfv-label">Model Estimate</div>
        <div class="rfv-price">£${formatNumber(fair_value)}/mo</div>
        <div class="rfv-range">Range: £${formatNumber(range_low)} – £${formatNumber(range_high)}</div>

        <div class="rfv-assessment ${colorClass}">
          <div class="rfv-assessment-value">${sign}${premium_pct}%</div>
          <div class="rfv-assessment-label">${label}</div>
        </div>

        ${amenitiesHtml}

        <div class="rfv-size-note">${escapeHtml(sizeNote)}</div>
        <div class="rfv-footer">Model V15 · Updated daily</div>
      </div>
    `;
  }

  // ============================================
  // UTILITIES
  // ============================================

  function formatNumber(num) {
    return num.toLocaleString('en-GB');
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

})();
