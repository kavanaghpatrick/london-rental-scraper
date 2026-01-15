/**
 * Rent Fair Value - Chrome Extension
 * Shows ML-powered fair rent estimates on Rightmove listings
 */

(function() {
  'use strict';

  // Configuration
  const CONFIG = {
    // Predictions cache hosted on GitHub (updated daily)
    PREDICTIONS_URL: 'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/chrome-extension/api/predictions.json',
    OCR_TIMEOUT: 60000,  // 60s max for OCR
  };

  // Prevent duplicate execution
  if (window.__rentFairValueLoaded) return;
  window.__rentFairValueLoaded = true;

  // Predictions cache (loaded once)
  let predictionsCache = null;

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
    const beds = propertyData.bedrooms || 1;
    const baths = propertyData.bathrooms || 1;
    const outcode = propertyData.address?.outcode || 'SW3';

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
      // Estimate from beds (simple heuristic)
      sizeSqft = estimateSize(outcode, beds);
      sizeSource = 'estimated';
    }

    // Simple fair value estimate (regression-based heuristic)
    // Based on model coefficients: ~£4.5/sqft base + location premium
    const isPrime = ['SW1', 'SW3', 'SW7', 'SW10', 'W1', 'W8', 'W11', 'NW3', 'NW8'].some(p => outcode.startsWith(p));
    const basePPSF = isPrime ? 5.5 : 4.0;
    const bedsBonus = beds * 200;
    const bathsBonus = (baths - 1) * 150;

    const fairValue = Math.round(sizeSqft * basePPSF + bedsBonus + bathsBonus);
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

  function estimateSize(postcode, beds) {
    // Simple size estimation by beds and area
    const baseSizes = { 0: 350, 1: 500, 2: 750, 3: 1000, 4: 1300, 5: 1600 };
    const base = baseSizes[Math.min(beds, 5)] || 500;

    // Prime areas tend to have smaller sqft per bed
    const isPrime = ['SW1', 'SW3', 'SW7', 'W1', 'W8'].some(p => postcode.startsWith(p));
    return isPrime ? Math.round(base * 0.9) : base;
  }

  // ============================================
  // DATA EXTRACTION
  // ============================================

  function extractPropertyData() {
    const script = document.getElementById('__NEXT_DATA__');
    if (!script) return null;

    try {
      const nextData = JSON.parse(script.textContent);
      const propertyData = nextData?.props?.pageProps?.propertyData;

      if (!propertyData) {
        console.log('[RFV] No propertyData in __NEXT_DATA__');
        return null;
      }

      return propertyData;
    } catch (e) {
      console.error('[RFV] Failed to parse __NEXT_DATA__:', e);
      return null;
    }
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
