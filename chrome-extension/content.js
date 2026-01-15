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
    // Predictions cache hosted on GitHub (updated daily by GitHub Actions)
    PREDICTIONS_URL: 'https://raw.githubusercontent.com/kavanaghpatrick/london-rental-scraper/main/chrome-extension/api/predictions.json',
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

      // 5. Not in cache - show "not analyzed" message
      // (Local fallback removed - lookup table had 15-30% error vs real model)
      console.log('[RFV] Not in cache, property not yet analyzed');
      injectNotAnalyzed(askingPrice);

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

  function injectNotAnalyzed(askingPrice) {
    const sidebar = createSidebarContainer();
    sidebar.innerHTML = `
      <div class="rfv-container">
        <div class="rfv-header">RENT FAIR VALUE</div>

        <div class="rfv-label">Asking</div>
        <div class="rfv-price">£${formatNumber(askingPrice)}/mo</div>

        <hr class="rfv-divider">

        <div class="rfv-not-analyzed">
          <div class="rfv-not-analyzed-icon">📊</div>
          <div class="rfv-not-analyzed-text">Not yet in our database</div>
          <div class="rfv-not-analyzed-subtext">This property will be analyzed in our next daily update</div>
        </div>

        <div class="rfv-footer">Model V15 · Updated daily</div>
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
