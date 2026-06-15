/**
 * PostHog Analytics Module for Rent Fair Value Chrome Extension
 * Uses REST API directly (no npm dependencies needed for extension)
 *
 * Events tracked:
 * - extension_loaded: Extension initialized on a property page
 * - sidebar_shown: Sidebar displayed to user
 * - property_extracted: Property data successfully extracted
 * - prediction_completed: ML prediction generated
 * - prediction_failed: ML prediction failed
 * - ocr_initiated: Floorplan OCR started
 * - ocr_completed: Floorplan OCR succeeded
 * - ocr_failed: Floorplan OCR failed
 * - similar_properties_loaded: Comparable properties displayed
 * - compare_page_opened: User clicked Compare button
 * - navigation_detected: SPA navigation to new property
 * - error_occurred: Any error in the extension
 */

(function() {
  'use strict';

  const POSTHOG_API_KEY = 'phc_uQbCXX8hDY6LYEgDohL8hYggfx98qwe1yFSfM1vjEf0';
  const POSTHOG_HOST = 'https://us.i.posthog.com';

  // Event queue for batching
  let eventQueue = [];
  let flushTimeout = null;
  let distinctId = null;
  let sessionId = null;
  let isInitialized = false;

  // Generate or retrieve persistent distinct ID
  async function getDistinctId() {
    if (distinctId) return distinctId;

    try {
      const result = await chrome.storage.local.get('posthog_distinct_id');
      if (result.posthog_distinct_id) {
        distinctId = result.posthog_distinct_id;
      } else {
        // Generate new UUID
        distinctId = 'rfv_' + crypto.randomUUID();
        await chrome.storage.local.set({ posthog_distinct_id: distinctId });
      }
    } catch (e) {
      // Fallback if storage fails
      distinctId = 'rfv_' + crypto.randomUUID();
    }

    return distinctId;
  }

  // Generate session ID (per browser session)
  function getSessionId() {
    if (sessionId) return sessionId;
    sessionId = 'sess_' + crypto.randomUUID().slice(0, 8);
    return sessionId;
  }

  // Get extension version from manifest
  function getExtensionVersion() {
    try {
      return chrome.runtime.getManifest().version;
    } catch (e) {
      return 'unknown';
    }
  }

  // Detect current site
  function getCurrentSite() {
    const hostname = window.location.hostname;
    if (hostname.includes('rightmove')) return 'rightmove';
    if (hostname.includes('knightfrank')) return 'knightfrank';
    if (hostname.includes('chestertons')) return 'chestertons';
    if (hostname.includes('savills')) return 'savills';
    return 'unknown';
  }

  /**
   * Capture an analytics event
   * @param {string} eventName - Name of the event (snake_case)
   * @param {object} properties - Event properties
   */
  async function capture(eventName, properties = {}) {
    if (!isInitialized) {
      await init();
    }

    const event = {
      event: eventName,
      distinct_id: await getDistinctId(),
      properties: {
        // Standard properties
        $current_url: window.location.href,
        $host: window.location.hostname,
        $pathname: window.location.pathname,

        // Extension context
        extension_version: getExtensionVersion(),
        source_site: getCurrentSite(),
        session_id: getSessionId(),

        // Custom properties
        ...properties
      },
      timestamp: new Date().toISOString()
    };

    eventQueue.push(event);

    // Debounce flush (send after 2 seconds of no new events, or immediately if queue > 10)
    if (eventQueue.length >= 10) {
      flush();
    } else {
      clearTimeout(flushTimeout);
      flushTimeout = setTimeout(flush, 2000);
    }
  }

  /**
   * Send queued events to PostHog
   */
  async function flush() {
    if (eventQueue.length === 0) return;

    const events = [...eventQueue];
    eventQueue = [];
    clearTimeout(flushTimeout);

    try {
      const response = await fetch(`${POSTHOG_HOST}/batch/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          api_key: POSTHOG_API_KEY,
          batch: events
        })
      });

      if (!response.ok) {
        // Re-queue on failure (will retry on next capture)
        console.warn('[RFV Analytics] Batch failed, re-queuing', response.status);
        eventQueue = [...events, ...eventQueue];
      }
    } catch (error) {
      // Re-queue on network error
      console.warn('[RFV Analytics] Network error, re-queuing', error.message);
      eventQueue = [...events, ...eventQueue];
    }
  }

  /**
   * Initialize analytics
   */
  async function init() {
    if (isInitialized) return;

    await getDistinctId();
    getSessionId();
    isInitialized = true;

    // Flush on page unload
    window.addEventListener('beforeunload', () => {
      if (eventQueue.length > 0) {
        // Use sendBeacon for reliable delivery on unload
        const payload = JSON.stringify({
          api_key: POSTHOG_API_KEY,
          batch: eventQueue
        });
        navigator.sendBeacon(`${POSTHOG_HOST}/batch/`, payload);
      }
    });
  }

  /**
   * Capture exception/error
   * @param {Error} error - The error object
   * @param {object} context - Additional context
   */
  async function captureException(error, context = {}) {
    await capture('error_occurred', {
      error_type: error.name || 'Error',
      error_message: error.message || 'Unknown error',
      error_stack: error.stack?.slice(0, 500) || '',
      ...context
    });
  }

  // ============================================
  // EVENT HELPERS - Called from content.js
  // ============================================

  const Analytics = {
    init,
    capture,
    captureException,
    flush,

    // Extension lifecycle
    extensionLoaded: (site) => capture('extension_loaded', {
      source_site: site,
      page_url: window.location.href
    }),

    sidebarShown: (trigger = 'auto') => capture('sidebar_shown', {
      trigger_method: trigger
    }),

    sidebarDismissed: (reason = 'unknown') => capture('sidebar_dismissed', {
      dismiss_reason: reason
    }),

    // Property extraction
    propertyExtracted: (data) => capture('property_extracted', {
      has_price: !!data.price,
      has_postcode: !!data.postcode,
      has_bedrooms: !!data.bedrooms,
      has_sqft: !!data.sqft,
      sqft_source: data.sqftSource || 'unknown',
      property_type: data.propertyType || 'unknown',
      bedrooms: data.bedrooms || 0,
      extraction_method: data.extractionMethod || 'dom'
    }),

    propertyExtractionFailed: (reason, site) => capture('property_extraction_failed', {
      failure_reason: reason,
      source_site: site
    }),

    // ML Prediction
    predictionCompleted: (result) => capture('prediction_completed', {
      asking_price: result.askingPrice,
      fair_value: result.fairValue,
      premium_pct: result.premiumPct,
      assessment: result.assessment,
      size_sqft: result.sizeSqft,
      size_source: result.sizeSource,
      bedrooms: result.bedrooms,
      postcode_district: result.postcodeDistrict,
      model_version: 'v20'
    }),

    predictionFailed: (reason, context = {}) => capture('prediction_failed', {
      failure_reason: reason,
      ...context
    }),

    // OCR
    ocrInitiated: (imageSource) => capture('ocr_initiated', {
      image_source: imageSource,
      trigger: 'auto'
    }),

    ocrCompleted: (result) => capture('ocr_completed', {
      extracted_sqft: result.sqft,
      has_text: result.hasText,
      ocr_confidence: result.confidence || null,
      processing_time_ms: result.processingTimeMs || null
    }),

    ocrFailed: (reason, imageSource) => capture('ocr_failed', {
      failure_reason: reason,
      image_source: imageSource
    }),

    // Similar properties
    similarPropertiesLoaded: (count) => capture('similar_properties_loaded', {
      num_properties: count
    }),

    similarPropertyClicked: (position, price) => capture('similar_property_clicked', {
      position: position,
      property_price: price
    }),

    comparePageOpened: (propertyData) => capture('compare_page_opened', {
      asking_price: propertyData.askingPrice,
      fair_value: propertyData.fairValue,
      bedrooms: propertyData.bedrooms
    }),

    // Navigation
    navigationDetected: (fromUrl, toUrl) => capture('navigation_detected', {
      from_site: getCurrentSite(),
      is_property_page: true
    }),

    // Short-term let detection
    shortLetDetected: (price) => capture('short_let_detected', {
      asking_price: price
    }),

    // Feature flag tracking (for future use)
    featureFlagCalled: (flagName, value) => capture('$feature_flag_called', {
      $feature_flag: flagName,
      $feature_flag_response: value
    })
  };

  // Expose globally for content.js to use
  window.RFVAnalytics = Analytics;

})();
