/**
 * Background Service Worker for Rent Fair Value Extension
 * Handles cross-origin image fetching for OCR (content scripts can't bypass CORS)
 */

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'fetchImage') {
    console.log('[RFV Background] Fetching image:', request.url);

    fetchImageAsBase64(request.url)
      .then(base64 => {
        console.log('[RFV Background] Image fetched successfully, size:', base64.length);
        sendResponse({ success: true, data: base64 });
      })
      .catch(error => {
        // A blocked/failed cross-origin floorplan fetch (e.g. Foxtons CDN HTTP 403
        // bot-mitigation) is EXPECTED and recoverable: the content script falls back
        // to page sqft / estimateSqft and renders normally. Log at debug level so a
        // hostile CDN can't flood the service-worker console with error-level noise.
        // Still report the failure to the caller so its OCR path returns cleanly.
        const msg = (error && error.message) || 'fetch blocked';
        console.debug('[RFV Background] Floorplan fetch unavailable:', msg);
        sendResponse({ success: false, error: msg });
      });

    return true; // Required for async sendResponse
  }
});

async function fetchImageAsBase64(url) {
  const response = await fetch(url, {
    credentials: 'omit',
    mode: 'cors',
  });

  if (!response.ok) {
    // Some CDNs (e.g. Foxtons) return an empty statusText with a 403, which would
    // produce a dangling "HTTP 403:" — include statusText only when present.
    const detail = response.statusText ? `HTTP ${response.status}: ${response.statusText}`
                                       : `HTTP ${response.status}`;
    throw new Error(detail);
  }

  const blob = await response.blob();
  console.log('[RFV Background] Blob received:', blob.size, 'bytes, type:', blob.type);

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('FileReader failed'));
    reader.readAsDataURL(blob);
  });
}

console.log('[RFV Background] Service worker loaded');
