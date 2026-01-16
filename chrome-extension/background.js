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
        console.error('[RFV Background] Fetch failed:', error.message);
        sendResponse({ success: false, error: error.message });
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
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
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
