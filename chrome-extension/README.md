# Rent Fair Value - Chrome Extension

ML-powered fair rent estimates for London properties on Rightmove.

## Structure

```
chrome-extension/
├── manifest.json       # Chrome extension manifest
├── content.js          # Main script (data extraction, OCR, UI)
├── sidebar.css         # Sidebar styles
├── lib/
│   └── tesseract.min.js  # Tesseract.js for client-side OCR
├── icons/              # Extension icons
│   ├── icon16.png
│   ├── icon48.png
│   └── icon128.png
└── api/                # Client-side model files
    ├── model.json      # XGBoost V20 model (JSON format)
    ├── features.json   # Feature column names
    └── predictions.json # Pre-computed predictions cache
```

## Local Testing

### 1. Load Extension in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select this `chrome-extension` directory

### 2. Test on Rightmove

1. Go to any Rightmove listing: https://www.rightmove.co.uk/properties/
2. The extension should show a sidebar with:
   - Asking price
   - Model estimate
   - Premium/discount percentage

## Deployment

### Publish to Chrome Web Store

1. Create ZIP: `zip -r rent-fair-value.zip . -x "api/*" -x "*.git*"`
2. Go to [Chrome Developer Dashboard](https://chrome.google.com/webstore/devconsole)
3. Pay $5 registration fee (one-time)
4. Upload ZIP and submit for review

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                 Chrome Extension (Client-Side)               │
│  1. Extract propertyData from __NEXT_DATA__ JSON            │
│  2. Try to get sqft from page JSON                          │
│  3. If no sqft, OCR floorplan with Tesseract.js             │
│  4. Engineer 143 features (xgboost.js)                      │
│  5. Run XGBoost V20 model locally (model.json)              │
│  6. Display fair value in sidebar                           │
└─────────────────────────────────────────────────────────────┘
```

All processing happens client-side - no API calls required.

## Configuration

### OCR Timeout

Default: 60 seconds. Adjust in `content.js`:
```javascript
const CONFIG = {
  OCR_TIMEOUT: 60000,
  // ...
};
```
