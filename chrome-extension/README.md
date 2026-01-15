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
├── api/                # Vercel serverless API
│   ├── valuate.py      # Main API endpoint
│   ├── requirements.txt
│   ├── size_lookup.json
│   ├── rental_model_v15.pkl
│   └── rental_model_v15_features.pkl
└── vercel.json         # Vercel configuration
```

## Local Testing

### 1. Load Extension in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select this `chrome-extension` directory

### 2. Test API Locally (optional)

```bash
cd api
pip install -r requirements.txt
python -c "
from valuate import handler
# Test the handler directly
"
```

### 3. Test on Rightmove

1. Go to any Rightmove listing: https://www.rightmove.co.uk/properties/
2. The extension should show a sidebar with:
   - Asking price
   - Model estimate
   - Premium/discount percentage

## Deployment

### Deploy API to Vercel

```bash
cd chrome-extension
vercel --prod
```

Set environment variable:
```bash
vercel env add RFV_API_KEY
# Enter: rfv-mvp-key-2024
```

### Update Extension API URL

Edit `content.js` and update:
```javascript
const CONFIG = {
  API_URL: 'https://your-vercel-app.vercel.app/api/valuate',
  // ...
};
```

### Publish to Chrome Web Store

1. Create ZIP: `zip -r rent-fair-value.zip . -x "api/*" -x "*.git*"`
2. Go to [Chrome Developer Dashboard](https://chrome.google.com/webstore/devconsole)
3. Pay $5 registration fee (one-time)
4. Upload ZIP and submit for review

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                     Chrome Extension                         │
│  1. Extract propertyData from __NEXT_DATA__ JSON            │
│  2. Try to get sqft from page JSON                          │
│  3. If no sqft, OCR floorplan with Tesseract.js             │
│  4. Send data to API                                         │
│  5. Display fair value in sidebar                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Vercel API (/api/valuate)               │
│  1. Parse amenities from description                        │
│  2. Estimate sqft if not provided (beds × postcode)         │
│  3. Engineer all 93 features                                │
│  4. Run XGBoost V15 model                                   │
│  5. Return fair value estimate                              │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### API Key

MVP uses hardcoded API key. For production:
- Store in `chrome.storage.sync`
- Or implement OAuth flow

### OCR Timeout

Default: 60 seconds. Adjust in `content.js`:
```javascript
const CONFIG = {
  OCR_TIMEOUT: 60000,
  // ...
};
```
