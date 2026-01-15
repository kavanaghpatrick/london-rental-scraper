# PRD: Chrome Extension MVP (Simplified)

## One-Sentence Summary

Chrome extension that shows ML fair value estimate on Rightmove listings, with accurate pricing powered by the same model we train daily.

---

## Why Data Quality Matters

Our V15 model's top features by importance:
| Feature | Importance | Source |
|---------|------------|--------|
| **size_bin** | 0.081 | sqft (from page JSON or estimation) |
| size_squared | 0.025 | sqft |
| log_sqft | 0.021 | sqft |
| beds_x_central | 0.014 | beds × location |
| **has_view** | 0.014 | description parsing |
| **has_outdoor_space** | 0.013 | description parsing |

**Sqft handling for MVP**: Use page JSON if available → client-side OCR via Tesseract.js → fall back to beds/postcode estimation.

---

## Core Flow (Dead Simple)

```typescript
// 1. Extension grabs propertyData
const propertyData = JSON.parse(document.getElementById('__NEXT_DATA__').textContent)
  .props.pageProps.propertyData;

// 2. Get sqft: page JSON → OCR floorplan → estimate
let size_sqft = extractSqftFromSizings(propertyData.sizings);
if (!size_sqft && propertyData.floorplans?.[0]?.url) {
  size_sqft = await ocrFloorplan(propertyData.floorplans[0].url);  // Tesseract.js
}

// 3. Send to API (with sqft already resolved)
const result = await fetch(API_URL, {
  headers: { 'X-API-Key': API_KEY },
  body: JSON.stringify({ source: 'rightmove', property: propertyData, size_sqft })
});

// 4. Show result
injectSidebar(result);
```

**Key insight**: OCR runs in the extension (Tesseract.js), not the server.
This avoids serverless limitations and keeps the API fast.

---

## MVP Scope (1 Week)

### Extension Extracts (What It Can See)
- ✅ Price (pcm or pw - API converts)
- ✅ Beds, baths, postcode
- ✅ Size sqft (if in page JSON)
- ✅ **Full description text** (critical for amenity parsing)
- ✅ **Floorplan URL** (for API to OCR if sqft missing)
- ✅ Property type, lat/long (if available)

### Extension Processes (Client-Side)
- ✅ Extract propertyData from `__NEXT_DATA__`
- ✅ Get sqft from page JSON OR **OCR floorplan via Tesseract.js**
- ✅ Send pre-processed data to API

### API Processes (Server-Side)
- ✅ Convert pw → pcm
- ✅ **Parse amenities from description** (same regex as `parse_amenities()`)
- ✅ Use sqft from extension (or estimate from beds/postcode as fallback)
- ✅ Engineer all 93 features
- ✅ Run XGBoost prediction
- ✅ **API key validation** (prevent abuse)

### Exclude from MVP
- ❌ Other 4 sites (add after validating on Rightmove)
- ❌ Save/compare feature
- ❌ Settings panel
- ❌ Client-side caching

---

## Extension Code (~150 lines total)

### manifest.json
```json
{
  "manifest_version": 3,
  "name": "Rent Fair Value",
  "version": "0.1.0",
  "content_scripts": [{
    "matches": ["https://www.rightmove.co.uk/properties/*"],
    "js": ["tesseract.min.js", "content.js"],
    "css": ["sidebar.css"]
  }],
  "permissions": ["activeTab"],
  "host_permissions": ["https://media.rightmove.co.uk/*"]
}
```

**Note**: `host_permissions` allows fetching floorplan images (bypasses CORS).

### content.js (~80 lines)
```typescript
const API_KEY = 'your-api-key-here';  // Hardcoded for MVP (env vars in V2)
const API_URL = 'https://your-api.vercel.app/api/valuate';

(async function() {
  // 1. Grab propertyData
  const script = document.getElementById('__NEXT_DATA__');
  if (!script) return;

  const nextData = JSON.parse(script.textContent);
  const propertyData = nextData?.props?.pageProps?.propertyData;
  if (!propertyData) return;

  // 2. Try to get sqft from page JSON
  let size_sqft = extractSqftFromSizings(propertyData.sizings);

  // 3. If no sqft, try OCR on floorplan (client-side via Tesseract.js)
  if (!size_sqft && propertyData.floorplans?.[0]?.url) {
    injectLoadingState('Analyzing floorplan...');
    size_sqft = await ocrFloorplan(propertyData.floorplans[0].url);
  }

  // 4. Send to API
  injectLoadingState('Getting estimate...');
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
    body: JSON.stringify({ source: 'rightmove', property: propertyData, size_sqft })
  });

  if (!response.ok) {
    injectError('Valuation failed');
    return;
  }

  const result = await response.json();
  injectSidebar(result);
})();

// Extract sqft from Rightmove sizings array
function extractSqftFromSizings(sizings) {
  if (!sizings?.length) return null;
  const sqftEntry = sizings.find(s => s.unit === 'sqft');
  return sqftEntry ? parseInt(sqftEntry.minimumSize || sqftEntry.maximumSize) : null;
}

// OCR floorplan using Tesseract.js (runs in browser)
async function ocrFloorplan(url) {
  try {
    const { data: { text } } = await Tesseract.recognize(url, 'eng', {
      logger: m => console.log(m)  // Progress logging
    });
    // Same regex as our server-side OCR
    const match = text.match(/(\d{2,4})\s*(?:sq\.?\s*ft|sqft|square\s*feet)/i);
    return match ? parseInt(match[1].replace(',', '')) : null;
  } catch (e) {
    console.error('OCR failed:', e);
    return null;
  }
}

function injectSidebar(result) {
  // API returns everything we need to display
  const { asking_price, fair_value, range_low, range_high, premium_pct, assessment } = result;

  const color = assessment === 'overpriced' ? '#ef4444' :
                assessment === 'good_deal' ? '#22c55e' : '#3b82f6';
  const label = assessment.toUpperCase().replace('_', ' ');

  const sidebar = document.createElement('div');
  sidebar.id = 'rent-fair-value';
  sidebar.innerHTML = `
    <div style="position:fixed;top:100px;right:20px;width:280px;background:#1f2937;color:white;padding:20px;border-radius:12px;z-index:9999;font-family:system-ui;box-shadow:0 4px 20px rgba(0,0,0,0.3);">
      <div style="font-size:14px;font-weight:bold;margin-bottom:12px;">RENT FAIR VALUE</div>
      <div style="font-size:12px;opacity:0.7;margin-bottom:4px;">ASKING</div>
      <div style="font-size:24px;font-weight:bold;">£${asking_price.toLocaleString()}/mo</div>
      <hr style="border-color:#374151;margin:16px 0;">
      <div style="font-size:12px;opacity:0.7;margin-bottom:4px;">MODEL ESTIMATE</div>
      <div style="font-size:24px;font-weight:bold;">£${fair_value.toLocaleString()}/mo</div>
      <div style="font-size:11px;opacity:0.6;">Range: £${range_low.toLocaleString()} - £${range_high.toLocaleString()}</div>
      <hr style="border-color:#374151;margin:16px 0;">
      <div style="text-align:center;padding:12px;background:${color};border-radius:8px;">
        <div style="font-size:20px;font-weight:bold;">${premium_pct > 0 ? '+' : ''}${premium_pct}%</div>
        <div style="font-size:12px;">${label}</div>
      </div>
      <div style="font-size:10px;opacity:0.4;margin-top:12px;text-align:center;">
        Model V15 · Updated daily
      </div>
    </div>
  `;
  document.body.appendChild(sidebar);
}
```

---

## API Endpoint

**URL**: `POST /api/valuate`

**Request** (extension sends propertyData + optional sqft from OCR):
```json
{
  "source": "rightmove",
  "property": { /* propertyData object */ },
  "size_sqft": 850  // From page JSON or client-side OCR (null if neither worked)
}
```

**Headers** (required):
```
X-API-Key: your-api-key
```

**API Processing Pipeline** (lightweight - no OCR):
```python
# api/valuate.py
from rental_price_models_v15 import parse_amenities, engineer_features_v15

def valuate(request):
    # 0. Validate API key
    if request.headers.get('X-API-Key') != API_KEY:
        return {'error': 'Invalid API key'}, 401

    data = request.json
    property_data = data['property']

    # Extract fields (same as rightmove_spider.py)
    price_text = property_data['prices']['primaryPrice']
    price_pcm = parse_price(price_text)  # Handles pw/pcm conversion

    beds = property_data['bedrooms']
    baths = property_data['bathrooms']
    postcode = f"{property_data['address']['outcode']} {property_data['address']['incode']}"
    description = property_data['text']['description']
    features = property_data.get('keyFeatures', [])

    # 1. Use sqft from extension (page JSON or client-side OCR)
    size_sqft = data.get('size_sqft')
    size_source = 'extension'

    # 2. Fallback: estimate from beds + postcode if extension didn't get it
    if not size_sqft:
        district = postcode.split()[0]
        size_sqft = estimate_size(district, beds)  # Lookup from training data
        size_source = 'estimated'

    # 3. Parse amenities from description (same regex as training)
    amenities = parse_amenities(description + ' ' + ' '.join(features))

    # 4. Engineer all 93 features (same as training)
    row = {
        'bedrooms': beds, 'bathrooms': baths, 'size_sqft': size_sqft,
        'postcode': postcode, 'description': description,
        **amenities
    }
    features_df = engineer_features_v15(pd.DataFrame([row]))

    # 5. Run model
    pred_log = model.predict(features_df[feature_cols])[0]
    fair_value = int(np.expm1(pred_log))

    # 6. Return everything for display
    premium_pct = round((price_pcm / fair_value - 1) * 100, 1)

    return {
        'asking_price': price_pcm,
        'fair_value': fair_value,
        'range_low': int(fair_value * 0.79),
        'range_high': int(fair_value * 1.21),
        'premium_pct': premium_pct,
        'assessment': 'overpriced' if premium_pct > 15 else 'good_deal' if premium_pct < -10 else 'fair',
        'size_sqft': size_sqft,
        'size_source': size_source,
        'amenities_detected': [k.replace('has_', '') for k, v in amenities.items() if v],
    }
```

**Response**:
```json
{
  "asking_price": 10914,
  "fair_value": 8728,
  "range_low": 6895,
  "range_high": 10561,
  "premium_pct": 25.0,
  "assessment": "overpriced",
  "size_sqft": 1312,
  "amenities_detected": ["balcony", "ac", "lift", "view", "garden"]
}
```

**Key Insight**: The API literally imports and reuses:
- `rightmove_spider.py` parsing logic
- `rental_price_models_v15.py` amenity parsing + feature engineering
- `scripts/ocr_enrich.py` for floorplan OCR
- The trained model pickle file

---

## Timeline (1 Week)

| Day | Task |
|-----|------|
| **1** | Extension skeleton + Rightmove extraction |
| **2** | API endpoint (reuse model code) |
| **3** | Sidebar UI + integration |
| **4** | Test on 20 real listings |
| **5** | Bug fixes + publish to Chrome Web Store |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Works on Rightmove | 100% of listing pages |
| API response | < 500ms |
| User can see estimate | Within 2s of page load |

---

## What's NOT in MVP

Explicitly deferred to v2 (only if MVP validates):

1. Other sites (Savills, Knight Frank, Foxtons, Chestertons)
2. Save/compare properties
3. Settings panel
4. Caching
5. Feature impact breakdown
6. Browser-side amenity parsing

---

## V2 Expansion (Only If MVP Works)

Add one site at a time:
1. Foxtons (also uses `__NEXT_DATA__`)
2. Savills (DOM scraping)
3. Knight Frank (DOM scraping)
4. Chestertons (DOM scraping)

Each site = ~1 day of work.

---

## Technical Notes

### Why Client-Side OCR?

**Problem**: Server-side OCR (Tesseract) won't work on Vercel/serverless:
- Tesseract requires binary dependencies (~30MB)
- Vercel limits: 50MB zipped, 250MB unzipped
- OCR takes 5-30s, serverless timeouts at 10-30s
- Cold starts with XGBoost + Pandas + Tesseract = 5s+

**Solution**: Run OCR in the extension via Tesseract.js
- Pure JavaScript, no server dependencies
- User's browser does the compute
- API stays lightweight and fast (<500ms)

### Tesseract.js Details

```
Bundle size: ~2MB (tesseract.min.js)
Language data: ~10MB (downloaded on first use, cached)
OCR time: 5-30s depending on floorplan complexity
```

**CORS handling**: Chrome extensions can fetch cross-origin images with `host_permissions`:
```json
"host_permissions": ["https://media.rightmove.co.uk/*"]
```

### Sqft Fallback Chain

1. **Page JSON** (`propertyData.sizings`) - instant, most reliable
2. **Client-side OCR** (Tesseract.js on floorplan) - 5-30s, good accuracy
3. **Beds/postcode estimation** (server-side lookup) - instant, ~80% accurate

### API Key Security

MVP uses hardcoded API key in extension. For V2:
- Store key in Chrome storage API
- Prompt user for key on first use
- Or use OAuth flow with your backend
