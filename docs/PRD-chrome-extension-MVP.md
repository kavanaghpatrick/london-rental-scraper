# PRD: Chrome Extension MVP (Simplified)

## One-Sentence Summary

Chrome extension that shows ML fair value estimate on Rightmove listings, with accurate pricing powered by the same model we train daily.

---

## Why Data Quality Matters

Our V15 model's top features by importance:
| Feature | Importance | Source |
|---------|------------|--------|
| **size_bin** | 0.081 | sqft (from page or OCR) |
| size_squared | 0.025 | sqft |
| log_sqft | 0.021 | sqft |
| beds_x_central | 0.014 | beds × location |
| **has_view** | 0.014 | description parsing |
| **has_outdoor_space** | 0.013 | description parsing |

**Without accurate sqft, predictions are significantly worse.** This is why we need floorplan OCR as a fallback.

---

## Core Flow (Dead Simple)

```typescript
// Extension just grabs the raw page data (like our spider does)
const nextData = document.getElementById('__NEXT_DATA__').textContent;

// Send to API - it does ALL the parsing (same as rightmove_spider.py)
const result = await fetch(API_URL, { body: nextData });

// Show result
injectSidebar(result);
```

**The extension is just a "manual scrape trigger" for a single page.**
The API reuses the exact same parsing logic as our spiders.

---

## MVP Scope (1 Week)

### Extension Extracts (What It Can See)
- ✅ Price (pcm or pw - API converts)
- ✅ Beds, baths, postcode
- ✅ Size sqft (if in page JSON)
- ✅ **Full description text** (critical for amenity parsing)
- ✅ **Floorplan URL** (for API to OCR if sqft missing)
- ✅ Property type, lat/long (if available)

### API Processes (Same Code as Training)
- ✅ Convert pw → pcm
- ✅ **Parse amenities from description** (same regex as `parse_amenities()`)
- ✅ **OCR floorplan** if sqft missing (same code as `ocr_enrich.py`)
- ✅ Estimate size from beds/postcode if OCR fails
- ✅ Engineer all 93 features
- ✅ Run XGBoost prediction

### Exclude from MVP
- ❌ Other 4 sites (add after validating on Rightmove)
- ❌ Save/compare feature
- ❌ Settings panel
- ❌ Client-side caching

---

## Extension Code (~100 lines total)

### manifest.json
```json
{
  "manifest_version": 3,
  "name": "Rent Fair Value",
  "version": "0.1.0",
  "content_scripts": [{
    "matches": ["https://www.rightmove.co.uk/properties/*"],
    "js": ["content.js"],
    "css": ["sidebar.css"]
  }],
  "permissions": ["activeTab"]
}
```

### content.js (~50 lines)
```typescript
(async function() {
  // 1. Grab the raw JSON (same data our spider extracts)
  const script = document.getElementById('__NEXT_DATA__');
  if (!script) return;

  // 2. Send to API - it parses EXACTLY like rightmove_spider.py
  const response = await fetch('https://your-api.vercel.app/api/valuate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      source: 'rightmove',
      raw_json: script.textContent  // API parses this
    })
  });

  if (!response.ok) {
    console.error('Valuation API failed');
    return;
  }

  const result = await response.json();
  injectSidebar(result);
})();

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

**Request** (from extension - just raw JSON):
```json
{
  "source": "rightmove",
  "raw_json": "{\"props\":{\"pageProps\":{\"propertyData\":{...}}}}"
}
```

**API Processing Pipeline** (reuses ALL existing code):
```python
# api/valuate.py
from property_scraper.spiders.rightmove_spider import parse_rightmove_json
from rental_price_models_v15 import parse_amenities, engineer_features_v15
from scripts.ocr_enrich import extract_sqft_from_floorplan

def valuate(request):
    # 1. Parse JSON exactly like rightmove_spider.py does
    data = json.loads(request.raw_json)
    property_data = data['props']['pageProps']['propertyData']

    # Extract same fields as spider
    price_text = property_data['prices']['primaryPrice']
    price_pcm = parse_price(price_text)  # Handles pw/pcm conversion

    beds = property_data['bedrooms']
    baths = property_data['bathrooms']
    postcode = f"{property_data['address']['outcode']} {property_data['address']['incode']}"
    description = property_data['text']['description']
    features = property_data.get('keyFeatures', [])
    floorplan_url = property_data.get('floorplans', [{}])[0].get('url')

    # 2. Get sqft - same logic as our enrichment pipeline
    size_sqft = extract_sqft_from_sizings(property_data.get('sizings', []))
    if not size_sqft and floorplan_url:
        size_sqft = extract_sqft_from_floorplan(floorplan_url)  # OCR!

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
