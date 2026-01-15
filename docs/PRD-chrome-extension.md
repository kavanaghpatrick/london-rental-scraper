# PRD: London Rental Fair Value Chrome Extension

## Overview

A Chrome extension that displays ML-powered fair value estimates when users browse rental property listings on major UK property portals. The extension extracts property details from the page and queries our trained model to show whether a listing is fairly priced, overpriced, or a good deal.

---

## Problem Statement

Renters browsing property listings have no objective way to assess whether an asking price is fair. They must:
- Manually compare dozens of listings
- Guess at price-per-sqft benchmarks
- Miss overpriced properties or undervalue good deals

Our model (V15, R²=0.73) already predicts fair rental values with 21% MAPE. This extension brings that intelligence directly into the browsing experience.

---

## Goals

| Goal | Metric | Target |
|------|--------|--------|
| **User Value** | Time to assess fair value | < 2 seconds (instant on page load) |
| **Accuracy** | Prediction vs actual market | Within 21% (model MAPE) |
| **Coverage** | Supported property portals | 5 major sites at launch |
| **Adoption** | Active weekly users | 1,000 in first 3 months |

---

## User Stories

### Primary User: Renter Searching for Property

1. **As a renter**, I want to see a fair value estimate when viewing a listing, so I know if the asking price is reasonable.

2. **As a renter**, I want to see how the asking price compares to similar properties, so I can negotiate effectively.

3. **As a renter**, I want to see which property features are driving the valuation, so I understand what I'm paying for.

4. **As a renter**, I want to save listings with their valuations, so I can compare properties later.

### Secondary User: Landlord/Agent

5. **As a landlord**, I want to check if my asking price is competitive, so I can price appropriately.

---

## Supported Sites (Launch)

| Site | Priority | Monthly UK Visits | Extraction Complexity |
|------|----------|-------------------|----------------------|
| Rightmove | P0 | 150M+ | Medium (JSON in page) |
| Zoopla | P0 | 50M+ | Medium (JSON in page) |
| OnTheMarket | P1 | 20M+ | Medium |
| Foxtons | P1 | 5M+ | Easy (clean JSON) |
| OpenRent | P2 | 10M+ | Easy |

---

## Feature Specifications

### F1: Fair Value Sidebar Panel

**Trigger**: Automatically appears when user loads a property listing page.

**Display Elements**:

```
┌─────────────────────────────────┐
│  FAIR VALUE ESTIMATE            │
│  ════════════════════════════   │
│                                 │
│  Asking:     £10,914/month      │
│  Fair Value: £8,728/month       │
│  Range:      £6,895 - £10,561   │
│                                 │
│  ┌─────────────────────────┐    │
│  │ ████████████░░░░ +25%   │    │
│  │     OVERPRICED          │    │
│  └─────────────────────────┘    │
│                                 │
│  Save to Compare  [★]           │
└─────────────────────────────────┘
```

**Price Assessment Categories**:
| Category | Condition | Color |
|----------|-----------|-------|
| Great Deal | Asking < Fair × 0.90 | Green |
| Fair Price | Fair × 0.90 ≤ Asking ≤ Fair × 1.10 | Blue |
| Slightly High | Fair × 1.10 < Asking ≤ Fair × 1.25 | Amber |
| Overpriced | Asking > Fair × 1.25 | Red |

---

### F2: Property Details Extraction

The extension must extract these fields from listing pages:

| Field | Required | Extraction Method |
|-------|----------|-------------------|
| Address | Yes | Page scraping / JSON |
| Postcode | Yes | Regex from address |
| Asking Price | Yes | Page scraping / JSON |
| Price Period | Yes | Detect pcm vs pw |
| Bedrooms | Yes | Page scraping / JSON |
| Bathrooms | Yes | Page scraping / JSON |
| Size (sqft) | Preferred | Page scraping / JSON |
| Property Type | Preferred | Page scraping / JSON |
| Amenities | Optional | Text analysis |
| Latitude/Longitude | Optional | JSON or geocode |

**Handling Missing Size Data**:
- If sqft not on page, estimate from beds using postcode-specific median
- Show "Estimated size: ~X sqft" with disclaimer
- Model handles this via `size_bin` feature

---

### F3: Valuation API

**Endpoint**: `POST /api/v1/valuate`

**Request**:
```json
{
  "address": "123 Example Street",
  "postcode": "SW1W 9JA",
  "bedrooms": 2,
  "bathrooms": 2,
  "size_sqft": 1312,
  "property_type": "flat",
  "amenities": {
    "has_balcony": true,
    "has_porter": false,
    "has_gym": false,
    "has_ac": true
  },
  "asking_price_pcm": 10914
}
```

**Response**:
```json
{
  "fair_value_pcm": 8728,
  "range_low": 6895,
  "range_high": 10561,
  "ppsf": 6.65,
  "market_percentile": 85,
  "assessment": "overpriced",
  "premium_pct": 25.0,
  "model_version": "v15",
  "confidence": "high",
  "size_estimated": false,
  "comparable_count": 47,
  "feature_impacts": [
    {"feature": "Prime Location (SW1)", "impact": "+15%"},
    {"feature": "Air Conditioning", "impact": "+5%"},
    {"feature": "No Pool/Gym", "impact": "-8%"}
  ]
}
```

---

### F4: Feature Impact Breakdown

Show users which features are driving the valuation up or down:

```
┌─────────────────────────────────┐
│  WHAT'S DRIVING THIS PRICE      │
│  ═══════════════════════════    │
│                                 │
│  ▲ Prime SW1 Location    +15%   │
│  ▲ Air Conditioning      +5%    │
│  ▲ Balcony              +3%    │
│  ▼ No Pool/Gym          -8%    │
│  ▼ No Porter            -5%    │
│                                 │
│  Net Premium: +10%              │
└─────────────────────────────────┘
```

---

### F5: Comparison List

Users can save listings to compare later:

```
┌─────────────────────────────────┐
│  SAVED PROPERTIES (3)           │
│  ═══════════════════════════    │
│                                 │
│  1. 123 Example St, SW1         │
│     £10,914 → Fair: £8,728      │
│     ████████░░ +25% OVERPRICED  │
│                                 │
│  2. 45 Another Rd, SW3          │
│     £8,500 → Fair: £8,200       │
│     █████████░ +4% FAIR         │
│                                 │
│  3. 78 Third Ave, W1            │
│     £7,800 → Fair: £9,100       │
│     ████░░░░░░ -14% GREAT DEAL  │
│                                 │
│  [Export to CSV]                │
└─────────────────────────────────┘
```

---

### F6: Settings Panel

```
┌─────────────────────────────────┐
│  SETTINGS                       │
│  ═══════════════════════════    │
│                                 │
│  Auto-show panel: [✓]           │
│  Panel position:  [Right ▼]     │
│  Show on:                       │
│    [✓] Rightmove                │
│    [✓] Zoopla                   │
│    [✓] Foxtons                  │
│    [ ] OnTheMarket              │
│                                 │
│  Price display:   [Monthly ▼]   │
│  Size units:      [Sqft ▼]      │
│                                 │
│  [Clear Saved Properties]       │
└─────────────────────────────────┘
```

---

## Technical Architecture

### Extension Components

```
┌─────────────────────────────────────────────────────────┐
│                    Chrome Extension                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Content    │  │  Background  │  │    Popup     │   │
│  │   Script     │  │   Worker     │  │    UI        │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │            │
│         │ Extract data    │ API calls       │ Settings   │
│         │ Inject sidebar  │ Caching         │ Saved list │
│         │                 │                 │            │
└─────────┼─────────────────┼─────────────────┼────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
    ┌───────────┐    ┌───────────────┐  ┌──────────┐
    │  Listing  │    │  Valuation    │  │  Chrome  │
    │   Page    │    │     API       │  │  Storage │
    └───────────┘    └───────────────┘  └──────────┘
```

### Content Script (per-site)

Each supported site needs a content script that:
1. Detects listing pages via URL pattern
2. Extracts property data from DOM/JSON
3. Injects the sidebar panel
4. Handles page navigation (SPAs)

**Site-Specific Extractors**:

```typescript
// extractors/rightmove.ts
export function extractRightmoveData(): PropertyData | null {
  // Rightmove embeds JSON in window.PAGE_MODEL
  const pageModel = (window as any).PAGE_MODEL;
  if (!pageModel?.propertyData) return null;

  return {
    address: pageModel.propertyData.address.displayAddress,
    postcode: pageModel.propertyData.address.outcode,
    bedrooms: pageModel.propertyData.bedrooms,
    bathrooms: pageModel.propertyData.bathrooms,
    size_sqft: parseSize(pageModel.propertyData.sizings),
    price_pcm: parsePrice(pageModel.propertyData.prices),
    price_period: detectPeriod(pageModel.propertyData.prices),
    property_type: pageModel.propertyData.propertySubType,
    amenities: extractAmenities(pageModel.propertyData.keyFeatures),
    latitude: pageModel.propertyData.location?.latitude,
    longitude: pageModel.propertyData.location?.longitude,
  };
}
```

### Background Worker

Handles:
- API calls to valuation service
- Response caching (by address hash)
- Rate limiting
- Error handling

```typescript
// background.ts
const CACHE_TTL = 24 * 60 * 60 * 1000; // 24 hours
const cache = new Map<string, CachedValuation>();

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'VALUATE') {
    const cacheKey = hashProperty(request.data);

    if (cache.has(cacheKey)) {
      const cached = cache.get(cacheKey)!;
      if (Date.now() - cached.timestamp < CACHE_TTL) {
        sendResponse({ success: true, data: cached.valuation });
        return true;
      }
    }

    fetch(API_URL + '/valuate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request.data)
    })
      .then(res => res.json())
      .then(data => {
        cache.set(cacheKey, { valuation: data, timestamp: Date.now() });
        sendResponse({ success: true, data });
      })
      .catch(err => sendResponse({ success: false, error: err.message }));

    return true; // Keep channel open for async response
  }
});
```

### Valuation API Service

**Option A: Serverless (Vercel)**
- Deploy as Vercel API route
- Load pickled model on cold start
- ~2-3s cold start, <100ms warm

**Option B: Dedicated Service**
- FastAPI or Flask
- Model loaded in memory
- Consistent <100ms response

**Recommended: Option A** for MVP (simpler deployment, same infra as dashboard)

```python
# api/valuate.py (Vercel serverless)
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = None
feature_cols = None

def load_model():
    global model, feature_cols
    if model is None:
        with open('rental_model_v15.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('rental_model_v15_features.pkl', 'rb') as f:
            feature_cols = pickle.load(f)

class PropertyInput(BaseModel):
    address: str
    postcode: str
    bedrooms: int
    bathrooms: int
    size_sqft: int | None
    property_type: str | None
    amenities: dict | None
    asking_price_pcm: int

@app.post("/api/v1/valuate")
def valuate(prop: PropertyInput):
    load_model()

    # Engineer features (same as training)
    features = engineer_features_for_prediction(prop)
    X = pd.DataFrame([features])[feature_cols].fillna(0)

    # Predict
    pred_log = model.predict(X)[0]
    fair_value = int(np.expm1(pred_log))

    # Calculate range and assessment
    mape = 0.21
    range_low = int(fair_value * (1 - mape))
    range_high = int(fair_value * (1 + mape))

    premium_pct = (prop.asking_price_pcm / fair_value - 1) * 100
    assessment = classify_assessment(premium_pct)

    return {
        "fair_value_pcm": fair_value,
        "range_low": range_low,
        "range_high": range_high,
        "premium_pct": round(premium_pct, 1),
        "assessment": assessment,
        "model_version": "v15",
        # ... etc
    }
```

---

## Data Flow

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────┐
│ User    │────▶│  Listing    │────▶│  Content    │────▶│ Extract │
│ Browses │     │    Page     │     │   Script    │     │  Data   │
└─────────┘     └─────────────┘     └─────────────┘     └────┬────┘
                                                              │
                                                              ▼
┌─────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────┐
│ Display │◀────│  Sidebar    │◀────│  Background │◀────│  API    │
│ Result  │     │   Panel     │     │   Worker    │     │  Call   │
└─────────┘     └─────────────┘     └─────────────┘     └─────────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │   Cache     │
                                    │  (24 hrs)   │
                                    └─────────────┘
```

---

## Edge Cases & Error Handling

| Scenario | Handling |
|----------|----------|
| Size not available | Estimate from beds/postcode median, show disclaimer |
| Short let detected | Show warning, use adjusted model or decline |
| Price in £/week | Convert to PCM (×52/12) |
| API timeout | Show cached result if available, else "Unavailable" |
| API error | Show error state with retry button |
| Page navigation (SPA) | Listen for URL changes, re-extract |
| Multiple units on page | Show "Multiple units - select one" |
| Commercial property | Detect and show "Not supported" |
| Outside London | Show "Model trained on London only" |

---

## Privacy & Security

| Concern | Mitigation |
|---------|------------|
| User tracking | No user accounts, no tracking, local storage only |
| Data sent to API | Only property details, no user info |
| API security | Rate limiting by IP, no API key for public use |
| Saved properties | Stored in chrome.storage.local, never sent to server |

---

## Metrics & Analytics

Track (anonymously, aggregated):
- Extension installs/uninstalls
- Valuations requested per day
- Sites used (Rightmove vs Zoopla vs others)
- Cache hit rate
- API response times
- Error rates by type

**No tracking of**:
- Individual property views
- User browsing patterns
- Saved properties

---

## Launch Plan

### Phase 1: MVP (2 weeks)
- Rightmove support only
- Basic sidebar with fair value + assessment
- No saved properties feature
- Vercel API endpoint

### Phase 2: Core Features (2 weeks)
- Add Zoopla support
- Saved properties list
- Feature impact breakdown
- Settings panel

### Phase 3: Expansion (2 weeks)
- Add Foxtons, OnTheMarket, OpenRent
- Export to CSV
- Improved size estimation
- Polish and performance

### Phase 4: Growth
- Chrome Web Store listing
- Landing page
- User feedback mechanism

---

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Valuation accuracy | < 25% MAPE on user-reported | Feedback form |
| Page load impact | < 200ms added | Performance profiling |
| API response time | < 500ms p95 | API monitoring |
| User retention | > 30% weekly active | Chrome analytics |
| User satisfaction | > 4.0 stars | Chrome Web Store |

---

## Open Questions

1. **Monetization**: Free forever? Freemium? (e.g., 10 valuations/day free)
2. **Model updates**: How to push new model to API without downtime?
3. **Size estimation**: How accurate is beds→sqft estimation by postcode?
4. **Short lets**: Separate model or just exclude?
5. **Non-London**: Expand model to other UK cities?

---

## Appendix: Site-Specific URL Patterns

```javascript
// manifest.json content_scripts matches
{
  "matches": [
    "https://www.rightmove.co.uk/properties/*",
    "https://www.zoopla.co.uk/to-rent/details/*",
    "https://www.foxtons.co.uk/properties-to-rent/*",
    "https://www.onthemarket.com/details/*",
    "https://www.openrent.com/property-to-rent/*"
  ]
}
```

---

## Appendix: Sidebar UI Mockup

```
┌─────────────────────────────────┐
│ ░░ RENT FAIR VALUE ░░░░░░░░░░░ │
├─────────────────────────────────┤
│                                 │
│  Asking Price                   │
│  £10,914/month                  │
│  £8.32/sqft                     │
│                                 │
│  ─────────────────────────────  │
│                                 │
│  Fair Value Estimate            │
│  £8,728/month                   │
│  £6.65/sqft                     │
│  Range: £6,895 - £10,561        │
│                                 │
│  ─────────────────────────────  │
│                                 │
│  ┌───────────────────────────┐  │
│  │                           │  │
│  │   ████████████░░░░░░░░    │  │
│  │                           │  │
│  │      +25% OVERPRICED      │  │
│  │                           │  │
│  └───────────────────────────┘  │
│                                 │
│  ─────────────────────────────  │
│                                 │
│  Property Details               │
│  ○ 2 bed, 2 bath                │
│  ○ 1,312 sqft                   │
│  ○ SW1W (Belgravia)             │
│  ○ Flat                         │
│                                 │
│  ─────────────────────────────  │
│                                 │
│  Price Factors                  │
│  ▲ Prime Location (SW1)   +15%  │
│  ▲ Air Conditioning       +5%   │
│  ▲ Balcony               +3%   │
│  ▼ No Pool/Gym           -8%   │
│  ▼ No Porter             -5%   │
│                                 │
│  ─────────────────────────────  │
│                                 │
│  [★ Save to Compare]            │
│                                 │
│  ─────────────────────────────  │
│  Model v15 • Updated today      │
│  Based on 7,623 London listings │
└─────────────────────────────────┘
```
