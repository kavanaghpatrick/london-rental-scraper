# PRD: London Rental Fair Value Chrome Extension

## Overview

A Chrome extension that displays ML-powered fair value estimates when browsing rental listings on the 5 agent sites we scrape. The extension extracts all available property data from the page—including parsing the description for amenities—and sends it to our trained model API for valuation.

---

## Problem Statement

Renters browsing property listings have no objective way to assess whether an asking price is fair. Our V15 model (R²=0.73, MAPE=21%) already predicts fair rental values based on 93 features including location, size, amenities, and floor data. This extension brings that intelligence directly into the browsing experience on the exact sites we scrape.

---

## Supported Sites

These are the 5 sources we scrape daily—we already understand their page structures:

| Site | URL Pattern | Data Quality | Spider Reference |
|------|-------------|--------------|------------------|
| **Rightmove** | `rightmove.co.uk/properties/*` | Good (JSON in `__NEXT_DATA__`) | `rightmove_spider.py` |
| **Savills** | `savills.com/property-detail/*` | Excellent (sqft 99%) | `savills_spider.py` |
| **Knight Frank** | `knightfrank.co.uk/properties/*` | Excellent (sqft 93%) | `knightfrank_spider.py` |
| **Foxtons** | `foxtons.co.uk/properties/*` | Excellent (JSON in `__NEXT_DATA__`) | `foxtons_spider.py` |
| **Chestertons** | `chestertons.com/en-gb/property/*` | Good (sqft 71%) | `chestertons_spider.py` |

---

## Core Functionality

### What The Extension Does

1. **Detects** when user is on a property listing page (any of our 5 sources)
2. **Extracts** all available data from the page:
   - Price (pcm or pw → convert to pcm)
   - Address & postcode
   - Bedrooms & bathrooms
   - Size in sqft (if available)
   - Property type
   - Full description text
   - Floorplan images (for OCR if needed)
   - Latitude/longitude (if in page JSON)
   - Agent name
3. **Parses** the description to extract amenities (same logic as model training):
   - Balcony, terrace, garden, roof terrace
   - Porter/concierge, gym, pool
   - Lift, air conditioning
   - High ceilings, period features, modern/contemporary
   - Views, parking, furnished
4. **Sends** extracted data to our valuation API
5. **Displays** fair value estimate with price assessment

---

## Data Extraction Strategy

### Reuse Spider Logic

Each site extractor should mirror our existing spider's extraction logic:

```typescript
// Example: Rightmove uses __NEXT_DATA__ JSON
function extractRightmove(): PropertyData {
  const script = document.getElementById('__NEXT_DATA__');
  const data = JSON.parse(script.textContent);
  const property = data.props.pageProps.propertyData;

  return {
    price_pcm: parsePriceToPCM(property.prices),
    address: property.address.displayAddress,
    postcode: property.address.outcode + ' ' + property.address.incode,
    bedrooms: property.bedrooms,
    bathrooms: property.bathrooms,
    size_sqft: extractSqft(property.sizings), // May be null
    property_type: property.propertySubType,
    description: property.description, // Full text for amenity parsing
    latitude: property.location?.latitude,
    longitude: property.location?.longitude,
    agent_name: property.customer?.branchDisplayName,
    features: property.keyFeatures, // Array of feature strings
    floorplan_url: property.floorplans?.[0]?.url, // For OCR fallback
  };
}
```

### Amenity Extraction from Description

Replicate `parse_amenities()` from `rental_price_models_v15.py`:

```typescript
function parseAmenities(description: string, features: string[]): Amenities {
  const text = (description + ' ' + features.join(' ')).toLowerCase();

  return {
    has_balcony: /balcony/.test(text),
    has_terrace: /terrace/.test(text) && !/roof terrace/.test(text),
    has_roof_terrace: /roof terrace/.test(text),
    has_garden: /garden/.test(text),
    has_porter: /porter|concierge/.test(text),
    has_gym: /\bgym\b/.test(text),
    has_pool: /pool|swimming/.test(text),
    has_parking: /parking|garage/.test(text),
    has_lift: /\blift\b|elevator/.test(text),
    has_ac: /air con|a\/c|aircon|air-con/.test(text),
    has_high_ceilings: /high ceiling/.test(text),
    has_view: /\bview\b/.test(text),
    has_modern: /modern|contemporary/.test(text),
    has_period: /period|victorian|georgian|edwardian/.test(text),
    has_furnished: /furnished/.test(text),
  };
}
```

### Handling Missing Size Data

When sqft is not on the page:

1. **Check floorplan**: If floorplan image URL available, could OCR it (future enhancement)
2. **Estimate from beds**: Use postcode-specific median sqft/bed from our data
3. **Flag as estimated**: Show user "Size estimated: ~X sqft"

```typescript
// Size estimation fallback (from our training data medians)
const MEDIAN_SQFT_PER_BED: Record<string, number> = {
  'SW1': 650,  // Belgravia - larger flats
  'SW3': 580,  // Chelsea
  'SW7': 550,  // South Ken
  'W1': 520,   // Mayfair
  'W8': 540,   // Kensington
  'NW3': 600,  // Hampstead
  'DEFAULT': 500,
};

function estimateSize(bedrooms: number, postcode: string): number {
  const area = postcode.match(/^([A-Z]+\d+)/)?.[1] || 'DEFAULT';
  const sqftPerBed = MEDIAN_SQFT_PER_BED[area] || MEDIAN_SQFT_PER_BED['DEFAULT'];
  return Math.round(bedrooms * sqftPerBed);
}
```

---

## Valuation API

### Endpoint

`POST https://your-domain.vercel.app/api/valuate`

### Request

```json
{
  "address": "4 South Eaton Place, London SW1W 9JA",
  "postcode": "SW1W 9JA",
  "bedrooms": 2,
  "bathrooms": 2,
  "size_sqft": 1312,
  "size_estimated": false,
  "property_type": "flat",
  "description": "Stunning duplex apartment on first and second floors...",
  "amenities": {
    "has_balcony": true,
    "has_terrace": true,
    "has_garden": true,
    "has_porter": true,
    "has_lift": true,
    "has_ac": true,
    "has_high_ceilings": true,
    "has_period": true,
    "has_modern": true,
    "has_view": true,
    "has_pool": false,
    "has_gym": false,
    "has_parking": false,
    "has_roof_terrace": false,
    "has_furnished": true
  },
  "latitude": 51.4934,
  "longitude": -0.1508,
  "agent_name": "Savills",
  "source": "savills",
  "asking_price_pcm": 10914,
  "floor_data": {
    "floor_count": 2,
    "has_basement": false,
    "has_ground": false,
    "has_first_floor": true,
    "has_second_floor": true
  }
}
```

### Response

```json
{
  "fair_value_pcm": 8728,
  "range_low": 6895,
  "range_high": 10561,
  "ppsf_predicted": 6.65,
  "ppsf_asking": 8.32,
  "market_percentile": 85,
  "assessment": "overpriced",
  "assessment_label": "25% above fair value",
  "premium_pct": 25.0,
  "model_version": "v15",
  "model_r2": 0.73,
  "model_mape": 21.0,
  "model_updated": "2026-01-15",
  "size_was_estimated": false,
  "training_samples": 1353,
  "amenities_detected": [
    "Air Conditioning",
    "Lift",
    "Balcony",
    "Terrace",
    "Garden Access",
    "Porter",
    "High Ceilings",
    "Period Features"
  ],
  "amenities_missing": [
    "Pool",
    "Gym"
  ],
  "feature_impacts": [
    {"feature": "Prime Location (SW1W)", "impact_pct": 15, "direction": "up"},
    {"feature": "Air Conditioning", "impact_pct": 5, "direction": "up"},
    {"feature": "Duplex (2 floors)", "impact_pct": 4, "direction": "up"},
    {"feature": "No Pool/Gym", "impact_pct": -8, "direction": "down"}
  ]
}
```

### API Implementation

The API loads the latest trained model and replicates `engineer_features_v15()`:

```python
# api/valuate.py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path('output/rental_model_v15.pkl')
FEATURES_PATH = Path('output/rental_model_v15_features.pkl')

# Load model once at startup
model = pickle.load(open(MODEL_PATH, 'rb'))
feature_cols = pickle.load(open(FEATURES_PATH, 'rb'))

def valuate(request: ValuationRequest) -> ValuationResponse:
    # Build feature row matching training data structure
    row = {
        'bedrooms': request.bedrooms,
        'bathrooms': request.bathrooms,
        'size_sqft': request.size_sqft,
        'postcode_normalized': extract_postcode_district(request.postcode),
        'latitude': request.latitude,
        'longitude': request.longitude,
        'source': request.source,
        'agent_brand': classify_agent(request.agent_name),
        'property_type_std': request.property_type.lower(),
        'let_type': 'long',
        'floor_count': request.floor_data.get('floor_count', 0),
        # ... floor flags ...
    }

    # Add amenities from request
    for amenity, value in request.amenities.items():
        row[amenity] = 1 if value else 0

    # Apply same feature engineering as training
    df = pd.DataFrame([row])
    df = engineer_features_v15(df)  # Same function as training

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].fillna(0)

    # Predict
    pred_log = model.predict(X)[0]
    fair_value = int(np.expm1(pred_log))

    # Calculate assessment
    mape = 0.21
    range_low = int(fair_value * (1 - mape))
    range_high = int(fair_value * (1 + mape))
    premium_pct = (request.asking_price_pcm / fair_value - 1) * 100

    return {
        'fair_value_pcm': fair_value,
        'range_low': range_low,
        'range_high': range_high,
        'premium_pct': round(premium_pct, 1),
        'assessment': classify_assessment(premium_pct),
        # ... etc
    }
```

---

## Extension Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Chrome Extension                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Content Scripts                          │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │ │
│  │  │Rightmove │ │ Savills  │ │  Knight  │ │ Foxtons  │ ...   │ │
│  │  │Extractor │ │Extractor │ │  Frank   │ │Extractor │       │ │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │ │
│  │       └────────────┴────────────┴────────────┘              │ │
│  │                           │                                  │ │
│  │                    extractPropertyData()                     │ │
│  │                           │                                  │ │
│  │                    parseAmenities()                          │ │
│  │                           │                                  │ │
│  │                    injectSidebar()                           │ │
│  └───────────────────────────┼──────────────────────────────────┘ │
│                              │                                    │
│  ┌───────────────────────────▼──────────────────────────────────┐ │
│  │                   Background Worker                           │ │
│  │                                                               │ │
│  │   • Receives extracted data from content script               │ │
│  │   • Calls Valuation API                                       │ │
│  │   • Caches responses (24hr TTL, keyed by address hash)        │ │
│  │   • Sends result back to content script                       │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                      Popup UI                                  │ │
│  │   • Settings (enable/disable sites)                           │ │
│  │   • Saved properties list                                     │ │
│  │   • Model info (version, last updated)                        │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Valuation API       │
                    │   (Vercel Serverless) │
                    │                       │
                    │   • Loads V15 model   │
                    │   • Feature eng.      │
                    │   • Returns estimate  │
                    └───────────────────────┘
```

---

## Site-Specific Extractors

### Rightmove

```typescript
// extractors/rightmove.ts
export function extractRightmove(): PropertyData | null {
  // Rightmove uses __NEXT_DATA__ JSON
  const script = document.getElementById('__NEXT_DATA__');
  if (!script) return null;

  const data = JSON.parse(script.textContent!);
  const p = data.props?.pageProps?.propertyData;
  if (!p) return null;

  const description = p.text?.description || '';
  const features = p.keyFeatures || [];

  return {
    source: 'rightmove',
    address: p.address?.displayAddress,
    postcode: `${p.address?.outcode} ${p.address?.incode}`.trim(),
    price_pcm: parsePriceToPCM(p.prices?.primaryPrice, p.prices?.displayPriceQualifier),
    bedrooms: p.bedrooms,
    bathrooms: p.bathrooms,
    size_sqft: extractSqftFromSizings(p.sizings),
    property_type: p.propertySubType || 'flat',
    description: description,
    features: features,
    amenities: parseAmenities(description, features),
    latitude: p.location?.latitude,
    longitude: p.location?.longitude,
    agent_name: p.customer?.branchDisplayName,
    floorplan_url: p.floorplans?.[0]?.url,
    floor_data: extractFloorData(description, features),
  };
}

function extractSqftFromSizings(sizings: any[]): number | null {
  if (!sizings) return null;
  for (const s of sizings) {
    if (s.unit === 'sqft') return s.minimumSize || s.maximumSize;
    if (s.unit === 'sqm') return Math.round((s.minimumSize || s.maximumSize) * 10.764);
  }
  return null;
}
```

### Savills

```typescript
// extractors/savills.ts
export function extractSavills(): PropertyData | null {
  // Savills is a React SPA, data is in window.__INITIAL_STATE__ or DOM
  const state = (window as any).__INITIAL_STATE__;

  // Alternative: scrape from DOM
  const priceEl = document.querySelector('.property-price');
  const addressEl = document.querySelector('.property-address');
  const descEl = document.querySelector('.property-description');
  const featuresEl = document.querySelectorAll('.property-features li');

  const description = descEl?.textContent || '';
  const features = Array.from(featuresEl).map(el => el.textContent || '');

  // Extract sqft from features or description
  const sqftMatch = (description + features.join(' ')).match(/(\d{1,2},?\d{3})\s*sq\s*ft/i);
  const size_sqft = sqftMatch ? parseInt(sqftMatch[1].replace(',', '')) : null;

  return {
    source: 'savills',
    address: addressEl?.textContent?.trim(),
    postcode: extractPostcode(addressEl?.textContent),
    price_pcm: parsePriceFromText(priceEl?.textContent),
    bedrooms: extractBedrooms(features),
    bathrooms: extractBathrooms(features),
    size_sqft: size_sqft,
    property_type: extractPropertyType(features),
    description: description,
    features: features,
    amenities: parseAmenities(description, features),
    agent_name: 'Savills',
    // ... etc
  };
}
```

### Knight Frank

```typescript
// extractors/knightfrank.ts
export function extractKnightFrank(): PropertyData | null {
  // Knight Frank has data in JSON-LD or specific DOM structure
  const jsonLd = document.querySelector('script[type="application/ld+json"]');
  if (jsonLd) {
    const data = JSON.parse(jsonLd.textContent!);
    // ... extract from structured data
  }

  // Fallback to DOM scraping
  const priceEl = document.querySelector('.kf-search-result__price, .property-price');
  const sizeEl = document.querySelector('.kf-search-result__size, .property-size');
  const descEl = document.querySelector('.property-description');
  // ... etc

  return {
    source: 'knightfrank',
    // ... extracted data
  };
}
```

### Foxtons

```typescript
// extractors/foxtons.ts
export function extractFoxtons(): PropertyData | null {
  // Foxtons uses __NEXT_DATA__ like Rightmove
  const script = document.getElementById('__NEXT_DATA__');
  if (!script) return null;

  const data = JSON.parse(script.textContent!);
  const p = data.props?.pageProps?.property;
  if (!p) return null;

  return {
    source: 'foxtons',
    address: p.address,
    postcode: p.postcode,
    price_pcm: p.price_per_month || convertWeeklyToPCM(p.price_per_week),
    bedrooms: p.bedrooms,
    bathrooms: p.bathrooms,
    size_sqft: p.floor_area_sq_ft,
    property_type: p.property_type,
    description: p.description,
    features: p.features || [],
    amenities: parseAmenities(p.description, p.features || []),
    latitude: p.latitude,
    longitude: p.longitude,
    agent_name: 'Foxtons',
    floor_data: {
      floor_count: p.floors || 1,
      // ... etc
    },
  };
}
```

### Chestertons

```typescript
// extractors/chestertons.ts
export function extractChestertons(): PropertyData | null {
  // Chestertons uses Pegasus property cards, data in DOM
  const card = document.querySelector('.pegasus-property-detail');
  if (!card) return null;

  const priceEl = card.querySelector('.property-price');
  const addressEl = card.querySelector('.property-address');
  const descEl = card.querySelector('.property-description');
  const featuresEl = card.querySelectorAll('.property-features li');

  // ... extract and return
}
```

---

## Amenity Parsing (Shared)

```typescript
// utils/amenities.ts
export interface Amenities {
  has_balcony: boolean;
  has_terrace: boolean;
  has_roof_terrace: boolean;
  has_garden: boolean;
  has_porter: boolean;
  has_gym: boolean;
  has_pool: boolean;
  has_parking: boolean;
  has_lift: boolean;
  has_ac: boolean;
  has_high_ceilings: boolean;
  has_view: boolean;
  has_modern: boolean;
  has_period: boolean;
  has_furnished: boolean;
}

export function parseAmenities(description: string, features: string[]): Amenities {
  const text = [description, ...features].join(' ').toLowerCase();

  return {
    has_balcony: /\bbalcon(y|ies)\b/.test(text),
    has_terrace: /\bterrace\b/.test(text) && !/roof terrace/.test(text),
    has_roof_terrace: /roof terrace/.test(text),
    has_garden: /\bgarden\b/.test(text),
    has_porter: /\bporter\b|\bconcierge\b/.test(text),
    has_gym: /\bgym\b|\bfitness\b/.test(text),
    has_pool: /\bpool\b|\bswimming\b/.test(text),
    has_parking: /\bparking\b|\bgarage\b/.test(text),
    has_lift: /\blift\b|\belevator\b/.test(text),
    has_ac: /\bair con|\ba\/c\b|\baircon|\bair-con|\bair conditioning\b/.test(text),
    has_high_ceilings: /high ceiling/.test(text),
    has_view: /\bview(s)?\b/.test(text) && !/viewings/.test(text),
    has_modern: /\bmodern\b|\bcontemporary\b/.test(text),
    has_period: /\bperiod\b|\bvictorian\b|\bgeorgian\b|\bedwardian\b/.test(text),
    has_furnished: /\bfurnished\b/.test(text) && !/unfurnished/.test(text),
  };
}

export function extractFloorData(description: string, features: string[]): FloorData {
  const text = [description, ...features].join(' ').toLowerCase();

  // Detect floor mentions
  const hasBasement = /\bbasement\b|\blower ground\b/.test(text);
  const hasGround = /\bground floor\b/.test(text);
  const hasFirst = /\bfirst floor\b/.test(text);
  const hasSecond = /\bsecond floor\b/.test(text);
  const hasThird = /\bthird floor\b/.test(text);
  const hasFourthPlus = /\b(fourth|fifth|sixth|seventh|top) floor\b/.test(text);

  // Count floors for multi-floor properties (duplex, triplex)
  const isDuplex = /\bduplex\b/.test(text);
  const isTriplex = /\btriplex\b/.test(text);
  const floorCount = isTriplex ? 3 : isDuplex ? 2 : 1;

  return {
    floor_count: floorCount,
    has_basement: hasBasement,
    has_ground: hasGround,
    has_first_floor: hasFirst,
    has_second_floor: hasSecond,
    has_third_floor: hasThird,
    has_fourth_plus: hasFourthPlus,
  };
}
```

---

## UI: Sidebar Panel

```
┌─────────────────────────────────┐
│ ◉ RENT FAIR VALUE              │
├─────────────────────────────────┤
│                                 │
│  ASKING PRICE                   │
│  ┌───────────────────────────┐  │
│  │  £10,914/month            │  │
│  │  £8.32/sqft               │  │
│  └───────────────────────────┘  │
│                                 │
│  MODEL ESTIMATE                 │
│  ┌───────────────────────────┐  │
│  │  £8,728/month             │  │
│  │  £6.65/sqft               │  │
│  │  Range: £6,895 - £10,561  │  │
│  └───────────────────────────┘  │
│                                 │
│  ASSESSMENT                     │
│  ┌───────────────────────────┐  │
│  │ ████████████████░░░░░░░░  │  │
│  │                           │  │
│  │    +25% OVERPRICED        │  │
│  │                           │  │
│  │  Asking is £2,186 above   │  │
│  │  fair value estimate      │  │
│  └───────────────────────────┘  │
│                                 │
│  DETECTED FEATURES              │
│  ┌───────────────────────────┐  │
│  │ ✓ Air Conditioning        │  │
│  │ ✓ Lift                    │  │
│  │ ✓ Balcony                 │  │
│  │ ✓ Terrace                 │  │
│  │ ✓ Porter/Concierge        │  │
│  │ ✓ Period Features         │  │
│  │ ✓ High Ceilings           │  │
│  │ ✗ Pool                    │  │
│  │ ✗ Gym                     │  │
│  └───────────────────────────┘  │
│                                 │
│  PROPERTY DETAILS               │
│  ┌───────────────────────────┐  │
│  │ 2 bed · 2 bath · 1,312 sf │  │
│  │ SW1W · Flat · Savills     │  │
│  └───────────────────────────┘  │
│                                 │
│  [★ Save] [📋 Copy] [⚙ Settings]│
│                                 │
├─────────────────────────────────┤
│  Model V15 · R²=0.73           │
│  Updated: 15 Jan 2026          │
│  Trained on 1,353 listings     │
└─────────────────────────────────┘
```

**Color Coding**:
| Assessment | Color | Condition |
|------------|-------|-----------|
| Great Deal | Green (#22c55e) | Asking < Fair × 0.90 |
| Fair Price | Blue (#3b82f6) | 0.90 ≤ Asking/Fair ≤ 1.10 |
| Slightly High | Amber (#f59e0b) | 1.10 < Asking/Fair ≤ 1.25 |
| Overpriced | Red (#ef4444) | Asking/Fair > 1.25 |

---

## Technical Implementation

### Project Structure

```
chrome-extension/
├── manifest.json
├── src/
│   ├── content/
│   │   ├── index.ts              # Main content script
│   │   ├── extractors/
│   │   │   ├── rightmove.ts
│   │   │   ├── savills.ts
│   │   │   ├── knightfrank.ts
│   │   │   ├── foxtons.ts
│   │   │   └── chestertons.ts
│   │   ├── utils/
│   │   │   ├── amenities.ts      # parseAmenities, extractFloorData
│   │   │   ├── price.ts          # parsePriceToPCM
│   │   │   └── postcode.ts       # extractPostcode, etc.
│   │   └── sidebar/
│   │       ├── Sidebar.tsx       # React component
│   │       └── styles.css
│   ├── background/
│   │   └── worker.ts             # API calls, caching
│   ├── popup/
│   │   ├── Popup.tsx
│   │   └── styles.css
│   └── types/
│       └── index.ts
├── api/                          # Vercel serverless
│   └── valuate.ts
└── package.json
```

### Manifest V3

```json
{
  "manifest_version": 3,
  "name": "Rent Fair Value",
  "version": "1.0.0",
  "description": "See ML-powered fair value estimates on rental listings",
  "permissions": ["storage", "activeTab"],
  "host_permissions": [
    "https://www.rightmove.co.uk/*",
    "https://www.savills.com/*",
    "https://www.knightfrank.co.uk/*",
    "https://www.foxtons.co.uk/*",
    "https://www.chestertons.com/*"
  ],
  "content_scripts": [
    {
      "matches": [
        "https://www.rightmove.co.uk/properties/*",
        "https://www.savills.com/property-detail/*",
        "https://www.knightfrank.co.uk/properties/*",
        "https://www.foxtons.co.uk/properties/*",
        "https://www.chestertons.com/*/property/*"
      ],
      "js": ["content.js"],
      "css": ["sidebar.css"]
    }
  ],
  "background": {
    "service_worker": "background.js"
  },
  "action": {
    "default_popup": "popup.html",
    "default_icon": "icon.png"
  }
}
```

---

## API Deployment

### Vercel Serverless Function

```typescript
// api/valuate.ts
import { NextRequest, NextResponse } from 'next/server';
import pickle from 'picklejs'; // Or load via Python subprocess
import { engineerFeaturesV15 } from '../lib/features';

let model: any = null;
let featureCols: string[] = [];

async function loadModel() {
  if (!model) {
    // Load pickled model (or use ONNX for better JS compatibility)
    const modelBuffer = await fetch(process.env.MODEL_URL!).then(r => r.arrayBuffer());
    model = pickle.loads(modelBuffer);
    featureCols = await fetch(process.env.FEATURES_URL!).then(r => r.json());
  }
}

export async function POST(request: NextRequest) {
  const data = await request.json();

  await loadModel();

  // Engineer features (same as training)
  const features = engineerFeaturesV15(data);

  // Predict
  const X = featureCols.map(col => features[col] ?? 0);
  const predLog = model.predict([X])[0];
  const fairValue = Math.round(Math.expm1(predLog));

  // Assessment
  const mape = 0.21;
  const rangeLow = Math.round(fairValue * (1 - mape));
  const rangeHigh = Math.round(fairValue * (1 + mape));
  const premiumPct = ((data.asking_price_pcm / fairValue) - 1) * 100;

  return NextResponse.json({
    fair_value_pcm: fairValue,
    range_low: rangeLow,
    range_high: rangeHigh,
    premium_pct: Math.round(premiumPct * 10) / 10,
    assessment: classifyAssessment(premiumPct),
    model_version: 'v15',
    // ... etc
  });
}
```

### Alternative: Python API with FastAPI

```python
# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load model at startup
with open('output/rental_model_v15.pkl', 'rb') as f:
    model = pickle.load(f)
with open('output/rental_model_v15_features.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

class ValuationRequest(BaseModel):
    address: str
    postcode: str
    bedrooms: int
    bathrooms: int
    size_sqft: int | None
    amenities: dict
    asking_price_pcm: int
    # ... etc

@app.post("/valuate")
def valuate(req: ValuationRequest):
    # Build features DataFrame
    df = build_features_df(req)  # Same as training
    X = df[feature_cols].fillna(0)

    # Predict
    pred_log = model.predict(X)[0]
    fair_value = int(np.expm1(pred_log))

    return {
        "fair_value_pcm": fair_value,
        # ... etc
    }
```

---

## Development Phases

### Phase 1: Core MVP (1 week)
- [ ] Rightmove extractor (highest traffic)
- [ ] Amenity parsing from description
- [ ] Basic sidebar UI
- [ ] Valuation API (Python/FastAPI)
- [ ] Local testing

### Phase 2: All Sources (1 week)
- [ ] Savills extractor
- [ ] Knight Frank extractor
- [ ] Foxtons extractor
- [ ] Chestertons extractor
- [ ] Handle site-specific quirks

### Phase 3: Polish (1 week)
- [ ] Caching in background worker
- [ ] Settings panel
- [ ] Save/compare feature
- [ ] Error handling & edge cases
- [ ] Size estimation fallback

### Phase 4: Launch
- [ ] Chrome Web Store submission
- [ ] Landing page
- [ ] Documentation

---

## Testing Strategy

### Unit Tests
- Amenity parsing: 50+ test cases for each amenity keyword
- Price parsing: pcm, pw, annual conversions
- Postcode extraction: various formats

### Integration Tests
- Each extractor against saved HTML fixtures
- API endpoint with various inputs
- Full flow: extract → API → display

### Manual Testing
- Test on 10 listings from each source
- Verify extracted data matches visible info
- Check amenity detection accuracy

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Extraction accuracy | >95% | Manual audit of 50 listings |
| Amenity detection | >90% | Compare to manual labels |
| API response time | <500ms | Monitoring |
| User retention | >30% WAU | Chrome analytics |

---

## Open Questions

1. **Model format**: Keep pickle or convert to ONNX for JS-native inference?
2. **Floorplan OCR**: Worth implementing for missing sqft? (Complex, may not be MVP)
3. **Rate limiting**: How many requests/day free? Monetization model?
4. **Model updates**: Auto-deploy new model daily, or manual releases?
