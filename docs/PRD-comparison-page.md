# PRD: Property Comparison Page Feature

## Overview
Add a "Compare" button to the Chrome extension that opens a dedicated comparison page showing detailed stats on the current property relative to similar properties ("peers").

## Problem Statement
Users viewing a Rightmove listing currently only see:
- Fair value estimate
- Price differential (over/under priced %)
- Basic property details

They lack context on:
- How this property compares to similar ones
- What similar properties are available
- Market positioning relative to peers

## Solution
A clickable "Compare" button that opens a new page with comprehensive comparison analytics.

## Architecture Decision: Real-time DB Query

**Rejected:** Pre-computed `similar_listings.json` file
- ~10k properties × 15 peers = wasteful computation
- Most properties never viewed
- Stale data immediately after generation
- Large file to download/cache

**Chosen:** Real-time Vercel Serverless API
- Query only when needed (~50ms response)
- Always fresh data from Postgres
- Smaller, targeted responses
- Can add dynamic filters (exclude seen, custom ranges)

## User Flow
1. User visits Rightmove listing
2. Chrome extension shows fair value overlay (existing)
3. New "Compare" button appears in overlay
4. Click opens `compare.html` page with:
   - Current property summary
   - Peer properties grid
   - Comparison charts/stats

## Feature Requirements

### 1. Peer Selection Algorithm
Peers are properties most similar based on:

| Factor | Weight | Matching Logic |
|--------|--------|----------------|
| Location | 30% | Same postcode district (SW3, W8, etc.) |
| Size | 25% | Within ±20% sqft |
| Bedrooms | 20% | Same bedroom count |
| Property Type | 15% | Same type (flat, house, etc.) |
| Price Range | 10% | Within ±30% of asking price |

**Target:** Find 5-15 peer properties from database.

### 2. Comparison Page Layout

```
┌─────────────────────────────────────────────────────────┐
│  [Property Address]                    📊 Comparison    │
│  3 bed flat • SW3 • 1,200 sqft                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  YOUR PROPERTY                    PEER AVERAGE          │
│  ┌─────────────┐                 ┌─────────────┐       │
│  │ £3,500 pcm  │                 │ £3,200 pcm  │       │
│  │ £2.92/sqft  │                 │ £2.75/sqft  │       │
│  │ Fair: £3,100│                 │             │       │
│  └─────────────┘                 └─────────────┘       │
│                                                         │
│  PRICE POSITIONING                                      │
│  ═══════════════════════════════════════════════════   │
│  Cheapest          │ YOU │              Most Expensive  │
│  £2,800      ◆     │  ●  │     ◆    ◆        £4,200   │
│                                                         │
│  KEY METRICS vs PEERS                                   │
│  ┌────────────────────────────────────────────────┐    │
│  │ Price/sqft:    £2.92  [▓▓▓▓▓▓▓▓░░] +6% above   │    │
│  │ Size:          1,200  [▓▓▓▓▓▓░░░░] -12% below  │    │
│  │ Tube distance: 0.3km  [▓▓▓▓▓▓▓▓▓░] Top 10%     │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  SIMILAR PROPERTIES (12 found)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ [Photo]  │ │ [Photo]  │ │ [Photo]  │ │ [Photo]  │  │
│  │ £3,100   │ │ £3,400   │ │ £2,950   │ │ £3,600   │  │
│  │ 1,150sqft│ │ 1,280sqft│ │ 1,100sqft│ │ 1,350sqft│  │
│  │ +12%     │ │ -3%      │ │ +5%      │ │ -15%     │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                         │
│  MARKET INSIGHTS                                        │
│  • This property is priced 9% above peer average       │
│  • 4 of 12 peers are better value (lower £/sqft)       │
│  • Closest tube station advantage vs 8/12 peers        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3. Data Sources

| Data | Source | Notes |
|------|--------|-------|
| Current property | Page scraping (existing) | Address, price, beds, sqft |
| Peer properties | Real-time API query | `/api/similar` endpoint |
| Fair values | `predictions.json` or local XGBoost | Model estimates |
| Property images | Rightmove URLs | Thumbnail display |

### 4. Similar Listings API

#### Endpoint: `GET /api/similar`

**Request Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| postcode | string | Yes | Postcode district (SW3, W8, NW1) |
| beds | int | Yes | Number of bedrooms |
| sqft | int | No | Size in sqft (improves matching) |
| price | int | Yes | Asking price PCM |
| type | string | No | Property type (flat, house) |
| exclude | string | No | Property ID to exclude |

**Example Request:**
```
GET /api/similar?postcode=SW3&beds=2&sqft=850&price=3500&type=flat
```

**Response:**
```json
{
  "peers": [
    {
      "id": "rightmove:12345679",
      "url": "https://rightmove.co.uk/properties/12345679",
      "address": "Flat 2, 100 Kings Road, SW3",
      "price_pcm": 3100,
      "size_sqft": 820,
      "bedrooms": 2,
      "property_type": "flat",
      "fair_value": 3050,
      "ppsf": 3.78,
      "similarity_score": 0.92,
      "source": "savills"
    }
  ],
  "stats": {
    "peer_count": 12,
    "avg_price": 3200,
    "avg_ppsf": 3.65,
    "min_price": 2800,
    "max_price": 4200,
    "your_percentile": 65
  },
  "query_ms": 48
}
```

#### SQL Query Logic:
```sql
SELECT
    id, source, property_id, address, postcode, url,
    price_pcm, size_sqft, bedrooms, property_type,
    -- Similarity scoring (0-1)
    (
        CASE WHEN bedrooms = $beds THEN 0.30 ELSE 0.10 END +
        CASE WHEN size_sqft BETWEEN $sqft * 0.8 AND $sqft * 1.2 THEN 0.25
             WHEN size_sqft BETWEEN $sqft * 0.6 AND $sqft * 1.4 THEN 0.10
             ELSE 0 END +
        CASE WHEN ABS(price_pcm - $price) <= $price * 0.20 THEN 0.20
             WHEN ABS(price_pcm - $price) <= $price * 0.35 THEN 0.10
             ELSE 0 END +
        CASE WHEN property_type = $type THEN 0.15 ELSE 0 END +
        CASE WHEN source IN ('savills','knightfrank') THEN 0.10 ELSE 0.05 END
    ) AS similarity_score
FROM listings
WHERE is_active = 1
  AND SPLIT_PART(postcode, ' ', 1) = $postcode_district
  AND bedrooms BETWEEN $beds - 1 AND $beds + 1
  AND price_pcm BETWEEN $price * 0.5 AND $price * 2.0
ORDER BY similarity_score DESC, ABS(price_pcm - $price) ASC
LIMIT 15;
```

### 5. Technical Implementation

#### Files to Create/Modify:

**New API (Vercel Serverless):**
1. `api/similar.py` - Similarity query endpoint

**Chrome Extension:**
1. `chrome-extension/compare.html` - Comparison page
2. `chrome-extension/compare.js` - Page logic + API calls
3. `chrome-extension/compare.css` - Styling
4. `chrome-extension/content.js` - Add "Compare" button

#### Extension Manifest Changes:
```json
{
  "web_accessible_resources": [
    {
      "resources": ["compare.html"],
      "matches": ["*://*.rightmove.co.uk/*"]
    }
  ],
  "permissions": ["https://your-vercel-app.vercel.app/*"]
}
```

#### API Deployment:
Deploy to existing Vercel project alongside dashboard:
```
vercel-dashboard/
├── api/
│   └── similar.py    # New endpoint
├── app/
│   └── ...           # Existing dashboard
```

## Success Metrics
- Users click "Compare" on >20% of listings viewed
- Average time on comparison page >30 seconds
- User feedback: "helps decision making"

## Out of Scope (v1)
- Historical price tracking per property
- Saved comparisons
- Multiple property comparison
- Mobile support

## Dependencies
- Vercel Postgres (existing)
- Vercel Serverless Functions
- XGBoost model for fair values (existing)

## Timeline
- Phase 1: API endpoint + basic comparison page (MVP)
- Phase 2: Charts and visualizations
- Phase 3: Market insights text generation

## Open Questions
1. Should peers include inactive (recently let) properties for context?
2. Cache API responses? (probably not needed at ~50ms)
3. How to handle properties with no/few peers? (show message, expand search)
4. Rate limiting for API? (Vercel has built-in limits)
