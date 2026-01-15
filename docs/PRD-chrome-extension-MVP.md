# PRD: Chrome Extension MVP (Simplified)

## One-Sentence Summary

Chrome extension that shows ML fair value estimate on Rightmove listings.

---

## Core Flow (5 Lines)

```typescript
if (url.includes('rightmove.co.uk/properties/')) {
  const data = extractFromNextData();  // {price, beds, baths, postcode, description}
  const result = await fetch(API_URL, { method: 'POST', body: JSON.stringify(data) });
  injectSidebar(result.fair_value, data.price);
}
```

---

## MVP Scope (1 Week)

### Include
- **Rightmove only** (validate before adding other sites)
- **5 fields**: price, beds, baths, postcode, description
- **Simple sidebar**: asking vs fair value + % difference
- **API does all heavy lifting**: amenity parsing, size estimation, feature engineering

### Exclude (Push to API or v2)
- ❌ Other 4 sites (Savills, Knight Frank, Foxtons, Chestertons)
- ❌ Browser-side amenity parsing
- ❌ Browser-side size estimation
- ❌ Price conversion (API handles pcm/pw)
- ❌ Floor data extraction
- ❌ Lat/long, agent name, property type
- ❌ Save/compare feature
- ❌ Settings panel
- ❌ Caching (premature optimization)

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

### content.js
```typescript
(async function() {
  // Extract from __NEXT_DATA__
  const script = document.getElementById('__NEXT_DATA__');
  if (!script) return;

  const json = JSON.parse(script.textContent);
  const p = json.props?.pageProps?.propertyData;
  if (!p) return;

  const data = {
    price_pcm: p.prices?.primaryPrice || 0,
    price_period: p.prices?.displayPriceQualifier || 'pcm',
    bedrooms: p.bedrooms || 0,
    bathrooms: p.bathrooms || 0,
    postcode: `${p.address?.outcode || ''} ${p.address?.incode || ''}`.trim(),
    description: p.text?.description || '',
    size_sqft: extractSqft(p.sizings),  // null if missing
  };

  // Call API (API handles all parsing/estimation)
  const response = await fetch('https://your-api.vercel.app/api/valuate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });

  if (!response.ok) return;
  const result = await response.json();

  // Inject sidebar
  injectSidebar(result, data);
})();

function extractSqft(sizings) {
  if (!sizings) return null;
  for (const s of sizings) {
    if (s.unit === 'sqft') return s.minimumSize || s.maximumSize;
    if (s.unit === 'sqm') return Math.round((s.minimumSize || s.maximumSize) * 10.764);
  }
  return null;
}

function injectSidebar(result, data) {
  const diff = Math.round((data.price_pcm / result.fair_value - 1) * 100);
  const color = diff > 10 ? '#ef4444' : diff < -10 ? '#22c55e' : '#3b82f6';
  const label = diff > 10 ? 'OVERPRICED' : diff < -10 ? 'GOOD DEAL' : 'FAIR';

  const sidebar = document.createElement('div');
  sidebar.id = 'rent-fair-value';
  sidebar.innerHTML = `
    <div style="position:fixed;top:100px;right:20px;width:280px;background:#1f2937;color:white;padding:20px;border-radius:12px;z-index:9999;font-family:system-ui;">
      <div style="font-size:12px;opacity:0.7;margin-bottom:4px;">ASKING</div>
      <div style="font-size:24px;font-weight:bold;">£${data.price_pcm.toLocaleString()}/mo</div>
      <hr style="border-color:#374151;margin:16px 0;">
      <div style="font-size:12px;opacity:0.7;margin-bottom:4px;">FAIR VALUE</div>
      <div style="font-size:24px;font-weight:bold;">£${result.fair_value.toLocaleString()}/mo</div>
      <div style="font-size:12px;opacity:0.7;">Range: £${result.range_low.toLocaleString()} - £${result.range_high.toLocaleString()}</div>
      <hr style="border-color:#374151;margin:16px 0;">
      <div style="text-align:center;padding:12px;background:${color};border-radius:8px;">
        <div style="font-size:20px;font-weight:bold;">${diff > 0 ? '+' : ''}${diff}%</div>
        <div style="font-size:12px;">${label}</div>
      </div>
      <div style="font-size:10px;opacity:0.5;margin-top:12px;text-align:center;">
        Model V15 · R²=0.73
      </div>
    </div>
  `;
  document.body.appendChild(sidebar);
}
```

---

## API Endpoint

**URL**: `POST /api/valuate`

**Request** (from extension):
```json
{
  "price_pcm": 10914,
  "price_period": "pcm",
  "bedrooms": 2,
  "bathrooms": 2,
  "postcode": "SW1W 9JA",
  "description": "Stunning duplex apartment...",
  "size_sqft": 1312
}
```

**API Responsibilities** (not extension):
- Convert pw → pcm if needed
- Parse amenities from description
- Estimate size if null
- Engineer all 93 features
- Run model prediction

**Response**:
```json
{
  "fair_value": 8728,
  "range_low": 6895,
  "range_high": 10561,
  "model_version": "v15"
}
```

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
