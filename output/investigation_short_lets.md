# Short Let / Serviced Apartment Investigation

**Date:** 2026-01-15
**Dataset:** 4,528 listings from residuals_analysis.csv
**Focus:** Understanding model underestimates (positive residuals) and identifying short lets/serviced apartments

---

## Executive Summary

Short lets and serviced apartments significantly distort the rental price prediction model. Excluding 142 properties (3.1% of data) based on three detection rules would reduce:
- **Squared error by 31.8%**
- **Mean Absolute Error by 11.0%** (from 1,497 to 1,332 GBP)

**Recommendation:** Exclude these properties from the training set. Do NOT model them separately - they are fundamentally different products (short-term furnished rentals vs long-term residential).

---

## 1. Suspected Short Lets Currently in Data

### 1.1 Explicit Short Lets (property_type = "short let")
**Count: 69 properties**

| Address | PPSF | Residual | Postcode |
|---------|------|----------|----------|
| Adam's Row, London, W1K | 13.8 | +27,631 | W1K |
| Upper Grosvenor Street, London, W1K | 15.1 | +24,498 | W1K |
| Pavilion Road, London, SW1X | 17.1 | +23,012 | SW1X |
| Melbury Road, London, W14 | 12.2 | +19,185 | W14 |
| Eaton Place, London, SW1X | 18.8 | +16,408 | SW1X |
| Abingdon Villas, Kensington, W8 | 12.7 | +15,191 | W8 |
| Melbury Road, Holland Park, W14 | 13.5 | +12,988 | W14 |
| Belgrave Mews South, SW1X | 12.5 | +7,035 | SW1X |
| Kensington High Street, W8 | 10.7 | +6,943 | W8 |
| Onslow Gardens, SW7 | 10.9 | +6,861 | SW7 |

**Statistics:**
- Mean residual: +3,370 GBP (model underestimates by 3,370 on average)
- 55 of 69 have positive residuals (80%)
- Total residual contribution: +232,497 GBP

### 1.2 Known Serviced Apartment Buildings (29 properties)

| Building/Brand | Count | Mean Residual | Notes |
|----------------|-------|---------------|-------|
| Ashburn Place (SW7) | 6 | +10,838 | Serviced apartments, 21-27 PPSF |
| Cheval (various) | 8 | +5,733 | Cheval Residences brand |
| 199 Knightsbridge | 8 | +4,356 | The Knightsbridge Apartments |
| The Other House (SW7) | 3 | +3,075 | Boutique aparthotel |
| Thorney Court | 1 | +3,845 | Serviced building |

**Top serviced apartment residuals:**
```
Ashburn Place, SW7              | PPSF: 27.0 | Residual: +25,355
Cheval Thorney Court, W8        | PPSF: 17.6 | Residual: +17,067
The Knightsbridge Apartments    | PPSF: 13.6 | Residual: +13,144
199 Knightsbridge, SW7          | PPSF: 13.1 | Residual: +12,821
Cheval Place, SW7               | PPSF: 12.3 | Residual: +11,102
```

### 1.3 Extreme PPSF Properties (>20 GBP/sqft) - 49 properties

These are almost certainly serviced/furnished short lets based on pricing alone:

| Address | PPSF | Residual | Notes |
|---------|------|----------|-------|
| Prince Albert Road, NW8 | 29.9 | +8,700 | Regent's Park |
| The Knightsbridge, SW7 | 29.7 | +9,286 | Luxury building |
| Woodford House, Chelsea Creek, SW6 | 29.4 | +17,261 | New build |
| Imperial House, Kensington, W8 | 29.2 | +19,184 | Prime location |
| Eaton Place, Belgravia, SW1X | 29.1 | +42,452 | **Largest residual** |
| Rutland Gardens Mews, SW7 | 28.9 | +13,035 | Knightsbridge |
| Talbot Road, Notting Hill, W2 | 28.6 | +23,285 | Notting Hill |
| Egerton Gardens, SW3 | 28.5 | +2,143 | Chelsea |

**Note:** Normal London luxury rentals rarely exceed 12-15 PPSF. Properties above 20 PPSF are priced at 2x market rate, consistent with short-term/serviced pricing.

---

## 2. Detection Patterns

### 2.1 Recommended Exclusion Rules (in order of priority)

```python
# Rule 1: Explicit short let type
if 'short let' in property_type.lower():
    exclude = True

# Rule 2: Known serviced apartment buildings
SERVICED_KEYWORDS = [
    'cheval', 'thorney court', 'the other house',
    'ashburn place', '199 knightsbridge', 'knightsbridge apartments'
]
if any(kw in address.lower() for kw in SERVICED_KEYWORDS):
    exclude = True

# Rule 3: Extreme PPSF (definitely not normal rental)
if ppsf > 20:
    exclude = True
```

### 2.2 Additional Detection Signals (for future refinement)

These patterns strongly correlate with short lets but may catch some legitimate high-end rentals:

| Signal | Threshold | Confidence | False Positive Risk |
|--------|-----------|------------|---------------------|
| PPSF > 15 with residual > 5,000 | High | Medium | ~10% |
| PPSF > 12 with residual > 15,000 | High | Medium | ~5% |
| Address contains "Apartments," | Medium | High | Some purpose-built blocks |
| Mayfair W1K/W1J with high PPSF | High | Medium | Ultra-luxury genuine lets |

### 2.3 Address Keyword Patterns

**High-confidence serviced apartment indicators:**
- "Cheval" (Cheval Residences brand)
- "Thorney Court" (serviced building)
- "The Other House" (aparthotel brand)
- "199 Knightsbridge" / "Knightsbridge Apartments"
- "Ashburn Place" (known serviced location)
- "Fraser" (Fraser Suites, not found in current data)
- "Marlin" (Marlin Apartments, not found in current data)
- "SACO" (Stay Serviced Apartments, not found in current data)

**Medium-confidence indicators (use with PPSF check):**
- "Apartments" at end of address
- "Suites" in address
- "Residences" in address

---

## 3. Recommendation: Exclude, Don't Model Separately

### Why Exclusion is Better Than Separate Modeling

1. **Fundamentally Different Product**
   - Short lets = furnished, bills included, flexible terms (1-12 months)
   - Long lets = unfurnished/part-furnished, bills separate, 12+ month ASTs
   - Comparing them is like comparing hotel rooms to rental apartments

2. **Pricing Drivers Are Different**
   - Short lets: Location convenience, furnishing quality, service level
   - Long lets: Location, size, condition, tenure security

3. **Sample Size Issue**
   - Only 142 confirmed short lets (3.1% of data)
   - Too few to train a robust separate model
   - High variance in short let pricing makes modeling unreliable

4. **User Goal Alignment**
   - Users seeking rent negotiation tools want long-term rental valuations
   - Short let pricing is irrelevant to their use case

### Recommended Implementation

```python
# In data preprocessing pipeline
def is_short_let(row):
    """Returns True if property should be excluded from training."""

    # 1. Explicit short let type
    if 'short let' in row.get('property_type', '').lower():
        return True

    # 2. Known serviced buildings
    serviced_keywords = [
        'cheval', 'thorney court', 'the other house',
        'ashburn place', '199 knightsbridge', 'knightsbridge apartments'
    ]
    address = row.get('address', '').lower()
    if any(kw in address for kw in serviced_keywords):
        return True

    # 3. Extreme PPSF (price-per-sqft)
    try:
        ppsf = float(row.get('ppsf', 0))
        if ppsf > 20:
            return True
    except (ValueError, TypeError):
        pass

    return False
```

---

## 4. Estimated Error Reduction Impact

### Before Exclusion
- **Records:** 4,528
- **Total Squared Error:** 47,268,125,485
- **MAE:** 1,497.43 GBP

### After Exclusion
- **Records:** 4,386 (-142, -3.1%)
- **Total Squared Error:** 32,221,415,851
- **MAE:** 1,332.44 GBP

### Improvement
| Metric | Reduction | Notes |
|--------|-----------|-------|
| Squared Error | **31.8%** | Major improvement in outlier handling |
| MAE | **11.0%** | Significant improvement in typical prediction |
| Records Lost | 3.1% | Minimal data loss |

### Error Contribution by Category

| Category | Records | % of Positive Residual Sum |
|----------|---------|---------------------------|
| All suspicious properties | 214 | 35.4% |
| High confidence (2+ flags) | 60 | 22.1% |
| Recommended exclusions | 142 | ~25% |

---

## 5. Complete List of Properties to Exclude

### By Exclusion Reason

**Short Let Type (69 properties):**
```
Adam's Row, W1K                    | PPSF: 13.8 | Residual: +27,631
Upper Grosvenor Street, W1K        | PPSF: 15.1 | Residual: +24,498
Pavilion Road, SW1X                | PPSF: 17.1 | Residual: +23,012
Melbury Road, W14                  | PPSF: 12.2 | Residual: +19,185
Eaton Place, SW1X                  | PPSF: 18.8 | Residual: +16,408
Abingdon Villas, W8                | PPSF: 12.7 | Residual: +15,191
... (69 total)
```

**Serviced Buildings (29 properties):**
```
Ashburn Place, SW7 (6x)
Cheval Thorney Court, W8 (2x)
Cheval Place (3x)
Cheval House (1x)
199 Knightsbridge / Knightsbridge Apartments (8x)
The Other House (3x)
Thorney Court (1x)
```

**Extreme PPSF >20 (44 properties):**
```
Eaton Place, Belgravia, SW1X       | PPSF: 29.1 | Residual: +42,452
Lancaster Road, W11                | PPSF: 20.6 | Residual: +37,067
Eaton Place, Belgravia             | PPSF: 27.2 | Residual: +34,640
Grosvenor Square, W1K              | PPSF: 26.9 | Residual: +24,671
Talbot Road, Notting Hill, W2      | PPSF: 28.6 | Residual: +23,285
... (44 total)
```

---

## 6. Next Steps

1. **Immediate:** Add `is_short_let()` filter to model training pipeline
2. **Scraping:** Consider flagging short lets at scrape time based on listing text ("short let available", "minimum 3 months")
3. **Monitoring:** Track new serviced apartment buildings entering the dataset
4. **Validation:** After exclusion, verify MAPE and Median APE improvements

---

## Appendix: Data Sources

- Short lets primarily from: Chestertons (explicit "short let" property_type), Knight Frank
- Serviced apartments from: Foxtons, Rightmove, Knight Frank
- Extreme PPSF properties: All sources, concentrated in SW1X, SW7, W1K, W8
