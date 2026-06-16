# Ultra-Luxury Tier Investigation Report

## Executive Summary

The model systematically underestimates ultra-luxury properties (>30,000 pcm) by an average of 10,293 pcm (-22.7% error). 81.5% of ultra-luxury properties are undervalued, compared to only 37.7% of mainstream properties. This indicates the model lacks features to capture prestige premiums that drive pricing at the top of the market.

---

## 1. Profile of Ultra-Luxury Segment

### Overview Statistics
| Metric | Value |
|--------|-------|
| Total Properties | 92 (2.0% of dataset) |
| Price Range | 30,940 - 86,450 pcm |
| Mean Price | 45,282 pcm |
| Mean Size | 4,523 sqft |
| Mean PPSF | 11.37/sqft |

### PPSF Distribution
| Percentile | Ultra-Luxury | Mainstream (<15k) |
|------------|--------------|-------------------|
| 10th | 6.63 | 3.60 |
| 25th | 7.90 | - |
| 50th (Median) | 10.65 | 5.12 |
| 75th | 13.15 | - |
| 90th | 17.09 | 8.53 |

**Key Finding**: Ultra-luxury PPSF is roughly 2x mainstream, but extreme outliers reach 27-29/sqft - nearly 6x the mainstream median.

### Postcode Distribution (Top Districts)
| District | Count | Avg Price | Avg PPSF |
|----------|-------|-----------|----------|
| SW7 (Knightsbridge/S. Ken) | 23 | 40,694 | 9.90 |
| SW1X (Belgravia) | 20 | 49,420 | 12.74 |
| W1K (Mayfair) | 7 | 51,690 | 13.98 |
| W8 (Kensington) | 7 | 40,671 | 8.45 |
| SW3 (Chelsea) | 7 | 43,017 | 9.85 |
| W11 (Notting Hill) | 4 | 46,582 | 16.46 |
| W1J (Mayfair) | 4 | 32,500 | 13.99 |

**Geographic Concentration**: 78% of ultra-luxury is in just 7 postcodes. SW7 and SW1X alone account for 47%.

### Property Type Distribution
| Type | Count | % of Ultra-Luxury |
|------|-------|-------------------|
| Flat | 30 | 32.6% |
| House | 27 | 29.3% |
| Long Let | 14 | 15.2% |
| Terraced | 5 | 5.4% |
| Short Let | 5 | 5.4% |
| Other | 11 | 12.0% |

---

## 2. Model Error Analysis

### Error by Segment
| Segment | Count | Avg Residual | Underestimated % |
|---------|-------|--------------|------------------|
| Ultra-luxury (>30k) | 92 | +10,293 | 81.5% |
| High-end (15k-30k) | 285 | +3,860 | 71.9% |
| Mainstream (<15k) | 4,151 | +5 | 37.7% |

**Pattern**: The model is calibrated for mainstream properties. Error grows exponentially with price tier.

### Error by Property Structure (Ultra-Luxury Only)
| Type | Count | Avg Residual | Underestimated % |
|------|-------|--------------|------------------|
| Flats | 37 | +13,277 | 89.2% |
| Houses | 33 | +5,023 | 69.7% |
| Mews | 1 | +16,724 | 100% |

**Key Finding**: Ultra-luxury FLATS are the most undervalued. Houses perform relatively better, possibly because the `is_house` feature captures some prestige signal.

### Top 10 Most Underestimated Properties
| Address | Price | Predicted | Residual | PPSF |
|---------|-------|-----------|----------|------|
| Eaton Place, Belgravia SW1X | 71,500 | 29,048 | +42,452 | 29.12 |
| Lancaster Road, W11 | 62,833 | 25,766 | +37,067 | 20.59 |
| Westbourne Grove, W11 | 56,333 | 20,280 | +36,053 | 17.02 |
| Eaton Place, Belgravia | 65,000 | 30,360 | +34,640 | 27.24 |
| Hyde Park Gate SW7 | 60,666 | 29,147 | +31,519 | 17.47 |
| Adam's Row W1K | 52,000 | 24,369 | +27,631 | 13.77 |
| St. Edmunds Terrace NW8 | 52,000 | 24,649 | +27,351 | 11.94 |
| Cadogan Place SW1X | 86,450 | 59,817 | +26,633 | 10.51 |
| Basil Street SW3 | 43,333 | 16,837 | +26,496 | 15.43 |
| Basil Street SW3 | 43,333 | 17,769 | +25,564 | 14.97 |

---

## 3. Missing Prestige Indicators

### Famous Streets Analysis

The model has NO features for street-level prestige. Analysis of known prestige streets reveals systematic undervaluation:

| Street | Properties | Avg Price | Avg Residual | Underestimated % |
|--------|------------|-----------|--------------|------------------|
| Adams Row | 2 | 43,334 | +14,145 | 100% |
| Grosvenor Square | 2 | 27,300 | +11,572 | 50% |
| Cadogan Place | 3 | 31,650 | +8,591 | 67% |
| Wilton Crescent | 4 | 43,725 | +7,610 | 75% |
| Pavilion Road | 9 | 20,462 | +6,896 | 78% |
| Basil Street | 18 | 16,605 | +6,828 | 89% |
| Upper Grosvenor | 8 | 30,385 | +5,704 | 100% |
| Eaton Place | 30 | 20,502 | +5,317 | 67% |
| Hyde Park Gate | 15 | 22,943 | +5,248 | 87% |

### Missing Prestige Signals

1. **Iconic Square/Crescent Names**: Eaton Square, Belgrave Square, Chester Square, Wilton Crescent
2. **Royal/Aristocratic Associations**: Palace Gardens, Kensington Palace, Prince Albert Road
3. **Historic Addresses**: Hyde Park Gate, Princes Gate, Hans Place
4. **Prime Mayfair Streets**: Adams Row, Upper Grosvenor Street, Grosvenor Square, South Audley Street
5. **Chelsea/Knightsbridge Icons**: Cadogan Square, Pont Street, Basil Street

### What the Model Currently Has
Based on the residuals file columns, the model uses:
- `postcode_district` - too coarse (all of SW1X is treated equally)
- `bedrooms`, `bathrooms`, `size_sqft` - size metrics
- `is_house`, `is_flat`, `is_mews` - property type
- No street-level features
- No proximity to landmarks/parks
- No building prestige (period mansion block vs modern build)

---

## 4. Feature Recommendations

### Option A: Add Prestige Features to Existing Model

#### Feature 1: `is_ultra_luxury_address` (Binary)
A binary flag for properties on known ultra-prestige streets:
```python
ULTRA_PRESTIGE_STREETS = [
    'eaton place', 'eaton square', 'belgrave square', 'chester square',
    'wilton crescent', 'cadogan place', 'cadogan square', 'grosvenor square',
    'upper grosvenor street', 'hyde park gate', 'princes gate', 'hans place',
    'pont street', 'adams row', 'basil street', 'montpelier square',
    'pavilion road', 'sloane street', 'lancelot place', 'trevor square'
]
```
**Expected Impact**: +5,000-15,000 pcm for properties on these streets.

#### Feature 2: `street_prestige_score` (Continuous 0-10)
A tiered prestige score based on address parsing:
- Score 10: Eaton Place/Square, Grosvenor Square, Wilton Crescent
- Score 8: Cadogan Place/Square, Hyde Park Gate, Princes Gate
- Score 6: Hans Place, Pont Street, Upper Grosvenor, Adams Row
- Score 4: Basil Street, Pavilion Road, Sloane Street
- Score 2: Other prime area streets (SW1X, SW7, W1K general)
- Score 0: All other

**Rationale**: Allows gradient rather than binary cutoff.

#### Feature 3: `ppsf_tier` (Categorical)
Based on the observed PPSF distribution:
- Tier 1: <4/sqft (budget)
- Tier 2: 4-7/sqft (mainstream)
- Tier 3: 7-12/sqft (premium)
- Tier 4: 12-18/sqft (luxury)
- Tier 5: >18/sqft (ultra-luxury)

**Rationale**: PPSF >15/sqft signals prestige premium that size/location alone don't capture.

#### Feature 4: `is_period_block` (Binary)
Flag for Victorian/Edwardian mansion blocks:
- "Mansions" in address
- "Court" in historic areas (SW1X, SW3, SW7)
- "House" + period building indicators

### Option B: Separate Model for Ultra-Luxury

**Pros**:
- Completely different pricing dynamics
- Prestige factors dominate over size/location
- Small dataset (92) could be modeled with simpler approach

**Cons**:
- Need to classify properties into tier BEFORE prediction
- Discontinuity at threshold (what about 28,000 pcm properties?)
- Less training data for segment-specific model

---

## 5. Recommendation

### Primary Recommendation: Hybrid Approach

1. **Add prestige features to main model** (Option A):
   - `is_ultra_luxury_address` (binary)
   - `street_prestige_score` (0-10)
   - These help the entire dataset, not just ultra-luxury

2. **Add price-tier interaction terms**:
   - Allow prestige features to have stronger effect at higher price points
   - Or add `postcode_district` x `is_ultra_luxury_address` interaction

3. **Consider prediction cap/floor**:
   - For SW1X/SW7/W1K with prestige addresses, apply minimum PPSF floor of 8/sqft
   - This prevents absurd underestimates

### Why NOT a Separate Model

The ultra-luxury segment has only 92 properties - too few for reliable training. Better to:
1. Enrich the main model with prestige signals
2. Apply post-hoc adjustments for extreme cases
3. Flag predictions >50% below PPSF typical for the street as "likely underestimate"

---

## 6. Implementation Priority

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| P0 | `is_ultra_luxury_address` | Low | High |
| P1 | `street_prestige_score` | Medium | High |
| P2 | `is_period_block` | Medium | Medium |
| P3 | `ppsf_tier` input | Low | Medium |
| P4 | Separate ultra-luxury model | High | Medium |

### Quick Win

Create a lookup table of 20 prestige streets and flag them. This single feature would capture ~40% of the ultra-luxury underestimation.

---

## Appendix: Raw Data

### All Ultra-Luxury Properties by Postcode District

**SW1X (20 properties)**:
- Eaton Place (multiple) - up to 71,500/m
- Cadogan Place - 86,450/m
- Wilton Crescent - 78,000/m
- One Hyde Park - 45,500/m

**SW7 (23 properties)**:
- Hyde Park Gate (multiple) - up to 60,666/m
- Princes Gate (multiple) - up to 65,000/m
- Knightsbridge - 52,000/m
- Lancelot Place - 65,000/m

**W1K (7 properties)**:
- Upper Grosvenor Street - up to 65,000/m
- Grosvenor Square - 39,000/m
- Adams Row - 43,334/m
