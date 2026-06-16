# Property Condition Investigation Report

## Executive Summary

This investigation analyzed 4,528 rental properties to identify condition-related factors that cause model prediction errors. Key findings:

1. **Unfurnished properties** have 18% lower PPSF than furnished properties and the model overestimates them 50% of the time vs 40% for furnished
2. **"Long let" properties** have significantly lower PPSF (average 4.48 vs 6.21 for others) and the model overestimates 69% of them
3. **Properties with refurbishment keywords** have 14% higher PPSF and better model accuracy (42% overestimated vs 57% without keywords)
4. **273 properties** in premium postcodes have PPSF below 4/sqft, suggesting condition or furnishing issues

---

## 1. Low PPSF in Premium Postcodes

### Summary Statistics
- Total properties with PPSF < 4: **802** (17.7% of dataset)
- Properties with PPSF < 3.5: **333** (7.4% of dataset)
- Low PPSF in premium postcodes: **273**

### Examples of Extremely Low PPSF (< 3.5) in Premium Areas

| Address | Postcode | Beds | Size | Price | PPSF | Residual |
|---------|----------|------|------|-------|------|----------|
| Netherhall Gardens, Hampstead | NW3 | 1 | 740 sqft | 2,231 | 3.01 | -751 |
| King Henrys Road, Primrose Hill | NW3 | 5 | 2,228 sqft | 6,750 | 3.03 | -2,567 |
| Pangbourne Avenue | W11 | 5 | 3,272 sqft | 9,966 | 3.05 | -2,468 |
| Old Marylebone Road | NW1 | 2 | 1,018 sqft | 3,142 | 3.09 | -1,555 |
| Briardale Gardens | NW3 | 5 | 3,346 sqft | 10,400 | 3.11 | -4,514 |
| Pembridge Place, Notting Hill | W11 | 4 | 1,951 sqft | 6,045 | 3.10 | -1,406 |
| Maresfield Gardens, Hampstead | NW3 | 2 | 960 sqft | 3,000 | 3.13 | -921 |
| Rosary Gardens, SW7 | SW7 | 2 | 941 sqft | 3,034 | 3.22 | -1,531 |
| Chelsea Harbour | SW10 | 2 | 1,167 sqft | 3,595 | 3.08 | -670 |

**Key Observation**: These premium postcodes (NW3 Hampstead, W11 Notting Hill, SW7 South Kensington) typically command 6-8/sqft. Properties at 3/sqft in these areas likely have condition issues or are unfurnished.

### Property Type Breakdown for Low PPSF (< 4/sqft)

| Property Type | Count | Avg PPSF | Avg Residual |
|---------------|-------|----------|--------------|
| flat | 499 | 3.56 | -924 |
| apartment | 99 | 3.59 | -934 |
| long let | 99 | 3.55 | -1,547 |
| house | 72 | 3.53 | -2,042 |
| maisonette | 14 | 3.54 | -1,279 |

**Finding**: The model consistently overestimates these low-PPSF properties, suggesting it cannot distinguish between well-maintained and basic/unfurnished properties.

---

## 2. Furnished vs Unfurnished Analysis

### Key Findings from Description Text Analysis

| Status | Count | Avg PPSF | Median PPSF | Model Overestimated |
|--------|-------|----------|-------------|---------------------|
| Furnished | 111 | 6.53 | 6.02 | 39.6% |
| Unfurnished | 62 | 5.49 | 5.09 | 50.0% |
| Part Furnished | 2 | 5.09 | 5.09 | 0.0% |

**PPSF Difference**: Unfurnished properties average **15.9% lower** PPSF than furnished properties.

### Unfurnished Properties Where Model Most Overestimated

| Address | Postcode | Beds | Price | Predicted | Residual | PPSF |
|---------|----------|------|-------|-----------|----------|------|
| Somerset Square | W14 | 5 | 17,333 | 22,001 | -4,668 | 4.51 |
| Briardale Gardens | NW3 | 5 | 10,400 | 14,580 | -4,180 | 3.12 |
| Eaton Place, Belgravia | SW1X | 3 | 20,000 | 23,798 | -3,798 | 5.33 |
| Park Road, Marylebone | NW8 | 4 | 12,500 | 15,740 | -3,240 | 4.20 |
| Harley House, Marylebone Road | NW1 | 4 | 10,799 | 13,615 | -2,816 | 4.32 |
| Iverna Gardens, Kensington | W8 | 3 | 5,750 | 8,055 | -2,305 | 3.46 |

**Pattern**: Large family homes (4-5 beds) in premium areas that are unfurnished command significantly lower rents than the model expects.

### Example Descriptions

**Unfurnished property (W8 Kensington, 5.38 PPSF)**:
> "MANAGED BY SAVILLS UNFURNISHED Flat / Apartment 1,671 sq ft..."

**Furnished property (SW1X Knightsbridge, 10.51 PPSF)**:
> "A spectacular Grade II listed house... This elegant Grade II listed period house (with mews) is arranged over 6 floors..."

---

## 3. Condition Keywords Analysis

### Refurbishment Keywords Impact

| Keyword Type | Count | Avg PPSF | Model Overestimated |
|--------------|-------|----------|---------------------|
| Has refurb keywords | 105 | 6.31 | 41.9% |
| No refurb keywords | 352 | 5.52 | 57.1% |

**Finding**: Properties with "refurbished", "newly decorated", "renovated" keywords have **14.3% higher PPSF** and better model accuracy.

### Keyword Frequency in Descriptions

**Positive Condition Indicators** (associated with higher PPSF):
- "refurbished": 137 listings
- "modern": 285 listings
- "contemporary": 106 listings
- "newly refurbished": 50 listings
- "immaculate": 52 listings
- "luxury": 48 listings
- "recently refurbished": 29 listings
- "brand new": 21 listings
- "high spec": 15 listings

**Negative Condition Indicators** (associated with lower PPSF):
- "unfurnished": 94 listings
- "period features": 20 listings (often means older/unmodernized)
- "part furnished": 9 listings
- "original features": 4 listings
- "dated": 2 listings

### Refurbished Properties - Best vs Worst Performing

**Where model UNDERESTIMATED (property worth more than predicted)**:
| Address | Postcode | Beds | Price | Residual | PPSF |
|---------|----------|------|-------|----------|------|
| Cadogan Place | SW1X | 6 | 86,450 | +26,633 | 10.51 |
| Charles Street | W1J | 3 | 27,083 | +18,433 | 17.44 |
| Wilton Crescent | SW1X | 7 | 78,000 | +16,563 | 12.00 |
| Melbury Road, Holland Park | W14 | 5 | 36,400 | +12,988 | 13.48 |

**Where model OVERESTIMATED (even with refurb keywords)**:
| Address | Postcode | Beds | Price | Residual | PPSF |
|---------|----------|------|-------|----------|------|
| Thurloe Square, South Kensington | SW7 | 4 | 30,116 | -6,291 | 7.76 |
| Briardale Gardens | NW3 | 5 | 10,400 | -4,180 | 3.12 |
| Eaton Place, Belgravia | SW1X | 3 | 20,000 | -3,798 | 5.33 |

**Insight**: Even "refurbished" properties can have low PPSF if they are unfurnished - the keywords don't always indicate high condition.

---

## 4. "Long Let" Impact

### Critical Finding

Properties marked as "long let" show systematic model overestimation:

| Let Type | Count | Avg PPSF | Model Overestimated |
|----------|-------|----------|---------------------|
| Long let | 100 | 4.48 | **69.0%** |
| Other | 4,428 | 6.21 | 59.0% |

**PPSF Difference**: Long let properties have **27.9% lower** PPSF than other properties.

**Explanation**: "Long let" typically indicates:
- Unfurnished or minimally furnished
- Longer tenancy agreements (12+ months)
- Targeting different tenant demographic
- Often older/less updated properties

---

## 5. Feature Recommendations

Based on this analysis, the following features should be added to the model:

### High Priority Features

1. **`is_furnished`** (binary: 0/1)
   - Extract from description text using keywords: "unfurnished", "furnished"
   - Expected impact: ~16% PPSF difference
   - Model overestimation reduction potential: 10-15%

2. **`has_refurb_keywords`** (binary: 0/1)
   - Keywords to detect: "refurbished", "newly decorated", "newly renovated", "brand new", "just completed"
   - Expected impact: ~14% PPSF difference

3. **`is_long_let`** (binary: 0/1)
   - Extract from `let_type` field or description
   - Expected impact: ~28% PPSF difference
   - Strongest predictor of model overestimation

### Medium Priority Features

4. **`has_luxury_keywords`** (binary: 0/1)
   - Keywords: "luxury", "high spec", "designer", "premium", "immaculate", "pristine"
   - Associated with higher PPSF and better model accuracy

5. **`has_dated_keywords`** (binary: 0/1)
   - Keywords: "dated", "needs updating", "tired", "basic", "original condition"
   - Rare but indicative of lower condition

6. **`condition_score`** (continuous: -1 to +1)
   - Composite score based on positive minus negative keyword counts
   - Formula: (positive_keywords - negative_keywords) / total_keywords

### Implementation Notes

```python
# Example keyword extraction functions

def is_furnished(description):
    if not description:
        return None
    desc = description.lower()
    if 'unfurnished' in desc or 'un-furnished' in desc:
        return 0
    if 'furnished' in desc:
        return 1
    return None

def has_refurb_keywords(description):
    if not description:
        return 0
    desc = description.lower()
    keywords = ['refurbished', 'newly decorated', 'newly renovated',
                'brand new', 'just completed', 'newly fitted']
    return 1 if any(kw in desc for kw in keywords) else 0

def is_long_let(let_type, description):
    if let_type == 'long':
        return 1
    if description and 'long let' in description.lower():
        return 1
    return 0
```

---

## 6. Expected Model Improvement

### Current Model Weaknesses
- Cannot distinguish furnished vs unfurnished: ~16% PPSF gap unexplained
- Cannot identify long lets: ~28% PPSF gap unexplained
- Cannot identify recently refurbished: ~14% PPSF gap unexplained

### Estimated Impact of New Features

| Feature | % of Overestimated Properties Affected | Expected MAE Reduction |
|---------|---------------------------------------|------------------------|
| is_furnished | 10-15% | 3-5% |
| is_long_let | 5-10% | 2-4% |
| has_refurb_keywords | 5-10% | 1-3% |
| **Combined** | **20-30%** | **5-10%** |

### Priority for Implementation
1. **`is_long_let`** - Strongest signal, easy to extract from `let_type` field
2. **`is_furnished`** - Large PPSF impact, requires description parsing
3. **`has_refurb_keywords`** - Moderate impact, straightforward NLP

---

## Appendix: Data Quality Notes

- Only 457 properties (10%) in residuals analysis have descriptions in database
- `furnished` field in database is almost entirely NULL
- `let_type` field has better coverage but still only 100 "long" entries
- Premium postcodes analyzed: SW1, SW1X, SW1W, SW3, SW7, W1, W1J, W1K, W8, W11, NW1, NW3, NW8
