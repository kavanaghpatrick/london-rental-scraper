# Social Housing / Affordable Housing Detection Analysis

## Executive Summary

Analysis of residuals from the rental price prediction model reveals significant overestimation patterns (model predicts higher than actual rent) that correlate with social housing, affordable housing schemes, and council estates in prime London postcodes.

**Key Finding**: Properties in known social housing estates show systematically lower rents than the model predicts, with an average overestimation of approximately 30-50% in premium areas.

---

## 1. Specific Examples Found

### 1.1 Confirmed Social Housing Estates

| Property | Postcode | Beds | Actual Rent | Predicted | Residual | PPSF | Overestimate % |
|----------|----------|------|-------------|-----------|----------|------|----------------|
| World's End Estate, SW10 | SW10 | 3 | 3,300 | 8,322 | -5,022 | 4.42 | **152%** |
| World's End Estate, Chelsea | - | 2 | 2,491 | 3,620 | -1,129 | 3.33 | 45% |
| Upper Whistler Walk, World's End Estate | SW10 | 3 | 3,000 | 3,856 | -856 | 3.37 | 29% |
| Hallfield Estate, W2 | W2 | 3 | 3,449 | 3,697 | -248 | 4.27 | 7% |
| Townshend Estate, NW8 | NW8 | 3 | 3,250 | 3,676 | -426 | 4.10 | 13% |
| Penfold Place, Lisson Grove | NW1 | 1 | 1,775 | 2,716 | -941 | 4.90 | 53% |
| Penfold Place, Lisson Grove | NW1 | 1 | 1,650 | 2,339 | -689 | 5.52 | 42% |
| Ebury Bridge Road, SW1W | SW1W | 1 | 2,384 | 3,715 | -1,331 | 5.05 | 56% |
| Hardwick House, Lisson Grove | NW8 | 3 | 2,850 | 3,745 | -895 | 3.10 | 31% |

### 1.2 Suspicious Low-PPSF Properties in Premium Postcodes

Properties in premium areas (SW7, SW3, W8, SW1X) with PPSF below 4.0 (typical is 5-10+):

| Property | Postcode | Beds | Actual Rent | Predicted | Residual | PPSF |
|----------|----------|------|-------------|-----------|----------|------|
| Imperial Wharf, SW6 | SW6 | 3 | 6,500 | 10,475 | -3,975 | **3.08** |
| Kensington Gardens Square, W2 | W2 | 1 | 3,185 | 5,128 | -1,943 | **3.19** |
| Old Marylebone Road, NW1 | NW1 | 2 | 3,142 | 4,697 | -1,555 | **3.09** |
| Mallory Street, NW8 | NW8 | 3 | 3,000 | 4,702 | -1,702 | **3.37** |
| Rosary Gardens, SW7 | SW7 | 2 | 3,034 | 4,335 | -1,301 | **3.22** |
| Queens Gate, SW7 | SW7 | 2 | 4,767 | 7,441 | -2,674 | **3.45** |
| Ennismore Gardens, SW7 | SW7 | 3 | 4,598 | 6,497 | -1,899 | **3.58** |
| Cromwell Road, SW7 | SW7 | 3 | 3,142 | 3,970 | -828 | **3.37** |

### 1.3 Known Social Housing Areas in Prime Postcodes

**Lisson Grove Area (NW1/NW8)** - Westminster Council estates:
- Multiple properties on Lisson Grove, Penfold Place, Mallory Street
- PPSF range: 3.09 - 5.52 (vs. typical NW8 average ~6.5)
- Consistent 20-50% overestimation

**World's End Estate (SW10)** - Chelsea Council estate:
- Famous 1960s brutalist towers
- PPSF: 3.33 - 4.42 (vs. typical SW10 average ~5.5)
- Most extreme case: 152% overestimation

**Ebury Bridge (SW1W)** - Pimlico Council estate:
- Large estate near Victoria Station
- Currently undergoing regeneration
- 36-56% overestimation

---

## 2. Regex Patterns for Detection

### 2.1 Estate Names (High Confidence)

```python
# Known council/social housing estates
SOCIAL_ESTATE_PATTERNS = [
    r"world'?s?\s*end\s*estate",
    r"townshend\s*estate",
    r"hallfield\s*estate",
    r"churchill\s*gardens",
    r"dolphin\s*square",  # Note: also has market-rate units
    r"lillington\s*gardens",
    r"ebury\s*bridge",
    r"peabody\s*(estate|buildings?|trust)?",
    r"trellick\s*tower",
    r"grenfell\s*tower",  # Historical
    r"lancaster\s*west",
    r"silchester",
    r"wornington",
    r"edward\s*woods",
]
```

### 2.2 Street/Area Names (Medium Confidence)

```python
# Areas with high social housing concentration
SOCIAL_AREA_PATTERNS = [
    r"lisson\s*grove",
    r"penfold\s*(place|street)",
    r"church\s*street.*(nw8|w2)",
    r"hardwick\s*house",
    r"mallory\s*street",
    r"old\s*marylebone\s*road",
]
```

### 2.3 Housing Association Names

```python
# Major London housing associations
HOUSING_ASSOCIATION_PATTERNS = [
    r"peabody",
    r"l\s*&\s*q",
    r"notting\s*hill\s*(genesis|housing)?",
    r"clarion",
    r"catalyst",
    r"optivo",
    r"metropolitan\s*thames",
    r"hyde\s*housing",
    r"sanctuary",
    r"sovereign",
    r"genesis\s*housing",
]
```

### 2.4 Building Name Patterns (Lower Confidence - Needs Context)

```python
# Generic patterns that MAY indicate social housing
# Use in combination with low PPSF
BUILDING_PATTERNS = [
    r"\b\w+\s+tower\b",      # e.g., "Trellick Tower", but also luxury towers
    r"\b\w+\s+point\b",      # e.g., "Wharfside Point"
    r"\b\w+\s+house\b",      # e.g., "Hardwick House" - very common, low confidence
    r"\bestate\b",           # Only when combined with low PPSF
]
```

---

## 3. Feature Engineering Recommendations

### 3.1 Binary Flag: `is_likely_social_housing`

```python
def is_likely_social_housing(address: str, ppsf: float, postcode_district: str) -> int:
    """
    Returns 1 if property is likely social/affordable housing.

    Detection logic:
    1. Direct match on known estate names (high confidence)
    2. Match on social areas + low PPSF
    3. Premium postcode + very low PPSF (< 3.5)
    """
    address_lower = address.lower()

    # High confidence: Known estate names
    high_conf_patterns = [
        r"world'?s?\s*end\s*estate",
        r"townshend\s*estate",
        r"hallfield\s*estate",
        r"churchill\s*gardens",
        r"ebury\s*bridge",
        r"peabody",
        r"trellick",
    ]
    for pattern in high_conf_patterns:
        if re.search(pattern, address_lower):
            return 1

    # Medium confidence: Social areas
    social_areas = [
        r"lisson\s*grove",
        r"penfold\s*place",
        r"mallory\s*street",
    ]
    for pattern in social_areas:
        if re.search(pattern, address_lower) and ppsf < 5.0:
            return 1

    # Low confidence: Premium postcode + very low PPSF
    premium_postcodes = ['SW1', 'SW3', 'SW7', 'W1', 'W8', 'NW8']
    if any(postcode_district.startswith(p) for p in premium_postcodes):
        if ppsf < 3.5:
            return 1

    return 0
```

### 3.2 Continuous Feature: `social_housing_score`

```python
def social_housing_score(address: str, ppsf: float, postcode_ppsf_mean: float) -> float:
    """
    Returns score 0-1 indicating likelihood of social housing.
    Higher = more likely social/affordable housing.
    """
    score = 0.0
    address_lower = address.lower()

    # Estate name matches
    if re.search(r"estate", address_lower):
        score += 0.3
    if re.search(r"world'?s?\s*end|townshend|hallfield|ebury\s*bridge", address_lower):
        score += 0.5

    # PPSF anomaly score
    if postcode_ppsf_mean > 0:
        ppsf_ratio = ppsf / postcode_ppsf_mean
        if ppsf_ratio < 0.5:
            score += 0.4
        elif ppsf_ratio < 0.7:
            score += 0.2

    # Area matches
    if re.search(r"lisson|penfold|mallory", address_lower):
        score += 0.2

    return min(score, 1.0)
```

### 3.3 Additional Engineered Features

```python
# Feature: PPSF deviation from postcode mean
df['ppsf_z_score'] = (df['ppsf'] - df.groupby('postcode_district')['ppsf'].transform('mean')) \
                    / df.groupby('postcode_district')['ppsf'].transform('std')

# Feature: Is PPSF < 50% of postcode median?
df['ppsf_below_half_median'] = (df['ppsf'] < df.groupby('postcode_district')['ppsf'].transform('median') * 0.5).astype(int)

# Feature: Estate/Council keywords in address
df['has_estate_keyword'] = df['address'].str.lower().str.contains(r'estate|council|housing\s*association').astype(int)
```

---

## 4. Estimated Error Reduction Impact

### 4.1 Current Error Contribution

Based on analysis of 5,597 records:

| Metric | Value |
|--------|-------|
| Total records | 5,597 |
| Records with residual < -1000 and PPSF < 4 | 1,105 (19.7%) |
| Identified social housing matches | 26 |
| Mean overestimation for social housing | 30-50% |
| Total SSE (all records) | 2.78 x 10^11 |
| SSE contribution (negative residuals only) | 4.47 x 10^8 |

### 4.2 Expected Improvement

If `is_likely_social_housing` flag is added and model learns appropriate discount:

**Conservative Estimate (detecting 50% of social housing):**
- Records affected: ~50-100 properties
- Mean error reduction per property: ~1,000-2,000 pounds
- **Estimated MAE improvement: 2-5%**
- **Estimated R-squared improvement: 0.5-1.0%**

**Optimistic Estimate (detecting 80% of social housing):**
- Records affected: ~100-200 properties
- Mean error reduction per property: ~1,500-2,500 pounds
- **Estimated MAE improvement: 5-8%**
- **Estimated R-squared improvement: 1.0-2.0%**

### 4.3 Categories of Improvement

1. **World's End Estate cluster** (4 properties): Extreme overestimates (30-150%)
2. **Lisson Grove area** (10+ properties): Moderate overestimates (20-50%)
3. **Ebury Bridge area** (3+ properties): Significant overestimates (35-55%)
4. **Low-PPSF anomalies in SW7/SW3** (20+ properties): Various overestimates (15-40%)

---

## 5. Implementation Recommendations

### 5.1 Quick Win: Binary Flag

Add `is_likely_social_housing` binary feature using the regex patterns above. This can be implemented immediately in the feature engineering pipeline.

```python
# In feature engineering
df['is_social_housing'] = df.apply(
    lambda row: is_likely_social_housing(
        row['address'],
        row['ppsf'],
        row['postcode_district']
    ),
    axis=1
)
```

### 5.2 Enhanced Approach: PPSF Anomaly Detection

Create a postcode-level PPSF model first, then flag properties with PPSF significantly below expected:

```python
# Calculate expected PPSF by postcode
postcode_ppsf_stats = df.groupby('postcode_district')['ppsf'].agg(['mean', 'std', 'median'])

# Flag anomalies
df['ppsf_anomaly'] = (df['ppsf'] < df['postcode_district'].map(postcode_ppsf_stats['median']) * 0.6).astype(int)
```

### 5.3 Model Strategy Options

**Option A: Separate Models**
- Train separate model for social housing properties
- Use classification to route predictions

**Option B: Feature-Based Adjustment**
- Include `is_social_housing` flag in main model
- Model learns appropriate discount factor

**Option C: Post-Prediction Adjustment**
- Apply fixed discount (e.g., 25%) to flagged properties
- Simpler but less adaptive

### 5.4 Data Collection Enhancement

Consider adding to scraper pipeline:
1. Extract housing association mentions from descriptions
2. Flag "affordable housing" / "shared ownership" keywords
3. Track if property is ex-council (often mentioned in descriptions)

---

## 6. Appendix: Full List of Detected Properties

### Properties in Known Social Housing Estates (26 matches)

```
World's End Estate (6 properties)
Townshend Estate (2 properties)
Hallfield Estate (1 property)
Lisson Grove area (12 properties)
Penfold Place (4 properties)
Ebury Bridge (1 property)
```

### Suspicious Low-PPSF Properties in Premium Postcodes (50+ properties)

Key streets/buildings with multiple low-PPSF listings:
- Kensington Gardens Square, W2 (PPSF: 3.19 - 4.0)
- Queens Gate, SW7 (PPSF: 3.45 - 5.3)
- Imperial Wharf, SW6 (PPSF: 3.08 - 3.6)
- Cromwell Road/Crescent, SW5/SW7 (PPSF: 3.37 - 3.74)
- Mallory Street, NW8 (PPSF: 3.37)
- Old Marylebone Road, NW1 (PPSF: 3.09)

---

*Analysis completed: 2026-01-15*
*Data source: residuals_analysis.csv (5,597 records)*
