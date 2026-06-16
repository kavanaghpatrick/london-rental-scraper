# Feature Engineering Recommendations - Model v1.0

**Generated**: 2026-01-15 from 10 parallel investigation agents
**Current Model**: v0.9.9 with 125 features
**Target**: Reduce MAPE from 19.7% to <15%

---

## Executive Summary

Analysis of 4,528 properties identified 6 major model weaknesses:

| Issue | Properties Affected | Current Error | Root Cause |
|-------|---------------------|---------------|------------|
| Short lets contaminating data | 142 (3.1%) | 31.8% SSE contribution | Not filtering serviced/short-term |
| Ultra-luxury underestimation | 92 (2.0%) | -22.7% avg error | Missing prestige signals |
| Social housing overestimation | 50-100 | +30-50% avg error | No institutional flag |
| Postcode micro-location | 500+ | 25-47% systematic bias | Too coarse location encoding |
| Size non-linearity | All | U-shaped PPSF curve | Missing size^2 and log terms |
| Floor/condition signals | 800+ | 16-28% PPSF gaps | No furnished/floor features |

**Estimated total MAE improvement: 20-35% relative reduction**

---

## Priority 1: Data Cleaning (IMMEDIATE)

### 1.1 Exclude Short Lets and Serviced Apartments

**Impact**: 31.8% reduction in squared error, 11% reduction in MAE

```python
def is_short_let(row):
    """Exclude from training set"""
    # Rule 1: Explicit type
    if 'short let' in row.get('property_type', '').lower():
        return True

    # Rule 2: Known serviced buildings
    serviced = ['cheval', 'thorney court', 'the other house',
                'ashburn place', '199 knightsbridge']
    if any(kw in row.get('address', '').lower() for kw in serviced):
        return True

    # Rule 3: Extreme PPSF (>20 = definitely not long-let)
    if row.get('ppsf', 0) > 20:
        return True

    return False
```

**Properties to exclude**: 142 (3.1% of data)

---

## Priority 2: HIGH-IMPACT Features

### 2.1 Size Non-Linearity Features

**Impact**: Captures U-shaped PPSF curve where tiny (<400 sqft) and huge (>3000 sqft) command premiums

```python
# Add to feature engineering
df['log_size'] = np.log(df['size_sqft'])
df['size_squared'] = df['size_sqft'] ** 2
df['is_tiny'] = (df['size_sqft'] < 400).astype(int)  # +12.6% PPSF premium
df['is_huge'] = (df['size_sqft'] >= 3000).astype(int)  # +24.2% PPSF premium
```

### 2.2 Ultra-Luxury Address Detection

**Impact**: Addresses systematic 22.7% underestimation of ultra-luxury

```python
ULTRA_PRESTIGE_STREETS = [
    'eaton place', 'eaton square', 'belgrave square', 'chester square',
    'wilton crescent', 'cadogan place', 'cadogan square', 'grosvenor square',
    'upper grosvenor street', 'hyde park gate', 'princes gate', 'hans place',
    'pont street', 'adams row', 'basil street', 'montpelier square',
    'pavilion road', 'sloane street', 'lancelot place', 'trevor square'
]

df['is_ultra_luxury_address'] = df['address'].apply(
    lambda x: any(s in x.lower() for s in ULTRA_PRESTIGE_STREETS)
).astype(int)
```

### 2.3 Social Housing Detection

**Impact**: Addresses 30-50% overestimation of affordable/institutional housing

```python
SOCIAL_ESTATE_PATTERNS = [
    r"world'?s?\s*end\s*estate", r"townshend\s*estate", r"hallfield\s*estate",
    r"churchill\s*gardens", r"ebury\s*bridge", r"peabody",
    r"lisson\s*grove", r"penfold\s*place"
]

def is_likely_social_housing(address, ppsf, postcode):
    # Known estates
    for pattern in SOCIAL_ESTATE_PATTERNS:
        if re.search(pattern, address.lower()):
            return 1
    # Premium postcode + very low PPSF
    premium = ['SW1', 'SW3', 'SW7', 'W1', 'W8', 'NW8']
    if any(postcode.startswith(p) for p in premium) and ppsf < 3.5:
        return 1
    return 0
```

### 2.4 Postcode Premium Adjustments

**Impact**: Addresses 25-47% systematic bias in specific districts

```python
POSTCODE_ADJUSTMENTS = {
    # Ultra-prime (systematically underestimated)
    'W1J': 1.41,  # +41% Mayfair
    'W1S': 1.47,  # +47% Mayfair
    'W1K': 1.25,  # +25% Mayfair
    'W8': 1.20,   # +20% Kensington
    'SW1X': 1.13, # +13% Belgravia

    # Outer areas (systematically overestimated)
    'SW14': 0.83, # -17% Mortlake
    'SW17': 0.65, # -35% Tooting
    'SW18': 0.80, # -20% Wandsworth
    'SW12': 0.64, # -36% Balham
}
```

---

## Priority 3: MEDIUM-IMPACT Features

### 3.1 Floor Level Features

**Impact**: +60% premium for penthouses, -22% discount for basement/garden flats

```python
df['is_penthouse'] = (
    (df['property_type'] == 'penthouse') |
    df['address'].str.contains('penthouse', case=False)
).astype(int)

df['is_basement'] = df['address'].str.contains(
    'basement flat|basement apartment', case=False
).astype(int)

df['is_garden_flat'] = df['address'].str.contains('garden flat', case=False).astype(int)
df['is_ground_floor'] = (df['has_ground'] == 1).astype(int)
```

### 3.2 Property Type Refinements

**Impact**: Terraced +39% PPSF vs standard house, serviced +high premium

```python
df['is_terraced'] = df['property_type'].isin(['terraced', 'town house']).astype(int)
df['is_studio'] = ((df['bedrooms'] == 1) & (df['size_sqft'] < 400)).astype(int)
df['is_serviced'] = df['property_type'].isin(['short let', 'serviced apartments']).astype(int)
df['is_houseboat'] = (df['property_type'] == 'house boat').astype(int)
```

### 3.3 Condition Keywords

**Impact**: Furnished +16% PPSF, refurbished +14% PPSF, long-let -28% PPSF

```python
def is_furnished(desc):
    if not desc: return None
    desc = desc.lower()
    if 'unfurnished' in desc: return 0
    if 'furnished' in desc: return 1
    return None

def has_refurb_keywords(desc):
    if not desc: return 0
    keywords = ['refurbished', 'newly decorated', 'newly renovated', 'brand new']
    return 1 if any(kw in desc.lower() for kw in keywords) else 0

df['is_long_let'] = (df['property_type'] == 'long let').astype(int)
```

### 3.4 Luxury Bathroom Signal

**Impact**: 4+ bathrooms = £1,142 higher underestimation

```python
df['bath_to_bed_ratio'] = df['bathrooms'] / df['bedrooms'].clip(lower=1)
df['has_ensuite_each'] = (df['bath_to_bed_ratio'] >= 1).astype(int)
df['high_bathroom_count'] = (df['bathrooms'] >= 4).astype(int)
df['excess_bathrooms'] = df['bathrooms'] - df['bedrooms']
```

---

## Priority 4: LOW-IMPACT/Optional

### 4.1 Agent Source (NOT recommended)
- Only 0.33% of residual variance explained by source
- Most "agent effects" already captured via size/location

### 4.2 Location Tier Encoding
```python
LOCATION_TIERS = {
    'tier1_mayfair': ['W1J', 'W1K', 'W1S'],
    'tier2_prime': ['SW1X', 'SW3', 'W8'],
    'tier3_good': ['SW7', 'SW1W', 'W2', 'W11'],
    'tier4_solid': ['SW5', 'SW6', 'SW10', 'W14'],
    'tier5_outer': ['SW12', 'SW13', 'SW14', 'SW17', 'SW18']
}
```

---

## Implementation Plan

### Phase 1: Data Cleaning
1. Filter out 142 short lets/serviced apartments
2. Fix 28 zero-bathroom records (data errors)
3. Fill missing postcode information (20 records)

### Phase 2: New Features (12 features)
```python
NEW_FEATURES = [
    # Size
    'log_size', 'size_squared', 'is_tiny', 'is_huge',
    # Location
    'is_ultra_luxury_address', 'is_social_housing', 'postcode_adjustment',
    # Property Type
    'is_terraced', 'is_penthouse', 'is_studio',
    # Luxury Signals
    'high_bathroom_count', 'bath_to_bed_ratio'
]
```

### Phase 3: Retraining
1. Retrain XGBoost with new features
2. Validate on holdout set
3. Compare MAPE/MAE to v0.9.9 baseline

### Expected Results

| Metric | Current (v0.9.9) | Target (v1.0) | Improvement |
|--------|------------------|---------------|-------------|
| MAPE | 19.7% | <15% | -24% |
| MAE | £1,497 | <£1,200 | -20% |
| Median APE | 15.1% | <12% | -21% |
| Worst segment error | 40-50% | <25% | -50% |

---

## Summary: Top 10 Features to Add

| Rank | Feature | Expected Impact | Complexity |
|------|---------|-----------------|------------|
| 1 | Exclude short lets | -11% MAE | Low |
| 2 | is_social_housing | -5% MAE on social segment | Low |
| 3 | is_ultra_luxury_address | -20% error on luxury | Low |
| 4 | postcode_adjustment | -10% on biased districts | Low |
| 5 | log_size | Better size modeling | Low |
| 6 | size_squared | U-shaped PPSF curve | Low |
| 7 | is_penthouse | +60% premium capture | Low |
| 8 | is_terraced | +39% premium capture | Low |
| 9 | high_bathroom_count | Luxury signal | Low |
| 10 | is_long_let | -28% discount capture | Low |

All 10 features are low complexity (simple regex or arithmetic) with measurable expected impact.
