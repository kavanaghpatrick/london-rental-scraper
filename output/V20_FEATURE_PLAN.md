# V20 Feature Engineering Plan

## Summary from 10 Parallel Investigation Agents

Based on analysis of 4,528 properties and their prediction residuals, here are the prioritized feature recommendations:

---

## TIER 1: EXCLUDE FROM TRAINING (Data Quality)

### 1.1 Short Lets / Serviced Apartments (~142 properties, 3.1%)
**Impact: 11% MAE reduction**

Detection rules:
```python
# Exclude if ANY of these match:
is_short_let = (
    'short let' in property_type.lower() or
    ppsf > 20 or  # Extreme pricing
    any(kw in address.lower() for kw in [
        'cheval', 'ashburn place', '199 knightsbridge',
        'thorney court', 'the other house'
    ])
)
```

### 1.2 Social Housing / Council Estates (~26-100 properties)
**Impact: 2-5% MAE improvement**

Detection rules:
```python
ESTATE_KEYWORDS = ['world\'s end estate', 'townshend estate', 'hallfield estate',
                   'ebury bridge', 'lisson grove', 'penfold place']
is_social_housing = (
    any(kw in address.lower() for kw in ESTATE_KEYWORDS) or
    (ppsf < 3.5 and is_prime_postcode)  # Anomalously cheap in prime area
)
```

---

## TIER 2: HIGH PRIORITY FEATURES (Significant Impact)

### 2.1 Penthouse Detection
**Impact: 60-66% PPSF premium, currently underpredicted**
```python
is_penthouse = (
    'penthouse' in property_type.lower() or
    'penthouse' in description.lower()
)
# 97 properties, mean PPSF 11.03 vs 6.49 baseline
```

### 2.2 Long Let Discount
**Impact: 28% lower PPSF - STRONGEST SIGNAL**
```python
is_long_let = let_type == 'long'
# 69% of long lets are overestimated by model
```

### 2.3 Terraced House Premium
**Impact: 39% higher PPSF, 47% higher errors**
```python
is_terraced = 'terraced' in property_type.lower()
# 96 properties - largest PPSF/error impact
```

### 2.4 Tiny Property Premium (< 400 sqft)
**Impact: 12.6% PPSF premium**
```python
is_tiny = size_sqft < 400
```

### 2.5 Roof Terrace Premium
**Impact: 32% premium - ALREADY IN DATABASE but not used!**
```python
# Just add has_roof_terrace to feature list
```

---

## TIER 3: MEDIUM PRIORITY FEATURES

### 3.1 Floor Level Features
```python
is_ground_floor = (has_ground == 1) or ('ground floor' in description.lower())  # -9%
is_garden_flat = 'garden flat' in description.lower()  # -22%
is_basement_flat = 'basement' in description.lower() and 'with basement' not in description.lower()  # -25%
is_high_floor = has_fourth_plus == 1  # +14%
```

### 3.2 Furnished Status (from description)
```python
is_furnished_from_desc = 'furnished' in desc and 'unfurnished' not in desc  # +16%
is_unfurnished_from_desc = 'unfurnished' in desc  # -16%
```

### 3.3 Property Condition Keywords
```python
has_refurb = any(kw in desc for kw in ['refurbished', 'newly refurbished', 'just refurbished'])  # +14%
has_luxury_finish = any(kw in desc for kw in ['high spec', 'luxury finish', 'designer'])
```

### 3.4 Studio Detection
```python
is_studio = (bedrooms == 1 and size_sqft < 400)  # 166 properties, +19% PPSF
```

### 3.5 Postcode Adjustment Multipliers
Top underestimated:
- W1J (Mayfair): +41%
- W1S (Mayfair): +47%
- W1K (Mayfair): +25%
- W8 (Kensington): +20%

Top overestimated:
- SW17 (Tooting): -35%
- SW12 (Balham): -36%

---

## TIER 4: LOWER PRIORITY / OPTIONAL

### 4.1 Bathroom Luxury Signals
```python
high_bathroom_count = bathrooms >= 4  # Luxury tier
excess_bathrooms = bathrooms - bedrooms  # Guest bathroom
```

### 4.2 Huge Property Flag
```python
is_huge = size_sqft >= 3000  # 24% PPSF premium
```

### 4.3 Property Subtypes
```python
is_duplex_maisonette = 'duplex' in type or 'maisonette' in type
is_houseboat = 'house boat' in type  # 6 properties, extreme PPSF
```

### 4.4 Ultra-Luxury Address Detection
```python
ULTRA_LUXURY_STREETS = ['eaton place', 'wilton crescent', 'adams row',
                         'grosvenor square', 'basil street', 'charles street']
is_ultra_luxury_address = any(st in address.lower() for st in ULTRA_LUXURY_STREETS)
```

---

## NOT RECOMMENDED

### Agent/Source Features
- Agent effect is minimal (0.33% R² improvement)
- Model already captures agent effects through property characteristics
- Skip adding agent-specific features

---

## IMPLEMENTATION PRIORITY ORDER

1. **Exclude short lets and serviced apartments** (clean data)
2. **Add is_penthouse** (largest impact)
3. **Add is_long_let** (strongest signal)
4. **Add is_terraced** (distinct pricing)
5. **Add has_roof_terrace** (already in DB!)
6. **Add is_tiny** (< 400 sqft premium)
7. **Add floor level features** (ground, garden flat, basement, high floor)
8. **Add postcode adjustment multipliers** (W1J, SW17, etc.)

---

## EXPECTED OVERALL IMPROVEMENT

- **Data cleaning (short lets)**: 11% MAE reduction
- **Tier 2 features**: 5-10% additional MAE reduction
- **Tier 3 features**: 3-5% additional MAE reduction

**Total estimated improvement: 15-25% MAE reduction**

Current MAE: £1,497 → Target MAE: ~£1,100-1,250
