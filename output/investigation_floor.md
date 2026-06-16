# Floor Level Investigation

## Executive Summary

Floor level has a significant impact on rental prices. Penthouses command a **56-66% premium** over baseline flats, while basement/garden flats see **22-25% discounts**. The database already contains rich floor-level information (21% coverage), but it's not being utilized in the prediction model.

---

## 1. Penthouse vs Other Flats PPSF Comparison

### Database Analysis (All Active Listings)

| Property Type | Count | Avg PPSF | vs Baseline |
|--------------|-------|----------|-------------|
| **Penthouse** | 34 | 10.38 | **+60.0%** |
| Other flats (flat/apartment) | 3,900 | 6.49 | baseline |

### Residuals Analysis (Model Performance)

| Property Type | Count | Mean Residual | Mean PPSF | Underpredicted % |
|--------------|-------|---------------|-----------|------------------|
| Penthouse | 28 | +873.76 | 9.45 | 43% |
| Regular flats | 3,468 | +369.92 | 6.05 | 39% |

**Key Finding**: Penthouses have a **56.2% PPSF premium** over regular flats in the residuals dataset. The model slightly underpredicts penthouses (mean residual +873), suggesting it's not fully capturing the premium.

### Sample High-Value Penthouses

| Address | PPSF | Price PCM | Size SqFt |
|---------|------|-----------|-----------|
| Egerton Gardens, SW3 | 28.46 | 17,333 | 609 |
| Nottingham Terrace, NW1 | 24.83 | 14,997 | 604 |
| Avenue Road, NW8 | 20.16 | 14,997 | 744 |
| Sloane Street, SW1X | 17.87 | 15,816 | 885 |

---

## 2. Floor-Level Information in the Data

### Existing Database Columns (High Coverage)

The database already has excellent floor-level columns from OCR enrichment:

| Column | Listings with Data | Coverage (of 5,610 with sqft) |
|--------|-------------------|------------------------------|
| floor_count | 1,574 | 28.1% |
| has_ground | 693 | 12.4% |
| has_first_floor | 575 | 10.2% |
| has_second_floor | 533 | 9.5% |
| has_lower_ground | 306 | 5.5% |
| has_third_floor | 303 | 5.4% |
| has_fourth_plus | 291 | 5.2% |
| has_basement | 94 | 1.7% |
| has_roof_terrace | 61 | 1.1% |
| has_mezzanine | 37 | 0.7% |

### Property Levels Column

| Level Type | Count | Avg PPSF |
|------------|-------|----------|
| single_floor | 858 | 6.17 |
| duplex | 244 | 6.89 |
| multi_floor | 14 | 7.68 |
| triplex | 45 | 6.04 |

### PPSF by Floor Columns (Flats Only)

| Column | Has=1 PPSF | Has=0 PPSF | Premium |
|--------|------------|------------|---------|
| has_roof_terrace | 8.03 | 6.07 | **+32.3%** |
| has_fourth_plus | 6.85 | 6.00 | +14.2% |
| has_basement | 6.82 | 6.07 | +12.4% |
| has_first_floor | 6.73 | 6.09 | +10.5% |
| has_second_floor | 6.34 | 6.18 | +2.6% |
| has_lower_ground | 6.32 | 6.09 | +3.8% |
| has_third_floor | 6.25 | 6.13 | +2.0% |
| **has_ground** | **6.03** | 6.20 | **-2.7%** |

---

## 3. Keywords Indicating Floor Level

### Comprehensive Keyword Analysis (32.6% of flats have keywords)

| Keyword | Count | Avg PPSF | vs Baseline (6.64) |
|---------|-------|----------|-------------------|
| **penthouse** | 97 | 11.03 | **+66.2%** |
| lower_ground | 99 | 6.96 | +4.9% |
| third_floor | 131 | 6.65 | +0.2% |
| fourth_floor | 95 | 6.38 | -3.9% |
| first_floor | 277 | 6.34 | -4.5% |
| sixth_plus | 54 | 6.33 | -4.6% |
| second_floor | 237 | 6.23 | -6.1% |
| fifth_floor | 52 | 6.10 | -8.1% |
| ground | 314 | 6.01 | **-9.4%** |
| raised_ground | 89 | 6.02 | -9.3% |
| top_floor | 91 | 5.34 | **-19.5%** |
| **garden_flat** | 31 | 5.14 | **-22.5%** |
| **basement** | 8 | 4.98 | **-25.0%** |

### Key Observations

1. **Penthouse keyword** shows massive +66% premium
2. **Basement and garden flat** keywords show 22-25% discounts
3. **Top floor** surprisingly shows -19.5% discount (possibly smaller units in attic conversions)
4. **Ground floor** shows consistent -9% discount
5. **Lower ground** shows slight +5% premium (possibly larger units with garden access)

---

## 4. Feature Recommendations

### High Priority Features (Strong PPSF Impact)

| Feature | Source | Expected Impact | Implementation |
|---------|--------|-----------------|----------------|
| **is_penthouse** | property_type='penthouse' OR keyword | +60-66% premium | `property_type == 'penthouse' OR 'penthouse' in description` |
| **is_basement** | keyword in address/description | -25% discount | `'basement' in text AND NOT 'with basement'` |
| **is_garden_flat** | keyword in address/description | -22% discount | `'garden flat' in text` |
| **is_ground_floor** | has_ground=1 OR keyword | -9% discount | `has_ground=1 OR 'ground floor' in text` |
| **has_roof_terrace** | database column | +32% premium | Already in DB |

### Medium Priority Features

| Feature | Source | Expected Impact |
|---------|--------|-----------------|
| **is_lower_ground** | has_lower_ground=1 OR keyword | +5% premium |
| **is_high_floor** | has_fourth_plus=1 | +14% premium |
| **is_duplex** | property_levels='duplex' | +12% premium |
| **floor_count** | database column | Non-linear effect |

### Feature Engineering Code

```python
def create_floor_features(df):
    """Create floor-level features from existing data"""

    # Text search function
    def has_keyword(row, keywords):
        text = (str(row.get('address', '')) + ' ' +
                str(row.get('description', '')) + ' ' +
                str(row.get('summary', ''))).lower()
        return any(kw in text for kw in keywords)

    # High-value penthouse indicator
    df['is_penthouse'] = (
        (df['property_type'] == 'penthouse') |
        df.apply(lambda r: has_keyword(r, ['penthouse']), axis=1)
    ).astype(int)

    # Discount indicators
    df['is_basement'] = df.apply(
        lambda r: has_keyword(r, ['basement flat', 'basement apartment']), axis=1
    ).astype(int)

    df['is_garden_flat'] = df.apply(
        lambda r: has_keyword(r, ['garden flat']), axis=1
    ).astype(int)

    df['is_ground_floor'] = (
        (df['has_ground'] == 1) |
        df.apply(lambda r: has_keyword(r, ['ground floor']), axis=1)
    ).astype(int)

    # Premium indicators
    df['has_roof_terrace_flag'] = (df['has_roof_terrace'] == 1).astype(int)
    df['is_high_floor'] = (df['has_fourth_plus'] == 1).astype(int)
    df['is_duplex'] = (df['property_levels'] == 'duplex').astype(int)

    return df
```

---

## 5. Model Impact Assessment

### Expected Improvements

1. **Penthouse predictions**: Adding `is_penthouse` should capture the 60%+ premium currently being partially missed (mean residual +873)

2. **Ground/basement predictions**: Adding `is_ground_floor`, `is_basement`, `is_garden_flat` should improve predictions for these discounted properties

3. **Overall model**: These features should reduce MAPE for ~33% of flats that have floor-level indicators

### Current Model Gaps

| Segment | Current Mean Residual | Issue |
|---------|----------------------|-------|
| Penthouse keyword | +2,289 | Significantly underpredicted |
| Lower ground keyword | -856 | Overpredicted (likely no sqft benefit from lower level) |
| Garden flat keyword | -274 | Slightly overpredicted |
| Top floor keyword | -105 | Neutral |

---

## 6. Data Quality Notes

### Coverage Summary

- **21% of listings** have explicit floor-level database columns
- **33% of flats** have floor-level keywords in text
- **property_type='penthouse'** captures 34 listings
- **property_type='ground flat'** captures only 9 listings (most ground floors not in property_type)

### Recommendations

1. **Use keyword extraction** - More comprehensive than property_type alone
2. **Combine database columns + keywords** - Maximum coverage
3. **Exclude "with basement" phrases** - These indicate extra storage, not basement dwelling
4. **Top floor needs investigation** - Unexpected discount may indicate attic/mansard conversions with lower ceiling height
