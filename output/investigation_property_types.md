# Property Type Investigation Report

**Date**: 2026-01-15
**Dataset**: 4,528 rental listings from residuals_analysis.csv
**Current flags**: is_mews, is_house, is_flat

---

## 1. PPSF and Error by Detailed Property Type

| Property Type | Count | PPSF Mean | PPSF Median | Abs Error Mean | Abs Error Median |
|--------------|-------|-----------|-------------|----------------|------------------|
| flat | 2,569 | 5.77 | 5.06 | 18.97% | 14.61% |
| apartment | 899 | 6.85 | 5.65 | 19.69% | 15.51% |
| long let | 404 | 5.75 | 5.05 | 25.31% | 21.63% |
| house | 337 | 6.12 | 5.17 | 17.02% | 13.03% |
| terraced | 87 | 9.36 | 7.19 | 24.16% | 19.21% |
| short let | 69 | 7.39 | 6.73 | 26.38% | 25.67% |
| maisonette | 32 | 5.82 | 4.64 | 22.38% | 16.51% |
| penthouse | 28 | 9.45 | 7.04 | 11.97% | 6.07% |
| mews | 24 | 8.14 | 7.26 | 17.90% | 9.39% |
| not specified | 19 | 8.53 | 5.98 | 22.95% | 18.00% |
| detached | 11 | 7.60 | 5.79 | 17.34% | 12.16% |
| duplex | 10 | 7.13 | 5.75 | 20.10% | 15.48% |
| town house | 9 | 9.27 | 9.24 | 13.66% | 15.06% |
| end of terrace | 8 | 6.26 | 5.81 | 16.76% | 17.93% |
| semi-detached | 7 | 6.43 | 4.69 | 14.43% | 14.43% |
| house boat | 6 | 11.86 | 12.53 | 18.15% | 20.85% |
| serviced apartments | 2 | 10.18 | 10.18 | 30.19% | 30.19% |
| house of multiple occupation | 1 | 4.26 | 4.26 | 12.39% | 12.39% |
| ground flat | 1 | 4.28 | 4.28 | 4.92% | 4.92% |
| parking | 1 | 6.62 | 6.62 | 23.28% | 23.28% |

---

## 2. Analysis of Key Questions

### Q1: Are "apartment" and "flat" treated the same? Should they be?

**Current Status**: Both are flagged with `is_flat=1` (100% coverage for both types).

**Statistical Comparison**:
| Metric | Flat (n=2,569) | Apartment (n=899) | Difference |
|--------|----------------|-------------------|------------|
| PPSF mean | 5.77 | 6.85 | +18.7% higher |
| PPSF median | 5.06 | 5.65 | +11.7% higher |
| Price median | 3,791 | 4,290 | +13.2% higher |
| Abs Error median | 14.61% | 15.51% | Similar |

**Statistical Tests**:
- T-test PPSF: t=-8.73, **p<0.0001** (highly significant)
- Mann-Whitney U: **p<0.0001**

**Source Analysis**:
- "Apartment" comes almost exclusively from Rightmove (896/899 = 99.7%)
- "Flat" comes from all sources (Rightmove 53%, Knight Frank 18%, Foxtons 15%, Savills 11%)

**RECOMMENDATION**: **Consider separating** - The significant PPSF difference suggests "apartment" may indicate slightly higher-end properties (possibly because Rightmove uses "apartment" for more premium listings). However, since the error rates are similar, the current treatment is acceptable. A new `is_apartment` flag could capture this premium signal.

---

### Q2: Does "terraced" have systematically different PPSF than "house"?

**Statistical Comparison**:
| Metric | House (n=337) | Terraced (n=87) | Difference |
|--------|---------------|-----------------|------------|
| PPSF mean | 6.12 | 9.36 | +53% higher |
| PPSF median | 5.17 | 7.19 | +39% higher |
| Size median | 1,902 sqft | 985 sqft | -48% smaller |
| Abs Error median | 13.03% | 19.21% | +47% higher error |

**Statistical Tests**:
- T-test PPSF: t=-6.91, **p<0.0001** (highly significant)
- Mann-Whitney U: **p<0.0001**

**Current Flag Coverage**:
- House with is_house=1: 83.7%
- Terraced with is_house=1: 94.3%

**RECOMMENDATION**: **Create separate `is_terraced` flag** - Terraced houses have dramatically higher PPSF (smaller homes command premium per sqft in London) and significantly worse model errors. This distinct pricing behavior warrants a separate feature.

---

### Q3: Are "maisonette" and "duplex" priced differently?

**Statistical Comparison**:
| Metric | Maisonette (n=32) | Duplex (n=10) |
|--------|-------------------|---------------|
| PPSF mean | 5.82 | 7.13 |
| PPSF median | 4.64 | 5.75 |
| Price median | 4,998 | 6,200 |
| Size median | 932 sqft | 883 sqft |
| Abs Error median | 16.51% | 15.48% |

**Statistical Test**: Mann-Whitney U p=0.0841 (not significant at 0.05 level, but small samples)

**Current Flags**: Both have is_flat=1

**RECOMMENDATION**: **Combine as `is_duplex_maisonette`** - Despite small samples, both are multi-floor flat variants with similar error profiles. A combined flag distinguishes them from single-floor flats. Note: maisonettes behave more like standard flats (PPSF ~4.64), while duplexes are pricier.

---

### Q4: Is "studio" captured?

**Current Status**: No studio property type exists, and no is_studio flag.

**Analysis of Likely Studios**:
- 1-bedroom properties under 400 sqft: **166 records**
- These have higher PPSF (6.44 median vs 5.43 for all 1-beds)
- Error rate is actually lower (12.88% median)

**Evidence**:
- No "studio" property_type in data
- 0-bedroom properties: 0 records
- "Studio" in address: only 4 records

**RECOMMENDATION**: **Create `is_studio` flag** using size threshold (e.g., 1-bed AND size_sqft < 400). Studios command premium PPSF due to smaller denominator. The model may already be handling this via size interactions, but explicit flagging could help.

---

### Q5: House boats and other unusual types

**House Boats (n=6)**:
- Extremely high PPSF: 11.86 mean, 12.53 median (2x normal flats)
- All located at Cheyne Walk, SW10 (Chelsea waterfront)
- Currently flagged as is_house=1 (incorrect - not a house)
- Moderate errors (18-22%)

**Other Unusual Types**:

| Type | Count | Issue | PPSF |
|------|-------|-------|------|
| short let | 69 | Tenure type, not property type; no flags set | 7.39 |
| long let | 404 | Tenure type, not property type; no flags set | 5.75 |
| serviced apartments | 2 | Premium, high PPSF (10.18) | 10.18 |
| not specified | 19 | No flags set, unknown actual type | 8.53 |
| parking | 1 | Data error - likely attached to a flat | 6.62 |

**RECOMMENDATION**:
- Create `is_houseboat` flag (very distinct pricing)
- Create `is_serviced` flag for short let + serviced apartments
- Clean "long let" and "short let" by inferring actual property type from size/location

---

## 3. Categories to Combine vs Separate

### Should COMBINE (treat the same):
| Current Types | Reasoning | Recommended Flag |
|--------------|-----------|------------------|
| flat + apartment | Both residential flats, PPSF difference likely source artifact | is_flat (keep) |
| maisonette + duplex | Multi-floor flat variants, similar errors | is_duplex_maisonette (new) |
| house + detached + semi-detached + end of terrace | Standard house pricing | is_house (keep, expand coverage) |
| serviced apartments + short let | Premium serviced/furnished, high PPSF | is_serviced (new) |

### Should SEPARATE (distinct pricing):
| Type | From | PPSF Difference | Reasoning |
|------|------|-----------------|-----------|
| terraced | house | +39% higher median | Smaller homes, premium PPSF |
| penthouse | flat | +39% higher median | Luxury premium |
| mews | house | +40% higher median | Already has is_mews flag |
| house boat | house | +142% higher median | Waterfront premium |
| town house | house | +79% higher median | Premium house variant |
| studio | 1-bed flat | +19% higher PPSF | Size-based premium |

---

## 4. Missing Property Type Flags to Add

### High Priority (significant impact):

| Flag | Definition | Count | Rationale |
|------|------------|-------|-----------|
| `is_terraced` | property_type in ('terraced', 'town house') | 96 | +39-79% PPSF premium, 24% error vs 17% for house |
| `is_penthouse` | property_type == 'penthouse' | 28 | +39% PPSF premium, but lowest errors (6.07%!) |
| `is_studio` | bedrooms == 1 AND size_sqft < 400 | 166 | Higher PPSF, size-based premium |
| `is_serviced` | property_type in ('short let', 'serviced apartments') | 71 | Premium furnished/serviced, currently no flags |

### Medium Priority:

| Flag | Definition | Count | Rationale |
|------|------------|-------|-----------|
| `is_duplex_maisonette` | property_type in ('duplex', 'maisonette') | 42 | Multi-floor flat distinction |
| `is_houseboat` | property_type == 'house boat' | 6 | Extreme PPSF (12.53 median) |
| `is_apartment` | property_type == 'apartment' | 899 | May capture Rightmove premium signal |

### Low Priority:

| Flag | Definition | Count | Rationale |
|------|------------|-------|-----------|
| `is_long_let` | property_type == 'long let' | 404 | Needs property type inference first |

---

## 5. Recommendations for Type Encoding

### Immediate Actions:

1. **Fix flag coverage gaps**: 486 records have all flags=0 (mostly "long let" and "short let"). These have 25.35% average error vs 19.71% overall.

2. **Add `is_terraced` flag**: Largest impact - 87 records with distinct 9.36 PPSF vs 6.12 for house.

3. **Add `is_penthouse` flag**: 28 records but model is already predicting these well (6.07% median error). Explicit flag could improve other premium flats.

4. **Add `is_studio` flag**: Use size threshold. 166 properties with distinct pricing behavior.

5. **Reclassify house boats**: Remove from is_house (they're floating apartments), create is_houseboat.

### Encoding Strategy Options:

**Option A: Expand Binary Flags (Recommended)**
```python
# Keep existing
is_flat, is_house, is_mews

# Add new
is_terraced = property_type in ('terraced', 'town house')
is_penthouse = property_type == 'penthouse'
is_studio = (bedrooms == 1) & (size_sqft < 400)
is_serviced = property_type in ('short let', 'serviced apartments')
is_houseboat = property_type == 'house boat'
```

**Option B: Categorical Encoding**
```python
property_category = {
    'standard_flat': ['flat', 'apartment', 'maisonette', 'duplex', 'ground flat'],
    'premium_flat': ['penthouse', 'serviced apartments'],
    'standard_house': ['house', 'detached', 'semi-detached', 'end of terrace'],
    'premium_house': ['terraced', 'town house', 'mews'],
    'unusual': ['house boat', 'short let', 'long let'],
    'unknown': ['not specified', 'parking', 'house of multiple occupation']
}
```

**Option C: PPSF-Based Grouping**
```python
# Group by observed PPSF behavior
low_ppsf = ['flat', 'maisonette', 'ground flat', 'long let']  # median ~5.0
mid_ppsf = ['apartment', 'house', 'detached', 'duplex']  # median ~5.5-6.0
high_ppsf = ['terraced', 'penthouse', 'mews', 'town house', 'short let']  # median ~7.0+
extreme_ppsf = ['house boat', 'serviced apartments']  # median ~10+
```

---

## 6. Summary of Key Findings

1. **Flat vs Apartment**: Statistically different PPSF (p<0.0001) but similar error rates. Current treatment is acceptable; separation optional.

2. **Terraced vs House**: SIGNIFICANTLY different (p<0.0001). Terraced has 39% higher PPSF and 47% higher errors. **Must separate**.

3. **Maisonette vs Duplex**: Not statistically different (p=0.08). Can combine as multi-floor flat variant.

4. **Studio**: Not captured. 166 small 1-beds have distinct 19% PPSF premium. **Should add flag**.

5. **House Boat**: 6 records, extremely high PPSF (12.53 median). Currently mislabeled as houses. **Must separate**.

6. **Coverage Gaps**: 486 records (11%) have no property type flags set. These have 25% vs 20% error. **Must fix**.

---

## Appendix: Flag Coverage Matrix

| Property Type | Count | is_flat | is_house | is_mews | Needs New Flag |
|--------------|-------|---------|----------|---------|----------------|
| flat | 2,569 | 100% | 0% | 1% | - |
| apartment | 899 | 100% | 0% | 0% | is_apartment? |
| long let | 404 | 0% | 0% | 2% | Need inference |
| house | 337 | 0% | 84% | 16% | - |
| terraced | 87 | 0% | 94% | 6% | is_terraced |
| short let | 69 | 0% | 0% | 6% | is_serviced |
| maisonette | 32 | 100% | 0% | 0% | is_duplex_maisonette |
| penthouse | 28 | 100% | 0% | 0% | is_penthouse |
| mews | 24 | 0% | 0% | 100% | - |
| not specified | 19 | 0% | 0% | 0% | Unknown |
| detached | 11 | 0% | 91% | 9% | - |
| duplex | 10 | 100% | 0% | 0% | is_duplex_maisonette |
| town house | 9 | 0% | 100% | 0% | is_terraced |
| end of terrace | 8 | 0% | 75% | 25% | - |
| semi-detached | 7 | 0% | 86% | 14% | - |
| house boat | 6 | 0% | 100% | 0% | is_houseboat |
| serviced apartments | 2 | 100% | 0% | 0% | is_serviced |
