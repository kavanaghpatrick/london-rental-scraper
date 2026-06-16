# Bathroom Ratio Investigation: Luxury Signal Analysis

**Date:** 2026-01-15
**Dataset:** 4,528 properties from residuals_analysis.csv
**Focus:** Bathroom-to-bedroom ratio as predictor of rental premium

---

## Executive Summary

Properties with **bathroom-to-bedroom ratio >= 1.0** (en-suite for each bedroom) command a **13.3% PPSF premium** (statistically significant, p < 0.001). However, the current model already captures most of this effect through size and location - the difference in **residuals is not significant** (p = 0.61). The stronger signal comes from **raw bathroom count**, which correlates with underestimation (r = 0.13) and indicates luxury properties the model misses.

**Key Finding:** High bathroom counts (4+) show £1,142 higher underestimation than standard properties - this is an actionable luxury signal not fully captured by current features.

---

## 1. PPSF by Bathroom Ratio Bins

| Ratio Bin | Mean PPSF | Median PPSF | Count | Interpretation |
|-----------|-----------|-------------|-------|----------------|
| 0-0.5     | £5.37     | £4.79       | 991   | Budget tier (shared bath) |
| 0.5-0.75  | £6.18     | £5.03       | 558   | Standard family homes |
| 0.75-1.0  | £6.37     | £5.47       | 2,826 | Modern standard |
| 1.0-1.25  | £6.38     | £4.84       | 16    | Rare - mostly 1-bed flats |
| 1.25-1.5  | £9.40     | £6.53       | 48    | **Luxury tier** |
| 1.5-2.0   | £7.33     | £6.15       | 58    | Ultra-luxury |

**Insight:** The jump from ratio 0.75-1.0 to 1.25+ shows a clear luxury premium (+47% PPSF). Properties in the 1.25-1.5 range are outliers worth investigating for luxury amenities.

---

## 2. Error Analysis by Bathroom Ratio

| Ratio Bin | Mean Residual | Median Residual | Mean % Error | Underest. Rate |
|-----------|---------------|-----------------|--------------|----------------|
| 0-0.5     | £363          | -£179           | 3.0%         | ~43%           |
| 0.5-0.75  | £715          | -£65            | 0.1%         | ~43%           |
| 0.75-1.0  | £439          | -£200           | 4.8%         | ~39%           |
| 1.0-1.25  | £986          | -£564           | 2.7%         | ~44%           |
| 1.25-1.5  | £549          | £115            | 0.3%         | **~55%**       |
| 1.5-2.0   | £504          | £93             | -1.7%        | **~55%**       |

**Key Finding:** Properties with ratios >= 1.25 are **underestimated more often** (55% vs 40% baseline). This suggests the model misses luxury signals present in high-bathroom-ratio properties.

---

## 3. Correlation Analysis

### Bathroom Features vs. Residual

| Feature              | Correlation with Residual | Correlation with PPSF |
|---------------------|---------------------------|----------------------|
| bathrooms (raw)     | **0.1342**                | 0.1909               |
| bath_to_bed_ratio   | -0.0087                   | 0.1285               |
| excess_bathrooms    | -0.0196                   | 0.1129               |
| has_excess_bath     | 0.0065                    | 0.0925               |

**Interpretation:**
- **Raw bathroom count** has the strongest correlation with residual - higher bathrooms = more underestimation
- Ratio-based features correlate more with PPSF but **not with prediction error**
- The model may already capture ratio effects via size/location but misses the **luxury signal from absolute bathroom counts**

### Statistical Tests

| Test | T-statistic | P-value | Significant? |
|------|-------------|---------|--------------|
| En-suite vs non-en-suite PPSF | 7.34 | <0.0001 | **YES** |
| En-suite vs non-en-suite Residual | -0.51 | 0.61 | NO |

**Conclusion:** The 13.3% PPSF premium is real but the model already accounts for it. The residual difference is noise.

---

## 4. High Bathroom Count as Luxury Signal

| Bathroom Count | Mean Residual | Mean PPSF | Count | Underest. Rate |
|----------------|---------------|-----------|-------|----------------|
| 0              | £2,688        | £7.56     | 28    | 57% (data issue) |
| 1              | £56           | £5.60     | 2,074 | 35% |
| 2              | £521          | £6.14     | 1,653 | 43% |
| 3              | £1,360        | £7.94     | 484   | 49% |
| **4**          | **£1,319**    | £7.88     | 166   | **54%** |
| **5**          | £947          | £6.25     | 68    | 41% |
| **6**          | £1,719        | £7.73     | 18    | 39% |
| **7**          | **£7,998**    | £9.36     | 6     | **83%** |

**Key Findings:**
- Properties with **4+ bathrooms** average £1,404 underestimation vs £262 for 1-2 bathrooms
- **7-bathroom properties** are severely underestimated (avg £7,998) - likely ultra-luxury
- Zero-bathroom entries are data errors (missing data encoded as 0)

### Luxury Tier Summary

| Segment | Count | Avg Residual | Underest. Rate |
|---------|-------|--------------|----------------|
| Standard (1-2 bath) | 3,727 | £262 | 39% |
| Luxury (4+ bath) | 258 | £1,404 | **50%** |
| **Delta** | - | **£1,142** | **+11pp** |

---

## 5. Excess Bathrooms Analysis

"Excess bathrooms" = bathrooms - bedrooms (e.g., 3-bed with 4 baths = +1 excess)

| Excess Bathrooms | Mean PPSF | Mean Residual | Count |
|------------------|-----------|---------------|-------|
| -2 or less       | £5.73     | £973          | 314   |
| -1               | £5.67     | £377          | 1,296 |
| 0 (matched)      | £6.37     | £436          | 2,765 |
| +1               | **£8.11** | £678          | 119   |
| +2               | £4.39     | -£3,113       | 3     |

**Insight:** Properties with +1 excess bathroom show the highest PPSF (£8.11) - these are likely premium builds with guest/powder rooms. The +2 category is too small for conclusions.

### By Property Size

| Size Category | Without Excess (Residual) | With Excess (Residual) | Delta |
|--------------|---------------------------|------------------------|-------|
| <500 sqft    | -£187                     | +£580                  | +£767 |
| 500-800      | +£64                      | +£644                  | +£580 |
| 800-1200     | +£534                     | +£1,342                | +£808 |
| 1200-2000    | +£1,056                   | +£304                  | -£752 |
| 2000+        | +£1,458                   | +£164                  | -£1,294 |

**Insight:** Excess bathrooms signal luxury most strongly in **smaller properties** (<1200 sqft). In larger properties, extra bathrooms are expected and don't add premium.

---

## 6. Feature Engineering Recommendations

### High Priority Features

| Feature | Type | Formula | Expected Impact |
|---------|------|---------|-----------------|
| `bath_to_bed_ratio` | Float | bathrooms / bedrooms | Captures en-suite premium |
| `has_ensuite_each` | Binary | 1 if ratio >= 1.0 | Simple luxury flag |
| `high_bathroom_count` | Binary | 1 if bathrooms >= 4 | Luxury tier identifier |

### Medium Priority Features

| Feature | Type | Formula | Rationale |
|---------|------|---------|-----------|
| `excess_bathrooms` | Integer | bathrooms - bedrooms | Luxury signal (guest bath) |
| `excess_bath_small_property` | Binary | 1 if excess > 0 AND sqft < 1200 | Interaction - strongest signal |

### Data Quality

| Feature | Type | Formula | Purpose |
|---------|------|---------|---------|
| `zero_bathroom_flag` | Binary | 1 if bathrooms = 0 | Flag data issues |

### Implementation Priority

```python
# Recommended feature additions
df['bath_to_bed_ratio'] = df['bathrooms'] / df['bedrooms']
df['has_ensuite_each'] = (df['bath_to_bed_ratio'] >= 1).astype(int)
df['high_bathroom_count'] = (df['bathrooms'] >= 4).astype(int)
df['excess_bathrooms'] = df['bathrooms'] - df['bedrooms']
df['excess_bath_small'] = ((df['excess_bathrooms'] > 0) & (df['size_sqft'] < 1200)).astype(int)
```

---

## 7. Key Takeaways

1. **En-suite premium is real** (13.3% PPSF) but **already captured** by the model via size/location proxies

2. **Raw bathroom count correlates with underestimation** (r = 0.13) - this is actionable

3. **4+ bathroom properties are 50% underestimated** vs 39% baseline - a clear luxury signal

4. **Excess bathrooms matter most in small properties** (<1200 sqft) where they're unexpected

5. **Zero-bathroom entries are data errors** (28 records) - should be excluded or imputed

### Model Improvement Estimate

Adding these bathroom features should reduce underestimation of luxury properties by capturing:
- ~£1,142 avg residual gap for 4+ bathroom properties (258 properties)
- ~£800 avg residual gap for excess-bathroom small properties (~70 properties)

**Expected MAPE improvement:** 0.3-0.5 percentage points on luxury segment

---

## Appendix: Data Quality Notes

- **Missing bathroom data:** 31 records (0.7%)
- **Zero bathroom records:** 28 records - likely data entry errors for:
  - Cheval Thorney Court (luxury, definitely has bathrooms)
  - Other premium addresses with £16-17 PPSF
- **Recommendation:** Impute missing bathrooms based on bedroom count median for postcode
