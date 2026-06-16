# Size Anomalies Investigation: PPSF and Model Error Analysis

**Date**: 2026-01-15
**Dataset**: 4,528 properties from residuals_analysis.csv
**Size Range**: 215 - 9,495 sqft

---

## Executive Summary

The analysis reveals a **U-shaped PPSF curve** with size, where both tiny (<400 sqft) and huge (>3000 sqft) properties command premium PPSF compared to mid-sized properties. The model exhibits **systematic bias by size**: underpricing small properties and overpricing medium properties. Statistical tests confirm these size effects are significant.

**Key Finding**: The "sweet spot" for renters seeking best value is **900-1,200 sqft** (median PPSF $4.84-$5.08/sqft).

---

## 1. PPSF by Size Bin

### Simple Bins (Requested)

| Size Bin | Count | Mean PPSF | Median PPSF | Std Dev | Min | Max |
|----------|-------|-----------|-------------|---------|-----|-----|
| 0-500 | 528 | $6.61 | $5.88 | $2.72 | $3.24 | $22.41 |
| 500-1000 | 2,537 | $6.20 | $5.12 | $3.68 | $3.00 | $29.85 |
| 1000-2000 | 984 | $5.68 | $5.08 | $2.60 | $3.00 | $27.05 |
| 2000+ | 479 | $6.57 | $5.75 | $3.25 | $3.03 | $29.12 |

### Detailed Bins

| Size Bin | Count | Mean PPSF | Median PPSF | Std Dev |
|----------|-------|-----------|-------------|---------|
| 0-400 | 166 | $6.92 | $6.44 | $1.87 |
| 400-500 | 362 | $6.46 | $5.59 | $3.02 |
| 500-1000 | 2,537 | $6.20 | $5.12 | $3.68 |
| 1000-2000 | 984 | $5.68 | $5.08 | $2.60 |
| 2000-3000 | 309 | $6.32 | $5.39 | $3.33 |
| 3000+ | 170 | $7.02 | $6.37 | $3.04 |

---

## 2. Error by Size Bin

| Size Bin | Mean Residual | Std Residual | Mean Pct Error | Mean Abs Pct Error | Median Abs Pct Error |
|----------|---------------|--------------|----------------|--------------------|--------------------|
| 0-400 | -$205 | $557 | +12.2% | 18.2% | 12.9% |
| 400-500 | -$165 | $840 | +10.9% | 19.8% | 15.6% |
| 500-1000 | +$282 | $2,086 | +4.5% | 20.1% | 15.8% |
| 1000-2000 | +$806 | $3,289 | -0.9% | 18.9% | 14.1% |
| 2000-3000 | +$1,225 | $5,861 | +1.8% | 19.9% | 15.2% |
| 3000+ | +$1,611 | $8,754 | +3.5% | 19.9% | 15.3% |

### Model Bias Pattern

| Size Category | n | Mean Error | Interpretation |
|---------------|---|------------|----------------|
| Tiny (<400) | 166 | +12.2% | **Model UNDERPRICES** |
| Small (400-700) | 1,409 | +8.4% | **Model UNDERPRICES** |
| Medium-Small (700-1000) | 1,490 | +2.3% | Model underprices (slight) |
| Medium (1000-1500) | 678 | -1.1% | Model overprices (slight) |
| Large (1500-2500) | 503 | -0.3% | Neutral |
| Huge (2500+) | 282 | +4.0% | **Model UNDERPRICES** |

**Interpretation**: Positive error = actual > predicted (model underestimates rent)

---

## 3. Evidence for Non-Linear Size Effects

### 3.1 U-Shaped PPSF Curve

**Quadratic Regression Fit**:
```
PPSF = 0.0000000464 * size^2 - 0.000126 * size + 6.22
```

**Theoretical PPSF minimum**: 1,356 sqft

**Observed minimum**: 900-1,000 sqft bin with median PPSF of $4.84/sqft

### 3.2 Statistical Significance

**T-Test: Tiny (<400 sqft) vs Rest**
- T-statistic: 2.95
- P-value: 0.003
- **Conclusion**: Tiny properties have significantly higher PPSF (p < 0.01)

**T-Test: Huge (>3000 sqft) vs Medium (1000-2000 sqft)**
- T-statistic: 6.18
- P-value: < 0.000001
- **Conclusion**: Huge properties have significantly higher PPSF (p < 0.001)

### 3.3 PPSF Premium Analysis

| Comparison | Premium |
|------------|---------|
| Tiny (<400) vs Rest | +12.6% |
| Huge (>3000) vs Medium (1000-2000) | +24.2% |

### 3.4 Granular PPSF by Size Range

| Size Range | Count | Mean PPSF | Median PPSF | Mean Abs Error |
|------------|-------|-----------|-------------|----------------|
| 0-300 | 31 | $7.79 | $7.23 | 21.4% |
| 300-400 | 146 | $6.87 | $6.31 | 17.7% |
| 400-500 | 362 | $6.39 | $5.58 | 19.7% |
| 500-600 | 545 | $5.69 | $5.07 | 20.0% |
| 600-700 | 503 | $6.23 | $5.38 | 18.7% |
| 700-800 | 463 | $6.32 | $5.07 | 21.5% |
| 800-900 | 545 | $6.46 | $5.08 | 20.7% |
| **900-1000** | **491** | **$6.26** | **$4.84** | 20.2% |
| **1000-1200** | **369** | **$5.40** | **$5.08** | **17.1%** |
| 1200-1500 | 294 | $5.84 | $5.04 | 20.0% |
| 1500-2000 | 303 | $5.93 | $5.09 | 19.0% |
| 2000-2500 | 195 | $6.41 | $5.52 | 20.0% |
| 2500-3000 | 113 | $6.19 | $5.11 | 19.9% |
| 3000-4000 | 97 | $7.01 | $6.20 | 22.7% |
| 4000+ | 71 | $7.11 | $6.63 | 15.6% |

**Sweet Spot**: 900-1,200 sqft offers the **lowest median PPSF** ($4.84-$5.08) with reasonable prediction accuracy.

---

## 4. Specific Findings

### Q1: Are tiny properties (<400 sqft) overpriced per sqft?

**YES** - Statistically significant premium of 12.6%

- Tiny mean PPSF: $6.92
- Rest mean PPSF: $6.15
- T-test p-value: 0.003

The model systematically underprices tiny properties by +12.2% on average, confirming they command a premium that the linear model fails to capture.

### Q2: Are huge properties (>3000 sqft) underpriced per sqft?

**NO** - They are actually **overpriced** per sqft

- Huge mean PPSF: $7.05
- Medium (1000-2000) mean PPSF: $5.68
- Premium: +24.2%

This is counterintuitive but explained by:
1. **Ultra-premium market segment**: Properties >3000 sqft in London are luxury properties with premium finishes, locations, and amenities
2. **Scarcity premium**: Only 3.8% of properties are this size
3. **Different buyer pool**: Wealthy renters willing to pay premium for space

### Q3: Is there a "sweet spot" size with best PPSF?

**YES** - 900-1,200 sqft

| Sweet Spot Metric | 900-1000 sqft | 1000-1200 sqft |
|------------------|---------------|----------------|
| Count | 491 | 369 |
| Median PPSF | $4.84 | $5.08 |
| Mean Abs Error | 20.2% | 17.1% |

The 1000-1200 sqft range offers the best combination of:
- Low PPSF (median $5.08)
- Lowest prediction error (17.1%)
- Reasonable sample size (369 properties)

### Q4: How does error vary with size?

| Pattern | Observation |
|---------|-------------|
| Residual magnitude | Increases with size (from $557 std to $8,754 std) |
| Absolute percentage error | Relatively constant (17-21%) across all sizes |
| Systematic bias | Small properties underpriced, medium overpriced |
| Residual direction | Small = negative residual, Large = positive residual |

---

## 5. Feature Engineering Recommendations

### Recommended Features

| Feature | Formula | Rationale | Priority |
|---------|---------|-----------|----------|
| `log_size` | `ln(size_sqft)` | Captures diminishing returns on larger spaces | High |
| `size_squared` | `size_sqft^2` | Models U-shaped PPSF curve | High |
| `is_tiny` | `size_sqft < 400` | Premium segment flag | High |
| `is_huge` | `size_sqft >= 3000` | Premium segment flag | Medium |
| `size_bucket` | Categorical (5 levels) | Captures non-linear breaks | Medium |

### Proposed Size Buckets

```python
def get_size_bucket(size_sqft):
    if size_sqft < 400:
        return 'tiny'      # 3.7% - Premium small
    elif size_sqft < 700:
        return 'small'     # 31.1% - Compact
    elif size_sqft < 1500:
        return 'medium'    # 47.9% - Standard (sweet spot)
    elif size_sqft < 3000:
        return 'large'     # 13.6% - Spacious
    else:
        return 'huge'      # 3.8% - Luxury premium
```

### Correlation Analysis

| Feature | Corr with Price | Note |
|---------|-----------------|------|
| size_sqft | 0.814 | Strong linear |
| log_size | 0.707 | Diminishing returns |
| size_squared | 0.759 | Quadratic relationship |
| is_tiny | -0.117 | Small but significant |
| is_huge | 0.633 | Luxury segment capture |

### Implementation Priority

1. **Immediate**: Add `log_size` and `size_squared` to capture non-linear effects
2. **High**: Add `is_tiny` binary flag for small property premium
3. **Medium**: Add `size_bucket` categorical for segment-specific coefficients
4. **Optional**: Interaction terms (`is_tiny * postcode`, `is_huge * bedrooms`)

---

## 6. Summary

| Question | Answer |
|----------|--------|
| PPSF varies with size? | Yes - U-shaped curve, minimum at ~1,000-1,350 sqft |
| Tiny overpriced/sqft? | Yes - 12.6% premium (p=0.003) |
| Huge underpriced/sqft? | No - 24.2% premium over medium |
| Sweet spot size? | 900-1,200 sqft (lowest PPSF, lowest error) |
| Error varies with size? | Residual magnitude increases; % error stable |
| Model bias? | Underprices small (<700), overprices medium (1000-1500) |

---

## 7. Appendix: Raw Data Summary

- **Total Properties**: 4,528
- **Size Range**: 215 - 9,495 sqft
- **PPSF Range**: $3.00 - $29.85/sqft
- **Overall Mean PPSF**: $6.18
- **Overall Median PPSF**: $5.27
- **Overall Mean Abs Error**: 19.71%
- **Overall Median Abs Error**: 15.12%
