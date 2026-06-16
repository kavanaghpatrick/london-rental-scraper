# Agent/Brand Pricing Analysis - Rental Property Residuals

**Date**: 2026-01-15
**Dataset**: 4,528 listings with sqft and price data
**Sources**: Foxtons, Rightmove, Knight Frank, Chestertons, Savills

---

## Executive Summary

The analysis reveals **statistically significant differences** in pricing patterns across listing sources (ANOVA p=0.005). However, the practical impact is minimal - source alone explains only **0.33% of residual variance**. The current model already captures most agent-related pricing through property characteristics (size, location, bedrooms).

**Key Findings**:
1. Premium agents (Knight Frank, Savills) do NOT charge higher PPSF - they list larger properties in similar areas
2. Chestertons and Knight Frank have highest mean residuals (+GBP 780, +GBP 736)
3. Foxtons and Savills have lowest mean residuals (+GBP 250, +GBP 232)
4. Adding source as a model feature would provide marginal improvement (<0.5% R2)

---

## 1. Pricing Metrics by Source

### Price Per Square Foot (PPSF)

| Source | Count | Mean PPSF | Median PPSF | Std Dev |
|--------|-------|-----------|-------------|---------|
| rightmove | 2,622 | GBP 6.55 | GBP 5.47 | 3.78 |
| knightfrank | 582 | GBP 5.92 | GBP 5.13 | 2.73 |
| chestertons | 557 | GBP 5.63 | GBP 5.01 | 2.39 |
| foxtons | 402 | GBP 5.48 | GBP 4.91 | 2.66 |
| savills | 365 | GBP 5.46 | GBP 5.04 | 2.00 |

**Finding**: Rightmove has the HIGHEST mean PPSF (GBP 6.55), likely due to its aggregation of diverse agents. Premium agents (Knight Frank, Savills) have LOWER mean PPSF than the overall market.

### Mean Residuals by Source

| Source | Mean Residual | Std Dev | Median | Mean % Error | Mean Abs % Error |
|--------|---------------|---------|--------|--------------|------------------|
| chestertons | GBP +779.64 | 3,720 | GBP -103.79 | 4.58% | 23.45% |
| knightfrank | GBP +736.43 | 4,668 | GBP -257.95 | 4.85% | 21.54% |
| rightmove | GBP +388.59 | 2,831 | GBP -171.09 | 3.43% | 19.51% |
| foxtons | GBP +249.96 | 2,465 | GBP -185.33 | 4.25% | 16.45% |
| savills | GBP +232.19 | 2,548 | GBP -161.52 | 4.09% | 16.09% |

**Interpretation**:
- Positive residual = actual price > predicted = listings priced HIGHER than model expects
- All sources have positive mean residuals but negative medians, indicating right-skewed distributions (some very overpriced outliers)
- Chestertons and Knight Frank show highest mean residuals - their listings tend to be priced higher than the model predicts
- Savills and Foxtons have lowest mean residuals and best model fit (16% mean absolute % error)

---

## 2. Premium Agent Analysis

### Are Premium Agents More Expensive Per Sqft?

Comparing "premium" agents (Knight Frank, Savills) vs others:

| Metric | Premium Agents | Non-Premium | Difference |
|--------|----------------|-------------|------------|
| Mean PPSF | GBP 5.74 | GBP 6.29 | -GBP 0.55 |
| Mean Price (PCM) | GBP 8,935 | GBP 6,257 | +GBP 2,678 |
| Mean Size (sqft) | 1,461 | 991 | +470 |
| Mean Residual | GBP +542 | GBP +434 | +GBP 108 |

**Statistical Test**: T-test for PPSF difference
- T-statistic: -4.52, P-value: 0.000006
- **Result**: Premium agents list properties at SIGNIFICANTLY LOWER PPSF

**Interpretation**: Premium agents don't charge more per sqft - they simply deal in larger, more expensive properties. The higher absolute prices are explained by property size, not agent premium.

---

## 3. Detailed Source Breakdown

### Foxtons
- **N**: 402 listings
- **Portfolio**: Smaller properties (mean 1,002 sqft), predominantly flats (93%)
- **Pricing Accuracy**: Best model fit (16.45% mean abs error)
- **Split**: 35% overpriced / 65% underpriced
- **Top Areas**: SW6, SW7, SW3

### Rightmove (Aggregator)
- **N**: 2,622 listings (58% of dataset)
- **Portfolio**: Smallest mean size (934 sqft), diverse mix
- **Pricing Accuracy**: Good fit (19.51% mean abs error)
- **Split**: 41% overpriced / 59% underpriced
- **Note**: Aggregates listings from multiple agents, hence highest PPSF variance

### Knight Frank
- **N**: 582 listings
- **Portfolio**: Larger properties (mean 1,414 sqft), premium locations
- **Pricing Accuracy**: Moderate (21.54% mean abs error)
- **Split**: 40% overpriced / 60% underpriced
- **Anomaly**: Flats have high positive residuals (+GBP 803), houses have negative (-GBP 479)

### Chestertons
- **N**: 557 listings
- **Portfolio**: Medium-large (mean 1,251 sqft)
- **Pricing Accuracy**: Worst fit (23.45% mean abs error)
- **Split**: 45% overpriced / 55% underpriced
- **Note**: Highest proportion of overpriced listings

### Savills
- **N**: 365 listings
- **Portfolio**: Largest properties (mean 1,536 sqft), premium focus
- **Pricing Accuracy**: Best fit alongside Foxtons (16.09% mean abs error)
- **Split**: 42% overpriced / 58% underpriced
- **Note**: Despite premium reputation, residuals are lowest among all sources

---

## 4. Does the Model Account for Agent Brand?

### Current State
The model does NOT explicitly include source/agent as a feature. Evidence:
- ANOVA test shows significant differences between sources (p=0.005)
- F-regression shows knightfrank has significant predictive power (p=0.024)

### Variance Explained
- R-squared of residuals explained by source alone: **0.33%**
- This is very low - the model already captures most pricing variation through other features

### Why So Little Unexplained Variance?

The analysis reveals that "agent effects" are largely confounded with property characteristics:

| Source | Mean Size (sqft) | Top Postcode Districts |
|--------|------------------|------------------------|
| savills | 1,536 | SW7, SW11, NW8, SW3, SW1X |
| knightfrank | 1,414 | SW3, SW7, SW1X, W2, W8 |
| chestertons | 1,251 | SW11, SW7, NW1, SW1X, NW3 |
| foxtons | 1,002 | SW6, SW7, SW3, SW5, W8 |
| rightmove | 934 | SW3, NW8, SW7, SW6, W8 |

Premium agents (Savills, Knight Frank) specialize in larger properties - the model already accounts for this via size features. All agents operate in similar prime London postcodes - location effects are already captured.

---

## 5. Agent Brand Analysis (Where Populated)

The `agent_brand` column has only 511 populated records (11% of dataset). Analysis of agents with 5+ listings:

| Agent Brand | Count | Mean PPSF | Mean Residual | Notes |
|-------------|-------|-----------|---------------|-------|
| Sotheby | 5 | GBP 9.63 | +GBP 3,012 | Ultra-premium, high overpricing |
| City Relay | 5 | GBP 6.83 | +GBP 1,089 | Short-let specialist |
| Marsh & Parsons | 14 | GBP 5.14 | +GBP 151 | Near model predictions |
| Knight Frank | 231 | GBP 5.70 | +GBP 95 | Moderate overpricing |
| Winkworth | 9 | GBP 5.06 | +GBP 11 | Well-priced |
| Dexters | 15 | GBP 5.48 | -GBP 38 | Slight underpricing |
| Hamptons | 11 | GBP 4.68 | -GBP 38 | Slight underpricing |
| Savills | 6 | GBP 10.76 | -GBP 63 | Lower residuals than source data |
| Chestertons | 103 | GBP 4.53 | -GBP 185 | Underpriced vs predictions |

**Key Insight**: Ultra-premium brands (Sotheby, Domus Nova) show very high positive residuals (GBP 3,000-8,000), suggesting a "luxury brand premium" that the model does not capture.

---

## 6. Evidence for Agent Pricing Strategy

### Systematic Patterns Found

1. **Model Fit Varies by Agent**: Savills/Foxtons have 16% mean abs error vs 23% for Chestertons
2. **Bedroom Interaction**: 3-4 bedroom properties show highest positive residuals across all sources
3. **Property Type Interaction**: Knight Frank flats are overpriced (+GBP 803) but houses underpriced (-GBP 479)

### Patterns NOT Explained by Property Characteristics

1. **Chestertons consistently prices higher** than the model predicts across all bedroom counts
2. **Knight Frank 3-4 bed properties** have unusually high residuals (+GBP 1,788 to +GBP 3,892)
3. **Ultra-luxury agents** (Sotheby, Domus Nova) command brand premiums not captured by property features

---

## 7. Recommendation: Should Source/Agent Be a Model Feature?

### Arguments FOR Including Source

1. Statistical significance exists (ANOVA p=0.005)
2. Some agents (Chestertons, Knight Frank) show systematic overpricing
3. Ultra-luxury brands have persistent positive residuals
4. Could reduce mean absolute error by ~0.5-1%

### Arguments AGAINST Including Source

1. Only 0.33% of residual variance explained by source
2. Most "agent effects" are confounded with property characteristics already in the model
3. Risk of overfitting to source-specific quirks
4. Training data may not generalize if agent behavior changes
5. Philosophical concern: source is not a property characteristic but a listing artifact

### Recommendation: **DO NOT ADD SOURCE AS FEATURE**

The marginal improvement (<0.5% R-squared) does not justify the added model complexity. Instead:

1. **Monitor by source**: Continue tracking residuals by source to detect drift
2. **Add luxury brand indicator**: Consider a binary "ultra_luxury" feature for Sotheby/Domus Nova level agents (if brand data becomes more complete)
3. **Focus on property features**: The model already captures 99.67% of what source could explain

---

## 8. Actionable Next Steps

1. **Data Quality**: Populate `agent_brand` field more consistently - only 11% coverage limits brand-level analysis
2. **Monitoring Dashboard**: Create source-level residual tracking to detect pricing strategy changes
3. **Outlier Investigation**: Examine the 45% of Chestertons listings that are overpriced
4. **Luxury Segment**: If focusing on GBP 15k+/month rentals, consider adding luxury brand indicator
5. **Cross-Validation**: Test model performance separately by source to ensure no source has degraded predictions

---

## Appendix: Statistical Tests

### ANOVA: Do Sources Have Different Mean Residuals?
- F-statistic: 3.71
- P-value: 0.0051
- Result: **Statistically significant** - sources have different pricing patterns

### T-Test: Premium vs Non-Premium PPSF
- T-statistic: -4.52
- P-value: 0.000006
- Result: **Premium agents have LOWER PPSF** (contrary to expectation)

### T-Test: Premium vs Non-Premium Residuals
- T-statistic: 0.93
- P-value: 0.35
- Result: **No significant difference** in residuals between premium and non-premium agents

### F-Regression: Source Dummy Variables
| Source | F-score | P-value | Significant? |
|--------|---------|---------|--------------|
| foxtons | 1.84 | 0.175 | No |
| knightfrank | 5.12 | 0.024 | Yes |
| rightmove | 2.81 | 0.094 | No |
| savills | 1.95 | 0.162 | No |

Only Knight Frank shows individually significant predictive power for residuals.
