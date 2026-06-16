# Investigation: Postcode Micro-Location Effects on Model Residuals

**Date:** 2026-01-15
**Dataset:** 4,528 rental properties across 81 postcode districts and 401 full postcodes
**Objective:** Identify systematic postcode-level bias in rental price predictions

---

## Executive Summary

The model exhibits significant systematic bias at the postcode district level, with certain ultra-prime areas (W1J, W1K, W1S) consistently **underestimated** by 25-47% and outer London areas (SW12, SW14, SW17, SW18) consistently **overestimated** by 17-36%. The current model uses basic postcode features (one-hot encoding for top 15 districts, frequency encoding, prime postcode flag) but lacks the granularity needed to capture micro-location premiums.

---

## 1. Top 10 Systematically Biased Postcode Districts

### Most Underestimated (Model Predicts Too Low)

| District | Mean Residual | Count | Mean % Error | Avg Price | Suggested Adjustment |
|----------|---------------|-------|--------------|-----------|---------------------|
| **W1J** (Mayfair) | +5,256 | 33 | -27.8% | 12,795 | **+41%** |
| **W1S** (Mayfair) | +5,098 | 5 | -28.7% | 10,841 | **+47%** |
| **W1K** (Mayfair) | +4,015 | 60 | -19.1% | 15,847 | **+25%** |
| **W8** (Kensington) | +1,998 | 247 | -11.0% | 9,883 | **+20%** |
| **SW1X** (Belgravia) | +1,802 | 217 | -0.9% | 14,345 | **+13%** |

### Most Overestimated (Model Predicts Too High)

| District | Mean Residual | Count | Mean % Error | Avg Price | Suggested Adjustment |
|----------|---------------|-------|--------------|-----------|---------------------|
| **SW14** (Mortlake) | -1,255 | 6 | +25.9% | 7,484 | **-17%** |
| **SW17** (Tooting) | -1,164 | 5 | +43.9% | 3,370 | **-35%** |
| **SW18** (Wandsworth) | -1,032 | 27 | +23.2% | 5,104 | **-20%** |
| **SW12** (Balham) | -990 | 10 | +39.8% | 2,745 | **-36%** |
| **SW13** (Barnes) | -935 | 15 | +19.2% | 9,030 | **-10%** |

---

## 2. Street-Level Patterns Discovered

### Ultra-Premium Addresses (Systematically Underestimated)

These specific addresses show the model cannot capture their true premium:

| Address | District | Mean Residual | # Listings | Issue |
|---------|----------|---------------|------------|-------|
| Ashburn Place | SW7 | +14,710 | 3 | Luxury serviced apartments |
| Eaton Place, Belgravia | SW1X | +12,869 | 4 | Prime Belgravia mansion flats |
| Charles Street, Mayfair | W1J | +12,205 | 5 | Ultra-prime Mayfair location |
| Duke Street, Mayfair | W1K | +11,308 | 3 | Adjacent to Grosvenor Square |
| Wilton Crescent | SW1X | +10,418 | 3 | Prime crescent in Belgravia |
| Basil Street | SW3 | +10,344 | 10 | Steps from Harrods |
| Prince of Wales Terrace, W8 | W8 | +9,633 | 3 | Premium Kensington address |
| Pavilion Road | SW1X | +8,188 | 7 | Knightsbridge village location |
| Hyde Park Gate | SW7 | +7,424 | 7 | Hyde Park-facing premium |

### Overvalued Addresses (Model Too High)

| Address | District | Mean Residual | # Listings | Issue |
|---------|----------|---------------|------------|-------|
| South Eaton Place | SW1W | -3,757 | 3 | Secondary Belgravia street |
| Kensington Road | W8 | -2,628 | 4 | Traffic-heavy main road |
| Collingham Gardens | SW5 | -2,029 | 5 | Lower SW5 area |
| Draycott Place | SW3 | -2,001 | 3 | Less desirable end of SW3 |
| Imperial Wharf | SW6 | -1,330 | 4 | New development, lower premiums |

---

## 3. Current Model Postcode Features

From `rental_price_models_v15.py`, the model uses:

```python
# Location features currently in model:
- is_prime_postcode      # Binary: SW1, SW3, SW7, SW10, W8, W11, W2, W1
- postcode_freq          # Frequency encoding (count-based, no price info)
- postcode_area_freq     # Area-level frequency (SW, W, etc.)
- pc_{district}          # One-hot for top 15 districts only
```

### Problems with Current Approach:

1. **Binary prime flag too coarse** - W1J (Mayfair) and W2 (Paddington) both flagged as "prime" but have vastly different premiums
2. **Only top 15 districts encoded** - Districts like SW14, SW17, SW18 have no specific features
3. **No street-level features** - Cannot capture within-district variations (Eaton Place vs South Eaton Place)
4. **No price-per-sqft encoding** - Removed in V15 to prevent leakage, but lost valuable location signal

---

## 4. Price-Per-Sqft Analysis by District

The model lacks this critical location signal:

| District | Avg PPSF | Median PPSF | Comment |
|----------|----------|-------------|---------|
| **W1J** | 10.13 | 9.64 | Highest - Ultra prime Mayfair |
| **SW1X** | 9.39 | 7.28 | Second highest - Belgravia |
| **W1S** | 8.65 | 5.45 | Mayfair South |
| **W1K** | 8.52 | 8.06 | Mayfair North |
| **W1G** | 8.06 | 5.61 | Marylebone Medical |
| **SW7** | 7.66 | 6.02 | South Kensington |
| **W8** | 7.45 | 6.12 | Kensington |
| SW12 | 3.55 | - | Balham - Low premium |
| SW17 | 3.43 | - | Tooting - Lowest |

The spread from 3.43 (SW17) to 10.13 (W1J) represents a **3x variation** in price-per-sqft that the model cannot capture without proper encoding.

---

## 5. Recommendations

### Immediate Actions (High Impact)

#### 5.1 Add Postcode Premium/Discount Table
Create explicit adjustment factors for inference:

```python
POSTCODE_ADJUSTMENTS = {
    # Ultra-prime (underestimated)
    'W1J': 1.41,   # +41%
    'W1S': 1.47,   # +47%
    'W1K': 1.25,   # +25%
    'W8':  1.20,   # +20%
    'SW1X': 1.13,  # +13%

    # Outer areas (overestimated)
    'SW14': 0.83,  # -17%
    'SW17': 0.65,  # -35%
    'SW18': 0.80,  # -20%
    'SW12': 0.64,  # -36%
    'SW13': 0.90,  # -10%
}
```

#### 5.2 Add Target-Encoded Postcode Features (Training)
Use leave-one-out encoding to add price signal without leakage:

```python
# For each property, calculate avg price of OTHER properties in same district
df['postcode_price_loo'] = df.groupby('postcode_district')['price_pcm'].transform(
    lambda x: (x.sum() - x) / (len(x) - 1)
)
```

#### 5.3 Expand District One-Hot Encoding
Increase from top 15 to top 30 districts to capture SW12, SW14, SW17, SW18, etc.

### Model Retraining (Medium-Term)

#### 5.4 Add Granular Location Features
- **Full postcode features** - One-hot for postcodes with 5+ samples (currently 47 postcodes qualify)
- **Street-type flags** - Mews, Place, Gardens, Square, Court, Road (already have `is_mews`)
- **Building-level features** - Named developments (Chelsea Harbour, Imperial Wharf, etc.)

#### 5.5 Geographic Clustering
Replace simple prime/non-prime with multi-tier location bands:
- **Tier 1:** W1J, W1K, W1S (Mayfair core)
- **Tier 2:** SW1X, SW3, W8 (Belgravia/Chelsea/Kensington)
- **Tier 3:** SW7, SW1W, W2, W11 (South Ken/Notting Hill)
- **Tier 4:** SW5, SW6, SW10, W14 (Earls Court/Fulham)
- **Tier 5:** SW12, SW13, SW14, SW15, SW17, SW18 (Outer)

---

## 6. Validation Requirements

Any postcode adjustments should be validated against:
1. **Holdout set** - 20% of properties never seen during training
2. **Time-based split** - Properties listed in last 3 months vs earlier
3. **Per-district MAPE** - Target <15% mean absolute percentage error per district

---

## Appendix: Raw Data

### District-Level Statistics (n >= 5)

```
District    Mean Residual    Count    Mean % Error    Avg PPSF
W1J              +5,256         33         -27.8%      10.13
W1S              +5,098          5         -28.7%       8.65
W1K              +4,015         60         -19.1%       8.52
W8               +1,998        247         -11.0%       7.45
SW1X             +1,802        217          -0.9%       9.39
W1G              +1,389         30         -10.1%       8.06
W11              +1,239        169          -5.2%       6.09
W1T                +953         13          -6.8%       6.67
SW1H               +904          6          -7.3%         -
SW7                +864        359          +0.8%       7.66
W2                 +755        225          -1.5%       6.37
W1U                +715         73          -1.4%       7.23
W1W                +471         46          -3.3%       6.17
W14                +266        104          +2.9%       5.24
...
SW1P               -335         31         +10.7%         -
W9                 -476          6         +16.3%         -
NW6                -510          7         +22.7%         -
SW1W               -481        114         +12.8%       6.19
E1W                -558          7         +30.6%         -
SW1E               -575         11          +4.6%         -
W12                -608         21         +19.1%         -
SW15               -662         18         +21.4%         -
SW19               -786         14         +25.3%         -
NW11               -815         12         +25.4%         -
SW13               -935         15         +19.2%       3.93
SW12               -990         10         +39.8%       3.55
SW18             -1,032         27         +23.2%       3.65
SW17             -1,164          5         +43.9%       3.43
SW14             -1,255          6         +25.9%       3.88
```

---

**Prepared by:** Claude Code Analysis
**Next Steps:** Implement postcode adjustment table and retrain model with expanded features
