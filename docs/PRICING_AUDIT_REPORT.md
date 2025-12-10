# Pricing Audit Report

**Date**: 2025-12-07
**Status**: Root cause analysis complete

---

## Executive Summary

Analyzed 132 suspicious listings (73 low ppsf, 59 high ppsf) across all sources. Found two primary root causes:

1. **Size Extraction Errors** (majority) - sqft values incorrectly parsed
2. **Weekly vs Monthly Confusion** (Savills primarily) - weekly prices stored as monthly

---

## Issue Type 1: Size Extraction Errors

### Symptoms
- **LOW ppsf (<£2.5/sqft)**: Size overestimated (e.g., 1,025 stored as 11,105)
- **HIGH ppsf (>£25/sqft)**: Size underestimated (e.g., 8,135 stored as 135)

### Verified Cases

| Source | Property ID | DB sqft | Page sqft | Error Type |
|--------|------------|---------|-----------|------------|
| savills | savills_353589 | 11,105 | 1,025 | Extra digit (10x) |
| savills | savills_551884 | 169 | 3,397 | Truncated |
| rightmove | 170043998 | 1,106 | 238 | Wrong value |
| rightmove | 166613738 | 135 | 8,135 | Missing "8," |
| rightmove | 167983163 | 300 | 6,300 | Missing "6," |

### Root Causes

1. **OCR extraction from floorplans**: Some sizes come from OCR which can misread digits
2. **HTML parsing errors**: Regex capturing partial numbers or wrong elements
3. **JSON field mismatch**: Some sources have multiple size fields (min/max)

### Fix Strategy

For extreme outliers (ppsf < 1.5 or ppsf > 50):
- Mark `size_sqft = NULL` to trigger re-enrichment
- Do NOT auto-correct (we don't know the right value)

---

## Issue Type 2: Weekly vs Monthly Confusion

### Symptoms
- Price looks correct but ppsf is ~4.33x too low
- Affects Savills primarily

### Verified Cases

| Source | Property ID | DB Price | Page Shows | Correct PCM |
|--------|------------|----------|------------|-------------|
| savills | savills_925418 | £850 pcm | £850 Weekly | £3,683 |
| savills | savills_579040 | £1,330 pcm | £1,330 Weekly | £5,763 |

### Root Cause

Savills spider extracts prices but the "Per Week" / "Per Month" text parsing may fail, defaulting to monthly when it's actually weekly.

### Fix Strategy

For Savills listings where:
- ppsf < 2.5 AND
- page shows weekly pricing

Convert: `price_pcm = price_pcm * 52 / 12`

---

## Issue Type 3: Legitimate Low-Price Properties

Some low ppsf listings are correct - large properties in outer areas:
- Knight Frank Canterbury (4,011 sqft, £2,600 pcm) - rural property
- Chestertons Putney - outer London prices are lower

### How to Identify
- ppsf between 2.0-2.5 may be legitimate
- Check postcode - outer London (SW13-19) vs prime (SW1, SW3, SW7)

---

## Counts by Issue Type (Estimated)

| Issue Type | Est. Count | Fix Approach |
|------------|------------|--------------|
| Size errors (high ppsf >50) | ~45 | NULL size, re-enrich |
| Size errors (low ppsf <1.5) | ~30 | NULL size, re-enrich |
| Weekly as monthly (Savills) | ~10-15 | Multiply price ×4.33 |
| Legitimate low ppsf | ~30 | No action |

---

## Recommended Cleanup Script

```python
# 1. Clear obviously wrong sizes (will trigger re-enrichment)
UPDATE listings
SET size_sqft = NULL
WHERE is_active = 1
  AND size_sqft > 0
  AND price_pcm > 0
  AND (
    price_pcm * 1.0 / size_sqft > 50  -- HIGH ppsf (size underestimated)
    OR price_pcm * 1.0 / size_sqft < 1.5  -- LOW ppsf (size overestimated)
  );

# 2. Savills weekly pricing fix (after manual verification)
UPDATE listings
SET price_pcm = CAST(price_pcm * 52.0 / 12 AS INTEGER)
WHERE source = 'savills'
  AND is_active = 1
  AND size_sqft > 0
  AND price_pcm * 1.0 / size_sqft < 2.0
  AND property_id IN (...verified IDs...);
```

---

## Next Steps

1. Run comprehensive Playwright verification on all 132 suspicious listings
2. Categorize each into: size_error, weekly_error, legitimate
3. Execute targeted fixes per category
4. Re-run enrichment to fill NULL sizes
5. Validate with new ppsf distribution

---

## Appendix: URLs Verified

### Batch 1 (Low PPSF)
- savills/gbnellwfl240044l: SIZE ERROR (11,105 → 1,025)
- savills/gbwprewsl160058l: WEEKLY ERROR (£850 pw → £850 pcm)
- rightmove/169861337: Price correct, size needs verification
- rightmove/170043998: SIZE ERROR (1,106 → 238)
- knightfrank/hub2538212: LEGITIMATE (rural Canterbury)

### Batch 2 (Mixed)
- chestertons/21236397: CORRECT (1,993 ≈ 2,000)
- savills/gbnnrelhl250830l: WEEKLY ERROR (£1,330 pw → £1,330 pcm)
- savills/gbpirephl250047l: SIZE ERROR (169 → 3,397)

### Batch 1 (High PPSF)
- rightmove/166613738: SIZE ERROR (135 → 8,135)
- rightmove/167983163: SIZE ERROR (300 → 6,300)
