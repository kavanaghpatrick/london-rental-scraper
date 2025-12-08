# Data Quality Issues - December 2025

## Summary

Database has **4,133 listings** but only **2,513 unique fingerprints** (39% potential duplicates).

---

## Issue 1: Fingerprint Algorithm Bug

**Severity**: HIGH
**Impact**: Cross-source deduplication fails completely

**Example - Gerald Road, Belgravia, SW1W:**
```
rightmove:    f7205b0d5b4eec81
knightfrank:  a7c6abb8c3abfb24
```

Same address → different fingerprints. The fingerprint service is not normalizing addresses consistently across sources.

**Suspected causes to investigate:**
- Different address formats between sources ("Gerald Road, Belgravia, London, SW1W" vs "Gerald Road, Belgravia, SW1W")
- Punctuation differences (commas, periods)
- Word order differences
- "London" inclusion/exclusion
- Postcode format variations

**File to fix:** `property_scraper/services/fingerprint.py`

---

## Issue 2: Same-Source Duplicates

**Severity**: MEDIUM
**Impact**: Training data inflation, skewed statistics

Same source is inserting duplicate records with identical address+price:

| Source | Address | Count |
|--------|---------|-------|
| foxtons | Kensington Road | 3 |
| knightfrank | 10 Park Drive, Canary Wharf | 2 |
| knightfrank | Duchess Of Bedford House | 2 |
| knightfrank | Lonsdale Road | 2 |
| knightfrank | Wood Lane, White City | 2 |
| chestertons | Gloucester Road, South Kensington | 2 |
| chestertons | St Augustines Road, Camden | 2 |
| rightmove | 1 Kensington High Street | 2 |
| rightmove | Richmond Court | 2 |

**Root cause to investigate:**
- Pipeline deduplication using `source:property_id` but property_id may differ for same listing
- Re-scraping same listing with new property_id
- Spider yielding same item multiple times

**Files to check:**
- `property_scraper/pipelines.py` - DuplicateFilterPipeline
- Individual spider files for duplicate yields

---

## Issue 3: Cross-Source Duplicates (Rightmove Aggregation)

**Severity**: HIGH
**Impact**: Model sees same property multiple times, inflating training data

Rightmove aggregates listings from agents. Same property appears on both:

| Fingerprint | Count | Sources | Example Address |
|-------------|-------|---------|-----------------|
| 2ae6c6e8411f1542 | 34 | rightmove | Cromwell Road, SW7 |
| 2134a140795c669d | 30 | rightmove, foxtons | Kensington Gardens Square |
| 81c4a67877e730d1 | 21 | rightmove | Cadogan Square, SW1X |
| fd48436a8ce0264a | 21 | rightmove, foxtons | Prince of Wales Terrace, W8 |
| 81684bdc59ab1ae2 | 16 | rightmove | Knightsbridge SW7 |
| 9d04cc725135eb05 | 13 | knightfrank, rightmove | Eaton Place, Belgravia |

**Note:** Even with fingerprint bug, some duplicates ARE being detected (same source).

**Solution approach:**
1. Fix fingerprint algorithm first
2. Re-generate all fingerprints
3. Run cross-source deduplication
4. Keep canonical record (prefer agent over Rightmove)

---

## Issue 4: Exact Duplicate Groups

**Severity**: MEDIUM
**Impact**: 15 groups of exact duplicates (same address + price)

| Address | Postcode | Price | Count |
|---------|----------|-------|-------|
| Addison Road, London, W14 | W14 | £17,983 | 3 |
| Duchess Of Bedford House | W8 | £6,998 | 3 |
| Herbert Crescent | SW1X | £6,716 | 3 |
| Kensington Road | None | £13,000 | 3 |
| King's Quay, Chelsea Harbour | SW10 | £12,500 | 3 |
| Lonsdale Road | W11 | £5,416 | 3 |
| South Eaton Place | SW1W | £17,116 | 3 |
| Wood Lane, White City | W12 | £2,899 | 3 |

---

## Issue 5: Missing Postcodes

**Severity**: LOW
**Impact**: Location features unavailable

Some listings have `postcode = None`:
- "Kensington Road" - £13,000
- "Richmond Court" - £4,333

**Solution:** Extract postcode from address string if present

---

## Issue 6: Model Training Contamination

**Severity**: HIGH
**Impact**: Artificially inflated model performance

With duplicates in dataset:
- Same property may appear in both train and test sets
- Model "memorizes" specific properties
- Cross-validation scores are overly optimistic

**Current state:**
- 4,133 total records
- ~2,513 unique properties (estimated)
- ~1,620 duplicates (39%)

**Solution:** Deduplicate BEFORE model training

---

## Recommended Fix Order

1. **Fix fingerprint algorithm** - Make it robust to address format variations
2. **Re-generate all fingerprints** - Update existing records
3. **Remove same-source duplicates** - Keep most recent
4. **Mark cross-source duplicates** - Set canonical_id
5. **Retrain model on deduplicated data** - Get honest performance metrics

---

## Investigation Commands

```bash
# Check fingerprint service
python3 -c "
from property_scraper.services.fingerprint import generate_fingerprint
print(generate_fingerprint('Gerald Road, Belgravia, London, SW1W', 'SW1W'))
print(generate_fingerprint('Gerald Road, Belgravia, SW1W', 'SW1W'))
"

# Find all duplicates
sqlite3 output/rentals.db "
SELECT address, postcode, price_pcm, COUNT(*) as cnt
FROM listings
GROUP BY address, postcode, price_pcm
HAVING cnt > 1
ORDER BY cnt DESC;
"

# Check fingerprint distribution
sqlite3 output/rentals.db "
SELECT address_fingerprint, COUNT(*) as cnt
FROM listings
WHERE address_fingerprint IS NOT NULL
GROUP BY address_fingerprint
HAVING cnt > 1
ORDER BY cnt DESC
LIMIT 20;
"
```

---

## Files to Modify

| File | Change |
|------|--------|
| `property_scraper/services/fingerprint.py` | Improve normalization |
| `property_scraper/pipelines.py` | Better duplicate detection |
| `dedupe_cross_source.py` | Run after fingerprint fix |
| `rental_price_models_v14.py` | Train on deduplicated data |

---

*Document created: 2025-12-08*
