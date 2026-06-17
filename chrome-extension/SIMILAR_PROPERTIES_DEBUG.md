# Debug: "Similar properties" are totally off

**Symptom (user):** the extension's "similar properties" / comps are totally off (irrelevant — wrong location, etc.).

## Where comps come from (lead scoped this — start here)
- content.js `findSimilarProperties(fairValue, postcodeDistrict, beds, baths, amenities, limit=3)` (~L1600) scores listings from `similar_listings.json` (fetched from GitHub-raw `chrome-extension/api/similar_listings.json` via `loadSimilarListings()`).
- similar_listings.json is a DICT keyed `source:property_id` (4,095 entries). Each listing uses SHORT keys: `pr`=price, `p`=postcode district, `b`=beds, `ba`=baths, `u`=url, `a`=address, `s`=sqft, `am`=amenities.
- The current property's `postcodeDistrict` is extracted at content.js ~L504 and fed in.

## PRIME SUSPECT (lead found this — verify + fix)
The scoring lets PRICE-ONLY matches through with NO location match:
- Price: ≤10%→40, ≤20%→30, ≤30%→20, ≤50%→10, >50%→skip.
- Location: same district→+30, same area-prefix→+15, **else +0** (soft, optional).
- Beds ≤15, baths ≤10, amenities ≤5.
- **Include threshold: score ≥ 30.**
=> A listing within 10% price (=40 pts) passes the threshold with ZERO location match → a property anywhere in London with a similar price shows up as "similar." That is the "totally off" symptom: comps are price-matched but location-irrelevant. For a rental fair-value tool, location must be a hard gate (same district, or at least same area), not an optional bonus. Also the top-3 ranking by raw score will favor price over location.

## Other things to verify (each owner takes one)
1. **Scoring logic (PRIME):** confirm the price-only-passes bug; design the fix — make location a HARD gate (require same district, or fall back to same area only if too few district matches), and/or raise/re-weight so a wrong-location comp can't rank top-3. Don't over-correct into "no comps" for thin areas — define the fallback (district → area → none).
2. **Comps DATA quality:** is `similar_listings.json`'s `p` (district) field actually populated + correct for most listings? If `p` is often empty/null, location scoring NEVER fires → price-only matching by construction. Check field coverage (what % have p / pr / b / ba), and whether districts look sane. Note prior export issues: [[similar-listings-export-gotcha]] / is_active snapshots.
3. **Input extraction:** is the CURRENT property's `postcodeDistrict` (L504) + beds/baths/fairValue extracted correctly — especially on Chestertons (which just had the hydration-race fix)? Garbage-in (wrong district) → wrong comps even with perfect scoring. Verify on the repro chestertons URL (SW10 9HD / Redcliffe Gardens / 2bed).
4. **fairValue anchor:** the price-similarity is anchored on the MODEL estimate (fairValue), not the listing's asking price. Confirm that's intended + sane (a bad estimate would skew price-matching — but /api/predict was verified working this session).

## Constraints
- Agents can't run the real extension — investigate by reading content.js + inspecting similar_listings.json + tracing the data flow; the user tests the fix in-browser.
- If similar_listings.json's data/coverage is the problem, the fix may be in the EXPORT (scripts/export_similar_listings.py) not just content.js — flag which.
- A content.js fix must not break the predictor (keep xgboost.js untouched / parity 0/0).

## RESOLUTION (implemented)
Root cause = the scoring bug (confirmed). Location was a soft bonus and `score>=30`
let a within-10%-price listing (40pts) clear the gate with zero location match, so
comps came from anywhere in London — worst in thin districts (real proof: N6 2bed →
top-3 all NW). 87.5% of `p` was populated so data was NOT the primary cause.

Fix, three parts (xgboost.js untouched; predictor parity stays 0/0):
1. **content.js `findSimilarProperties`** — location is now a HARD GATE:
   tier0 = same district, tier1 = same area-prefix, everything else (incl.
   empty/garbage `p`) DISCARDED. Result = `tier0.concat(tier1).slice(0,limit)`,
   so a same-district comp always outranks a same-area one and a wrong-area comp
   can never appear. Thin districts degrade to fewer/zero comps, never garbage.
   Removed the `score>=30` include gate (price/beds/baths/amenities are now
   intra-tier tie-breakers only).
2. **content.js `extractPostcode`** — no longer returns the silent `'SW3'`
   default on a no-postcode address (that mislocated comps to Chelsea). Returns
   `null` → `postcode_district` becomes `''` → the `if (r.postcode_district && r.beds)`
   guard suppresses comps instead of faking a location. Regex anchored to a valid
   UK outcode and prefers the LAST match ("A12 Building" no longer reads as A12).
   New `normalizeDistrict()` helper keeps the input district in the SAME shape the
   gate compares against. (Model's own missing-postcode default lives in xgboost.js
   and is intentionally left alone.)
3. **scripts/export_similar_listings.py `get_postcode_district`** — strips the
   inward code so space-less full postcodes ("SW36SN") normalize to a clean
   district ("SW3") instead of garbage; backfills district from a full postcode in
   the address when the postcode field is missing. Regenerated similar_listings.json
   now has 0 malformed districts (was 121) and 7,804 entries (was a stale 4,095).

Test: `_similar_properties_test.mjs` runs the REAL findSimilarProperties over a
synthetic fixture + the real JSON and asserts no out-of-area comp is ever returned.
Run all guards: `node _content_load_test.mjs && node _similar_properties_test.mjs && node _fixture_diff.mjs`.
