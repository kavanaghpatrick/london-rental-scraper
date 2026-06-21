/**
 * Next-free, single-source query logic for /api/similar-sale (getSimilarSaleListings).
 * FORK of similarQuery.js — the FOR-SALE analogue. Table `sale_listings`, price column
 * `asking_price`, sale-scale PPSF bounds, and a SALE-SPECIFIC `is_under_offer` exclusion
 * (drop SSTC / under-offer comps).
 *
 * WHY A PLAIN .js MODULE (not inline in saleDb.ts):
 *   The route's DB client is @vercel/postgres (-> @neondatabase/serverless) which cannot
 *   talk to a plain local Postgres (its `sql` path always goes through the neon HTTP
 *   `fetchEndpoint`). To give /api/similar-sale REAL pre-merge coverage WITHOUT a fragile
 *   neon proxy, we factor the SQL + scoring + stats into this pure module. saleDb.ts imports
 *   it (production behaviour unchanged — same SQL, same params, same stats), and the CI
 *   harness (dashboard/test/similar_sale_query_test.mjs) imports the SAME module and runs the
 *   EXACT query against a Postgres SERVICE CONTAINER via the plain `pg` driver. One source of
 *   truth -> the test exercises the real query.
 *
 * Dependency-free (no Next, no @vercel/postgres, no pg) so it loads in any Node/test context.
 * Returns a PARAMETERIZED ($1,$2,…) query both @vercel/postgres and pg accept identically.
 */

'use strict';

// Sale-scale price-per-sqft sanity bounds (rental was 3–30 £/sqft per MONTH; sale is the
// asking-price-per-sqft LUMP SUM). Documented sale constants; used by deriveSaleBounds so
// the test and route share the same scale.
const SALE_PPSF_MIN = 400;
const SALE_PPSF_MAX = 5000;

/**
 * Derive the safe integer bounds the scoring query needs from raw params.
 * Mirrors deriveBounds in similarQuery.js but on the sale price column (asking_price):
 * beds ±1, asking_price 0.5x–2x, sqft 0.7x–1.3x.
 */
function deriveSaleBounds(params) {
  const {
    postcodeDistrict,
    bedrooms,
    askingPrice,
    sizeSqft,
    propertyType,
    excludeId,
  } = params;

  // `?? 0` (nullish), matching similarQuery.js (Math.round never returns null/undefined,
  // so this is equivalent to plain Math.round — kept identical to the rental derivation).
  const safeBedrooms = Math.max(0, Math.round(bedrooms) ?? 0);
  const safeAskingPrice = Math.max(1, Math.round(askingPrice) ?? 1);
  const safeSizeSqft = sizeSqft ? Math.round(sizeSqft) : 0;

  const minBedrooms = Math.max(0, safeBedrooms - 1);
  const maxBedrooms = safeBedrooms + 1;

  const minSqft = safeSizeSqft > 0 ? Math.floor(safeSizeSqft * 0.7) : 0;
  const maxSqft = safeSizeSqft > 0 ? Math.ceil(safeSizeSqft * 1.3) : 99999;
  const minSqftWide = Math.floor(minSqft * 0.8);
  const maxSqftWide = Math.ceil(maxSqft * 1.2);

  const priceRangeMin = Math.floor(safeAskingPrice * 0.5);
  const priceRangeMax = Math.ceil(safeAskingPrice * 2.0);
  const priceTolerance15 = Math.round(safeAskingPrice * 0.15);
  const priceTolerance30 = Math.round(safeAskingPrice * 0.30);

  const safePropertyType = propertyType ?? '';
  const safeExcludeId = excludeId ?? '__NO_MATCH__';

  return {
    postcodeDistrict,
    safeBedrooms,
    safeAskingPrice,
    safeSizeSqft,
    minBedrooms,
    maxBedrooms,
    minSqft,
    maxSqft,
    minSqftWide,
    maxSqftWide,
    priceRangeMin,
    priceRangeMax,
    priceTolerance15,
    priceTolerance30,
    safePropertyType,
    safeExcludeId,
  };
}

/**
 * Build the parameterized sale similar-listings query (text + positional values array).
 * Same scoring weights as the rental query (beds 30%, size 25%, price 25%, type 10%,
 * source 10%), threshold similarity_score > 0.3, ORDER BY similarity_score DESC then
 * ABS(asking_price - $price) ASC, LIMIT 15. Adds the SALE-SPECIFIC is_under_offer
 * exclusion (drop SSTC comps). Returns { text, values }.
 */
function buildSaleSimilarQuery(params) {
  const b = deriveSaleBounds(params);

  // Positional params, in the order they're referenced below.
  const values = [
    b.safeBedrooms, // $1
    b.safeSizeSqft, // $2
    b.minSqft, // $3
    b.maxSqft, // $4
    b.minSqftWide, // $5
    b.maxSqftWide, // $6
    b.safeAskingPrice, // $7
    b.priceTolerance15, // $8
    b.priceTolerance30, // $9
    b.safePropertyType, // $10
    b.postcodeDistrict, // $11
    b.minBedrooms, // $12
    b.maxBedrooms, // $13
    b.priceRangeMin, // $14
    b.priceRangeMax, // $15
    b.safeExcludeId, // $16
  ];

  const text = `
    WITH scored AS (
      SELECT
        id,
        source,
        property_id,
        address,
        postcode,
        url,
        asking_price::int as asking_price,
        size_sqft::int as size_sqft,
        bedrooms::int as bedrooms,
        property_type,
        last_seen::text as last_seen,
        CASE WHEN size_sqft > 0 THEN ROUND((asking_price::numeric / size_sqft::numeric), 2)::float ELSE NULL END as ppsf,
        (
          CASE WHEN bedrooms = $1 THEN 0.30
               WHEN ABS(bedrooms - $1) = 1 THEN 0.15
               ELSE 0 END +
          CASE WHEN $2 > 0 AND size_sqft > 0 THEN
            CASE WHEN size_sqft BETWEEN $3 AND $4 THEN 0.25
                 WHEN size_sqft BETWEEN $5 AND $6 THEN 0.10
                 ELSE 0 END
          ELSE 0.15 END +
          CASE WHEN ABS(asking_price - $7) <= $8 THEN 0.25
               WHEN ABS(asking_price - $7) <= $9 THEN 0.15
               ELSE 0.05 END +
          CASE WHEN $10 != '' AND LOWER(property_type) = LOWER($10) THEN 0.10
               WHEN $10 = '' THEN 0.05
               ELSE 0 END +
          CASE WHEN source IN ('savills', 'knightfrank') THEN 0.10
               WHEN source IN ('chestertons', 'foxtons') THEN 0.07
               ELSE 0.05 END
        )::float as similarity_score
      FROM sale_listings
      WHERE is_active = 1
        AND SPLIT_PART(postcode, ' ', 1) = $11
        AND bedrooms BETWEEN $12 AND $13
        AND asking_price BETWEEN $14 AND $15
        AND asking_price > 0
        AND (is_under_offer IS NULL OR is_under_offer = 0)
        AND property_id != $16
    )
    SELECT *
    FROM scored
    WHERE similarity_score > 0.3
    ORDER BY similarity_score DESC, ABS(asking_price - $7) ASC
    LIMIT 15
  `;

  return { text, values };
}

/**
 * Compute the result stats from the returned rows. Mirrors computeSimilarStats in
 * similarQuery.js but on asking_price. Empty rows -> graceful-empty defaults
 * (peer_count 0, your_percentile 50, avg_ppsf null).
 */
function computeSaleSimilarStats(rows, params) {
  const safeAskingPrice = Math.max(1, Math.round(params.askingPrice) || 1);

  const peerCount = rows.length;
  const avgPrice =
    peerCount > 0
      ? Math.round(rows.reduce((sum, r) => sum + r.asking_price, 0) / peerCount)
      : null;

  const ppsfValues = rows
    .map((r) => r.ppsf)
    .filter((v) => v !== null && v !== undefined);
  const avgPpsf =
    ppsfValues.length > 0
      ? Math.round(
          (ppsfValues.reduce((sum, v) => sum + v, 0) / ppsfValues.length) * 100
        ) / 100
      : null;

  const prices = rows.map((r) => r.asking_price);
  const minPrice = prices.length > 0 ? Math.min(...prices) : null;
  const maxPrice = prices.length > 0 ? Math.max(...prices) : null;

  const belowCount = prices.filter((p) => p < safeAskingPrice).length;
  const yourPercentile =
    peerCount > 0 ? Math.round((belowCount / peerCount) * 100) : 50;

  return {
    peer_count: peerCount,
    avg_price: avgPrice,
    avg_ppsf: avgPpsf,
    min_price: minPrice,
    max_price: maxPrice,
    your_percentile: yourPercentile,
  };
}

module.exports = {
  SALE_PPSF_MIN,
  SALE_PPSF_MAX,
  deriveSaleBounds,
  buildSaleSimilarQuery,
  computeSaleSimilarStats,
};
