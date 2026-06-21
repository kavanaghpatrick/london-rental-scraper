/**
 * Next-free, single-source query logic for /api/similar (getSimilarListings).
 *
 * WHY A PLAIN .js MODULE (not inline in db.ts):
 *   The route's DB client is @vercel/postgres, which wraps @neondatabase/serverless
 *   and CANNOT talk to a plain local Postgres (its `sql` path always goes through the
 *   neon HTTP `fetchEndpoint`, so a service-container Postgres needs a neon-proxy). To
 *   give the /api/similar behaviour REAL pre-merge test coverage WITHOUT that fragile
 *   proxy, we factor the SQL + scoring + stats into this pure module. db.ts imports it
 *   (production behaviour unchanged — same SQL, same params, same stats), and the CI
 *   harness (chrome-extension/.. no — dashboard/test/similar_query_test.mjs) imports
 *   the SAME module and runs the EXACT query against a Postgres SERVICE CONTAINER via
 *   the plain `pg` driver. One source of truth → the test exercises the real query.
 *
 * This module is dependency-free (no Next, no @vercel/postgres, no pg) so it loads in
 * any Node/test context. It returns a PARAMETERIZED ($1,$2,…) query, which both
 * @vercel/postgres (via its template tag building the same positional params) and pg
 * accept identically.
 */

'use strict';

/**
 * Derive the safe integer bounds the scoring query needs from raw params.
 * Mirrors the original inline logic in db.ts getSimilarListings exactly.
 */
function deriveBounds(params) {
  const {
    postcodeDistrict,
    bedrooms,
    pricePcm,
    sizeSqft,
    propertyType,
    excludeId,
  } = params;

  // NOTE: `?? 0` (nullish), matching the original db.ts EXACTLY (Math.round never
  // returns null/undefined, so this is equivalent to plain Math.round here — kept
  // identical to production so the harness exercises the real derivation).
  const safeBedrooms = Math.max(0, Math.round(bedrooms) ?? 0);
  const safePricePcm = Math.max(1, Math.round(pricePcm) ?? 1);
  const safeSizeSqft = sizeSqft ? Math.round(sizeSqft) : 0;

  const minBedrooms = Math.max(0, safeBedrooms - 1);
  const maxBedrooms = safeBedrooms + 1;

  const minSqft = safeSizeSqft > 0 ? Math.floor(safeSizeSqft * 0.7) : 0;
  const maxSqft = safeSizeSqft > 0 ? Math.ceil(safeSizeSqft * 1.3) : 99999;
  const minSqftWide = Math.floor(minSqft * 0.8);
  const maxSqftWide = Math.ceil(maxSqft * 1.2);

  const priceRangeMin = Math.floor(safePricePcm * 0.5);
  const priceRangeMax = Math.ceil(safePricePcm * 2.0);
  const priceTolerance15 = Math.round(safePricePcm * 0.15);
  const priceTolerance30 = Math.round(safePricePcm * 0.30);

  const safePropertyType = propertyType ?? '';
  const safeExcludeId = excludeId ?? '__NO_MATCH__';

  return {
    postcodeDistrict,
    safeBedrooms,
    safePricePcm,
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
 * Build the parameterized similar-listings query (text + positional values array).
 * The SQL is byte-for-byte the same scoring logic as the original db.ts template; the
 * only change is explicit $N placeholders (which is what the template tag compiled to
 * anyway). Returns { text, values }.
 */
function buildSimilarQuery(params) {
  const b = deriveBounds(params);

  // Positional params, in the order they're referenced below.
  const values = [
    b.safeBedrooms, // $1
    b.safeSizeSqft, // $2
    b.minSqft, // $3
    b.maxSqft, // $4
    b.minSqftWide, // $5
    b.maxSqftWide, // $6
    b.safePricePcm, // $7
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
        price_pcm::int as price_pcm,
        size_sqft::int as size_sqft,
        bedrooms::int as bedrooms,
        property_type,
        last_seen::text as last_seen,
        CASE WHEN size_sqft > 0 THEN ROUND((price_pcm::numeric / size_sqft::numeric), 2)::float ELSE NULL END as ppsf,
        (
          CASE WHEN bedrooms = $1 THEN 0.30
               WHEN ABS(bedrooms - $1) = 1 THEN 0.15
               ELSE 0 END +
          CASE WHEN $2 > 0 AND size_sqft > 0 THEN
            CASE WHEN size_sqft BETWEEN $3 AND $4 THEN 0.25
                 WHEN size_sqft BETWEEN $5 AND $6 THEN 0.10
                 ELSE 0 END
          ELSE 0.15 END +
          CASE WHEN ABS(price_pcm - $7) <= $8 THEN 0.25
               WHEN ABS(price_pcm - $7) <= $9 THEN 0.15
               ELSE 0.05 END +
          CASE WHEN $10 != '' AND LOWER(property_type) = LOWER($10) THEN 0.10
               WHEN $10 = '' THEN 0.05
               ELSE 0 END +
          CASE WHEN source IN ('savills', 'knightfrank') THEN 0.10
               WHEN source IN ('chestertons', 'foxtons') THEN 0.07
               ELSE 0.05 END
        )::float as similarity_score
      FROM listings
      WHERE is_active = 1
        AND SPLIT_PART(postcode, ' ', 1) = $11
        AND bedrooms BETWEEN $12 AND $13
        AND price_pcm BETWEEN $14 AND $15
        AND price_pcm > 0
        AND property_id != $16
    )
    SELECT *
    FROM scored
    WHERE similarity_score > 0.3
    ORDER BY similarity_score DESC, ABS(price_pcm - $7) ASC
    LIMIT 15
  `;

  return { text, values };
}

/**
 * Compute the result stats from the returned rows. Mirrors db.ts exactly so the
 * harness asserts the SAME stats the route returns.
 */
function computeSimilarStats(rows, params) {
  const safePricePcm = Math.max(1, Math.round(params.pricePcm) || 1);

  const peerCount = rows.length;
  const avgPrice =
    peerCount > 0
      ? Math.round(rows.reduce((sum, r) => sum + r.price_pcm, 0) / peerCount)
      : 0;

  const ppsfValues = rows
    .map((r) => r.ppsf)
    .filter((v) => v !== null && v !== undefined);
  const avgPpsf =
    ppsfValues.length > 0
      ? Math.round(
          (ppsfValues.reduce((sum, v) => sum + v, 0) / ppsfValues.length) * 100
        ) / 100
      : null;

  const prices = rows.map((r) => r.price_pcm);
  const statsMinPrice = prices.length > 0 ? Math.min(...prices) : 0;
  const statsMaxPrice = prices.length > 0 ? Math.max(...prices) : 0;

  const belowCount = prices.filter((p) => p < safePricePcm).length;
  const yourPercentile =
    peerCount > 0 ? Math.round((belowCount / peerCount) * 100) : 50;

  return {
    peer_count: peerCount,
    avg_price: avgPrice,
    avg_ppsf: avgPpsf,
    min_price: statsMinPrice,
    max_price: statsMaxPrice,
    your_percentile: yourPercentile,
  };
}

module.exports = { deriveBounds, buildSimilarQuery, computeSimilarStats };
