import { sql } from '@vercel/postgres';
// Single-source query logic for /api/similar-sale — shared with the CI harness
// (dashboard/test/similar_sale_query_test.mjs) which runs the SAME query against a Postgres
// service container via `pg`. Keeping the bounds derivation + stats in saleSimilarQuery.js
// means the route and the test exercise identical logic (no drift). FORK of db.ts's
// getSimilarListings — the FOR-SALE analogue (sale_listings / asking_price).
import { computeSaleSimilarStats } from './saleSimilarQuery';

export interface SaleSimilarListing {
  id: number;
  source: string;
  property_id: string;
  address: string;
  postcode: string;
  url: string;
  asking_price: number;
  size_sqft: number | null;
  bedrooms: number;
  property_type: string | null;
  ppsf: number | null;
  similarity_score: number;
  last_seen: string | null; // ISO timestamp of when listing was last verified active
}

export interface SaleSimilarListingsResult {
  peers: SaleSimilarListing[];
  stats: {
    peer_count: number;
    avg_price: number | null;
    avg_ppsf: number | null;
    min_price: number | null;
    max_price: number | null;
    your_percentile: number;
  };
}

export interface SaleSimilarListingsParams {
  postcodeDistrict: string;
  bedrooms: number;
  askingPrice: number;
  sizeSqft?: number;
  propertyType?: string;
  excludeId?: string;
}

// Graceful-empty shape (AMENDMENT FIX 6): returned when sale_listings is empty OR does
// not exist (pre-Inc4b, before real sale rows are synced into prod Neon). This is what
// lets Inc4a ship green before Inc4b's production data exists.
const EMPTY_SALE_RESULT: SaleSimilarListingsResult = {
  peers: [],
  stats: {
    peer_count: 0,
    avg_price: null,
    avg_ppsf: null,
    min_price: null,
    max_price: null,
    your_percentile: 50,
  },
};

/**
 * Detect "relation sale_listings does not exist" (the table hasn't been created yet in
 * prod) so we can degrade to empty INSTEAD of throwing a 500. Postgres SQLSTATE 42P01 is
 * undefined_table; we also string-match "does not exist" / the table name as a fallback
 * across drivers. Any OTHER error is a GENUINE failure and MUST surface (re-thrown).
 */
function isMissingSaleTable(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) return false;
  const e = error as { code?: unknown; message?: unknown };
  if (e.code === '42P01') return true;
  const msg = typeof e.message === 'string' ? e.message.toLowerCase() : '';
  return msg.includes('does not exist') && msg.includes('sale_listings');
}

export async function getSimilarSaleListings(
  params: SaleSimilarListingsParams
): Promise<SaleSimilarListingsResult> {
  const { postcodeDistrict, bedrooms, askingPrice, sizeSqft, propertyType, excludeId } = params;

  // Pre-calculate all numeric values as integers (mirror getSimilarListings exactly).
  const safeBedrooms = Math.max(0, Math.round(bedrooms) ?? 0);
  const safeAskingPrice = Math.max(1, Math.round(askingPrice) ?? 1);
  const safeSizeSqft = sizeSqft ? Math.round(sizeSqft) : 0;

  // Bedroom range
  const minBedrooms = Math.max(0, safeBedrooms - 1);
  const maxBedrooms = safeBedrooms + 1;

  // Size ranges for similarity scoring (all as integers)
  const minSqft = safeSizeSqft > 0 ? Math.floor(safeSizeSqft * 0.7) : 0;
  const maxSqft = safeSizeSqft > 0 ? Math.ceil(safeSizeSqft * 1.3) : 99999;
  const minSqftWide = Math.floor(minSqft * 0.8);
  const maxSqftWide = Math.ceil(maxSqft * 1.2);

  // Asking-price ranges (sale band 0.5x–2x; all as integers)
  const priceRangeMin = Math.floor(safeAskingPrice * 0.5);
  const priceRangeMax = Math.ceil(safeAskingPrice * 2.0);
  const priceTolerance15 = Math.round(safeAskingPrice * 0.15);
  const priceTolerance30 = Math.round(safeAskingPrice * 0.30);

  // Safe property type (empty string if not provided)
  const safePropertyType = propertyType ?? '';

  // Safe excludeId - use impossible value if not provided to keep SQL simple
  const safeExcludeId = excludeId ?? '__NO_MATCH__';

  try {
    // Query for similar SALE listings with similarity scoring. The tagged-template SQL is
    // byte-equivalent to buildSaleSimilarQuery's text (single source of truth via the
    // structural-parity test). Drops SSTC / under-offer comps (is_under_offer exclusion).
    // Freshness gate is within 7 days of the data's own MAX(last_seen) (cycle-relative;
    // frozen-snapshot-safe, not wall-clock).
    const { rows } = await sql<SaleSimilarListing>`
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
          -- Similarity scoring (0-1 scale)
          (
            -- Bedroom match (30%)
            CASE WHEN bedrooms = ${safeBedrooms} THEN 0.30
                 WHEN ABS(bedrooms - ${safeBedrooms}) = 1 THEN 0.15
                 ELSE 0 END +
            -- Size match (25%) - only if we have sqft data
            CASE WHEN ${safeSizeSqft} > 0 AND size_sqft > 0 THEN
              CASE WHEN size_sqft BETWEEN ${minSqft} AND ${maxSqft} THEN 0.25
                   WHEN size_sqft BETWEEN ${minSqftWide} AND ${maxSqftWide} THEN 0.10
                   ELSE 0 END
            ELSE 0.15 END +
            -- Price match (25%)
            CASE WHEN ABS(asking_price - ${safeAskingPrice}) <= ${priceTolerance15} THEN 0.25
                 WHEN ABS(asking_price - ${safeAskingPrice}) <= ${priceTolerance30} THEN 0.15
                 ELSE 0.05 END +
            -- Property type match (10%)
            CASE WHEN ${safePropertyType} != '' AND LOWER(property_type) = LOWER(${safePropertyType}) THEN 0.10
                 WHEN ${safePropertyType} = '' THEN 0.05
                 ELSE 0 END +
            -- Source quality bonus (10%)
            CASE WHEN source IN ('savills', 'knightfrank') THEN 0.10
                 WHEN source IN ('chestertons', 'foxtons') THEN 0.07
                 ELSE 0.05 END
          )::float as similarity_score
        FROM sale_listings
        WHERE is_active = 1
          AND (
            last_seen IS NULL
            OR last_seen::timestamp >= (SELECT MAX(last_seen::timestamp) FROM sale_listings) - INTERVAL '7 days'
          )
          AND UPPER(
            COALESCE(
              SUBSTRING(REPLACE(postcode, ' ', '') FROM '^([A-Z]{1,2}[0-9][0-9A-Z]?)[0-9][A-Z]{2}$'),
              SUBSTRING(REPLACE(postcode, ' ', '') FROM '^([A-Z]{1,2}[0-9][0-9A-Z]?)$'),
              SPLIT_PART(postcode, ' ', 1)
            )
          ) = ${postcodeDistrict}
          AND bedrooms BETWEEN ${minBedrooms} AND ${maxBedrooms}
          AND asking_price BETWEEN ${priceRangeMin} AND ${priceRangeMax}
          AND asking_price > 0
          AND (is_under_offer IS NULL OR is_under_offer = 0)
          AND property_id != ${safeExcludeId}
      )
      SELECT *
      FROM scored
      WHERE similarity_score > 0.3
      ORDER BY similarity_score DESC, ABS(asking_price - ${safeAskingPrice}) ASC
      LIMIT 15
    `;

    // Single-source stats helper (shared with the CI harness). excludeId already filtered.
    const stats = computeSaleSimilarStats(rows, params);

    return {
      peers: rows,
      stats,
    };
  } catch (error) {
    // GRACEFUL-EMPTY (AMENDMENT FIX 6): a MISSING sale_listings table (pre-Inc4b) is NOT
    // an error — degrade to empty peers so /api/similar-sale returns 200 with no comps.
    if (isMissingSaleTable(error)) {
      return EMPTY_SALE_RESULT;
    }
    // A GENUINE error (connection failure, bad query, etc.) MUST surface so real failures
    // are not silently masked.
    throw error;
  }
}
