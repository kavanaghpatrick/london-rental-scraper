// Type declarations for the Next-free similarQuery.js single-source query module.
// db.ts imports the pure derivation + stats from here; the CI harness imports the
// same .js to run the real query against a Postgres service container.

export interface SimilarQueryParams {
  postcodeDistrict: string;
  bedrooms: number;
  pricePcm: number;
  sizeSqft?: number;
  propertyType?: string;
  excludeId?: string;
}

export interface SimilarQueryRow {
  price_pcm: number;
  ppsf: number | null;
}

export interface SimilarQueryStats {
  peer_count: number;
  avg_price: number;
  avg_ppsf: number | null;
  min_price: number;
  max_price: number;
  your_percentile: number;
}

export function deriveBounds(params: SimilarQueryParams): Record<string, number | string>;

export function buildSimilarQuery(params: SimilarQueryParams): {
  text: string;
  values: Array<number | string>;
};

export function computeSimilarStats(
  rows: SimilarQueryRow[],
  params: SimilarQueryParams
): SimilarQueryStats;
