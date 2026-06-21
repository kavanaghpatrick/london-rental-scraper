// Type declarations for the Next-free saleSimilarQuery.js single-source query module.
// saleDb.ts imports the pure derivation + stats from here; the CI harness imports the
// same .js to run the real query against a Postgres service container.

export interface SaleSimilarQueryParams {
  postcodeDistrict: string;
  bedrooms: number;
  askingPrice: number;
  sizeSqft?: number;
  propertyType?: string;
  excludeId?: string;
}

export interface SaleSimilarQueryRow {
  asking_price: number;
  ppsf: number | null;
}

export interface SaleSimilarQueryStats {
  peer_count: number;
  avg_price: number | null;
  avg_ppsf: number | null;
  min_price: number | null;
  max_price: number | null;
  your_percentile: number;
}

export const SALE_PPSF_MIN: number;
export const SALE_PPSF_MAX: number;

export function deriveSaleBounds(
  params: SaleSimilarQueryParams
): Record<string, number | string>;

export function buildSaleSimilarQuery(params: SaleSimilarQueryParams): {
  text: string;
  values: Array<number | string>;
};

export function computeSaleSimilarStats(
  rows: SaleSimilarQueryRow[],
  params: SaleSimilarQueryParams
): SaleSimilarQueryStats;
