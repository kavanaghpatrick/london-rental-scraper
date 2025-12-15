import { sql } from '@vercel/postgres';

export interface ScrapeRun {
  id: number;
  run_id: string;
  spider_name: string;
  started_at: string;
  finished_at: string | null;
  duration_seconds: number | null;
  status: string;
  items_scraped: number;
  items_new: number;
  items_updated: number;
  items_dropped: number;
  items_errors: number;
  request_count: number;
  response_count: number;
  error_count: number;
  retry_count: number;
  memory_start_mb: number | null;
  memory_peak_mb: number | null;
  memory_end_mb: number | null;
  exit_reason: string | null;
  error_summary: string | null;
}

export interface ListingStats {
  source: string;
  total: number;
  active: number;
  with_sqft: number;
}

export interface DailyStats {
  date: string;
  items_scraped: number;
  errors: number;
  duration_minutes: number;
}

export async function getRecentRuns(limit: number = 20): Promise<ScrapeRun[]> {
  const { rows } = await sql<ScrapeRun>`
    SELECT
      id, run_id, spider_name,
      started_at::text as started_at,
      finished_at::text as finished_at,
      duration_seconds, status, items_scraped, items_new, items_updated,
      items_dropped, items_errors, request_count, response_count,
      error_count, retry_count, memory_start_mb, memory_peak_mb,
      memory_end_mb, exit_reason, error_summary
    FROM scrape_runs
    ORDER BY started_at DESC
    LIMIT ${limit}
  `;
  return rows;
}

export async function getRunsByRunId(runId: string): Promise<ScrapeRun[]> {
  const { rows } = await sql<ScrapeRun>`
    SELECT * FROM scrape_runs 
    WHERE run_id = ${runId}
    ORDER BY started_at
  `;
  return rows;
}

export async function getListingStats(): Promise<ListingStats[]> {
  const { rows } = await sql<ListingStats>`
    SELECT 
      source,
      COUNT(*)::int as total,
      SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END)::int as active,
      SUM(CASE WHEN size_sqft > 0 THEN 1 ELSE 0 END)::int as with_sqft
    FROM listings
    GROUP BY source
    ORDER BY total DESC
  `;
  return rows;
}

export async function getDailyStats(days: number = 14): Promise<DailyStats[]> {
  const { rows } = await sql<DailyStats>`
    SELECT
      DATE(started_at::timestamp)::text as date,
      SUM(items_scraped)::int as items_scraped,
      SUM(error_count)::int as errors,
      COALESCE(ROUND(SUM(duration_seconds) / 60), 0)::int as duration_minutes
    FROM scrape_runs
    WHERE started_at::timestamp >= NOW() - INTERVAL '14 days'
    GROUP BY DATE(started_at::timestamp)
    ORDER BY date DESC
  `;
  return rows;
}

export interface RunningSpider {
  run_id: string;
  spider_name: string;
  started_at: string;
  items_scraped: number;
  error_count: number;
  elapsed_minutes: number;
}

export async function getRunningSpiders(): Promise<RunningSpider[]> {
  const { rows } = await sql<RunningSpider>`
    SELECT
      run_id,
      spider_name,
      started_at::text as started_at,
      items_scraped,
      error_count,
      ROUND(EXTRACT(EPOCH FROM (NOW() - started_at::timestamp)) / 60)::int as elapsed_minutes
    FROM scrape_runs
    WHERE status = 'running'
    ORDER BY started_at ASC
  `;
  return rows;
}

export interface ModelRun {
  id: number;
  run_date: string;
  run_id: string;
  version: string;
  samples_total: number;
  features_count: number;
  r2_score: number;
  mae: number;
  mape: number;
  median_ape: number;
  training_time_seconds: number;
}

export async function getModelRuns(limit: number = 20): Promise<ModelRun[]> {
  const { rows } = await sql<ModelRun>`
    SELECT
      id, run_date, run_id, version, samples_total, features_count,
      r2_score, mae, mape, median_ape, training_time_seconds
    FROM model_runs
    ORDER BY created_at DESC
    LIMIT ${limit}
  `;
  return rows;
}

// ============ Valuation Report Functions ============

export interface Comparable {
  address: string;
  postcode: string;
  district: string;
  source: string;
  price_pcm: number;
  size_sqft: number;
  bedrooms: number;
  url: string;
  ppsf: number;
}

export interface MarketStats {
  total_listings: number;
  median_ppsf: number;
  p25_ppsf: number;
  p75_ppsf: number;
  avg_ppsf: number;
}

export interface PpsfDistribution {
  bucket: number;
  count: number;
}

export async function getComparables(
  sizeSqft: number,
  bedrooms: number,
  sizeRange: number = 0.20
): Promise<Comparable[]> {
  const minSize = Math.floor(sizeSqft * (1 - sizeRange));
  const maxSize = Math.ceil(sizeSqft * (1 + sizeRange));

  const { rows } = await sql<Comparable>`
    SELECT
      address,
      postcode,
      CASE
        WHEN POSITION(' ' IN postcode) > 0 THEN SUBSTRING(postcode, 1, POSITION(' ' IN postcode) - 1)
        ELSE postcode
      END as district,
      source,
      price_pcm::int as price_pcm,
      size_sqft::int as size_sqft,
      bedrooms::int as bedrooms,
      url,
      ROUND((price_pcm::numeric / size_sqft::numeric), 2)::float as ppsf
    FROM listings
    WHERE is_active = 1
      AND size_sqft IS NOT NULL
      AND size_sqft > 0
      AND price_pcm IS NOT NULL
      AND price_pcm > 0
      AND size_sqft BETWEEN ${minSize} AND ${maxSize}
      AND bedrooms = ${bedrooms}
    ORDER BY price_pcm DESC
    LIMIT 200
  `;
  return rows;
}

export async function getMarketStats(): Promise<MarketStats> {
  const { rows } = await sql<MarketStats>`
    SELECT
      COUNT(*)::int as total_listings,
      ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_pcm::numeric / size_sqft::numeric))::numeric, 2)::float as median_ppsf,
      ROUND((PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price_pcm::numeric / size_sqft::numeric))::numeric, 2)::float as p25_ppsf,
      ROUND((PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price_pcm::numeric / size_sqft::numeric))::numeric, 2)::float as p75_ppsf,
      ROUND((AVG(price_pcm::numeric / size_sqft::numeric))::numeric, 2)::float as avg_ppsf
    FROM listings
    WHERE is_active = 1
      AND size_sqft IS NOT NULL AND size_sqft > 100
      AND price_pcm IS NOT NULL AND price_pcm > 500
  `;
  return rows[0];
}

export async function getPpsfDistribution(): Promise<PpsfDistribution[]> {
  const { rows } = await sql<PpsfDistribution>`
    SELECT
      FLOOR(price_pcm::numeric / size_sqft::numeric)::int as bucket,
      COUNT(*)::int as count
    FROM listings
    WHERE is_active = 1
      AND size_sqft IS NOT NULL AND size_sqft > 100
      AND price_pcm IS NOT NULL AND price_pcm > 500
      AND (price_pcm::numeric / size_sqft::numeric) BETWEEN 1 AND 20
    GROUP BY bucket
    ORDER BY bucket
  `;
  return rows;
}

export async function getPpsfByDistrict(): Promise<{ district: string; median_ppsf: number; count: number }[]> {
  const { rows } = await sql<{ district: string; median_ppsf: number; count: number }>`
    WITH district_data AS (
      SELECT
        CASE
          WHEN POSITION(' ' IN postcode) > 0 THEN SUBSTRING(postcode, 1, POSITION(' ' IN postcode) - 1)
          ELSE postcode
        END as district,
        price_pcm::numeric / size_sqft::numeric as ppsf
      FROM listings
      WHERE is_active = 1
        AND size_sqft IS NOT NULL AND size_sqft > 100
        AND price_pcm IS NOT NULL AND price_pcm > 500
    )
    SELECT
      district,
      ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ppsf))::numeric, 2)::float as median_ppsf,
      COUNT(*)::int as count
    FROM district_data
    WHERE district IS NOT NULL AND district != ''
    GROUP BY district
    HAVING COUNT(*) >= 5
    ORDER BY median_ppsf DESC
    LIMIT 15
  `;
  return rows;
}

export async function getHealthStatus(): Promise<{
  lastRun: string | null;
  lastStatus: string;
  totalListings: number;
  itemsLast24h: number;
  errorsLast24h: number;
}> {
  // Get last run info (cast to text to ensure string type)
  const lastRunResult = await sql`
    SELECT run_id, started_at::text as started_at, status
    FROM scrape_runs
    ORDER BY started_at DESC
    LIMIT 1
  `;
  
  // Get total listings
  const totalResult = await sql`SELECT COUNT(*)::int as count FROM listings`;
  
  // Get 24h stats
  const recentResult = await sql`
    SELECT
      COALESCE(SUM(items_scraped), 0)::int as items,
      COALESCE(SUM(error_count), 0)::int as errors
    FROM scrape_runs
    WHERE started_at::timestamp >= NOW() - INTERVAL '24 hours'
  `;
  
  return {
    lastRun: lastRunResult.rows[0]?.started_at || null,
    lastStatus: lastRunResult.rows[0]?.status || 'unknown',
    totalListings: totalResult.rows[0]?.count || 0,
    itemsLast24h: recentResult.rows[0]?.items || 0,
    errorsLast24h: recentResult.rows[0]?.errors || 0,
  };
}
