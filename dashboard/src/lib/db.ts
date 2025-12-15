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
