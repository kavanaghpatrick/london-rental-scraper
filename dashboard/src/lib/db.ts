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

// ============ Property Valuations ============

export interface PropertyValuation {
  id: number;
  address: string;
  postcode: string;
  size_sqft: number;
  bedrooms: number;
  bathrooms: number;
  predicted_pcm: number;
  range_low: number;
  range_high: number;
  model_version: string;
  model_r2: number;
  model_mape: number;
  created_at: string;
}

export async function getLatestValuation(address: string): Promise<PropertyValuation | null> {
  const { rows } = await sql<PropertyValuation>`
    SELECT
      id, address, postcode, size_sqft, bedrooms, bathrooms,
      predicted_pcm, range_low, range_high,
      model_version, model_r2, model_mape,
      created_at::text as created_at
    FROM property_valuations
    WHERE address = ${address}
    ORDER BY created_at DESC
    LIMIT 1
  `;
  return rows[0] || null;
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
  bathrooms: number;
  url: string;
  ppsf: number;
  ppsf_diff: number;  // Difference from subject £/sqft
  size_diff_pct: number;  // Size difference as percentage
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
  sizeRange: number = 0.40,
  subjectPpsf: number,
  postcodeArea?: string  // e.g., 'SW1' to filter to SW1A, SW1W, SW1X etc. (NOT SW10, SW11)
): Promise<Comparable[]> {
  const minSize = Math.floor(sizeSqft * (1 - sizeRange));
  const maxSize = Math.ceil(sizeSqft * (1 + sizeRange));
  const minBedrooms = Math.max(0, bedrooms - 1);
  const maxBedrooms = bedrooms + 1;

  // UNBIASED COMP SELECTION - no cherry-picking
  // Returns ALL properties matching criteria, sorted by £/sqft (neutral ordering)
  // Dedupe by price+size+district to remove cross-source duplicates
  //
  // PRIME CENTRAL LONDON: SW1, SW3, SW7, W1
  // - SW1: Belgravia, Westminster (subject property area)
  // - SW3: Chelsea (equally prestigious)
  // - SW7: South Kensington (equally prestigious, museum district)
  // - W1: Mayfair, Marylebone (arguably MORE prestigious)
  //
  // EXCLUDED:
  // - W8 (Kensington): Different market, outliers near Kensington Palace skew data
  // - NW1/NW3/NW8: Nice but tier below Belgravia/Chelsea
  // - SW10/SW11: NOT SW1 - different areas (Chelsea Harbour, Battersea)
  //
  // IMPORTANT: Only include long-term lets (price_period = 'pcm')
  // Short lets (price_period = 'pw') have 2x £/sqft and aren't comparable
  const { rows } = await sql<Comparable>`
    WITH ranked AS (
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
        COALESCE(bathrooms, 1)::int as bathrooms,
        url,
        ROUND((price_pcm::numeric / size_sqft::numeric), 2)::float as ppsf,
        ROUND(ABS((price_pcm::numeric / size_sqft::numeric) - ${subjectPpsf}), 2)::float as ppsf_diff,
        ROUND((size_sqft::numeric - ${sizeSqft}) / ${sizeSqft} * 100, 0)::int as size_diff_pct,
        -- Dedupe: keep agent source over aggregator (Rightmove)
        ROW_NUMBER() OVER (
          PARTITION BY price_pcm, size_sqft,
            CASE WHEN POSITION(' ' IN postcode) > 0 THEN SUBSTRING(postcode, 1, POSITION(' ' IN postcode) - 1) ELSE postcode END
          ORDER BY CASE WHEN source = 'rightmove' THEN 1 ELSE 0 END, source
        ) as rn
      FROM listings
      WHERE is_active = 1
        AND size_sqft IS NOT NULL
        AND size_sqft > 0
        AND price_pcm IS NOT NULL
        AND price_pcm > 0
        AND size_sqft BETWEEN ${minSize} AND ${maxSize}
        AND (
          postcode ~ '^SW1[A-Z]'  -- Belgravia/Westminster (NOT SW10/11/12)
          OR postcode ~ '^SW3'     -- Chelsea
          OR postcode ~ '^SW7'     -- South Kensington
          OR postcode ~ '^W1[A-Z]' -- Mayfair/Marylebone
        )
        AND (price_period = 'pcm' OR price_period IS NULL OR price_period = '')
        AND (description NOT ILIKE '%short let%' AND description NOT ILIKE '%short-let%' OR description IS NULL)
        AND COALESCE(is_short_let, 0) = 0  -- Exclude flagged short lets
        AND (price_pcm::numeric / size_sqft::numeric) >= 3  -- Filter obvious bad data (< £3/sqft impossible in Prime Central)
        AND (price_pcm::numeric / size_sqft::numeric) <= 30  -- Filter extreme outliers (> £30/sqft likely data errors)
        AND bedrooms BETWEEN ${minBedrooms} AND ${maxBedrooms}  -- Match similar property types
    )
    SELECT address, postcode, district, source, price_pcm, size_sqft, bedrooms, bathrooms, url, ppsf, ppsf_diff, size_diff_pct
    FROM ranked
    WHERE rn = 1
    ORDER BY ppsf ASC
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

// ============ Agent Performance Functions ============

export interface AgentPerformance {
  source: string;
  active_listings: number;
  price_reductions: number;
  avg_reduction_pct: number;
  price_cut_rate: number;
  turnover_rate: number;
}

export interface AgentTrend {
  date: string;
  source: string;
  inventory: number;
  cumulative_reductions: number;
  cumulative_new: number;
}

export async function getAgentPerformance(): Promise<AgentPerformance[]> {
  const { rows } = await sql<AgentPerformance>`
    WITH price_changes AS (
      SELECT
        ph.listing_id,
        ph.price_pcm as current_price,
        LAG(ph.price_pcm) OVER (PARTITION BY ph.listing_id ORDER BY ph.recorded_at) as prev_price,
        ph.recorded_at
      FROM price_history ph
      WHERE ph.recorded_at::timestamp >= NOW() - INTERVAL '30 days'
    ),
    reductions AS (
      SELECT
        l.source,
        COUNT(DISTINCT pc.listing_id)::int as price_reductions,
        COALESCE(ROUND(AVG((pc.current_price - pc.prev_price)::numeric / pc.prev_price::numeric * 100)::numeric, 2), 0)::float as avg_reduction_pct
      FROM price_changes pc
      JOIN listings l ON l.id = pc.listing_id
      WHERE pc.prev_price IS NOT NULL
        AND pc.current_price < pc.prev_price
        AND l.source != 'rightmove'
      GROUP BY l.source
    ),
    inventory AS (
      SELECT
        source,
        COUNT(*)::int as active_listings
      FROM listings
      WHERE is_active = 1
        AND source != 'rightmove'
      GROUP BY source
    ),
    turnover AS (
      SELECT
        source,
        COUNT(*)::int as turned_over
      FROM listings
      WHERE source != 'rightmove'
        AND (
          (is_active = 0 AND last_seen::timestamp >= NOW() - INTERVAL '30 days')
          OR first_seen::timestamp >= NOW() - INTERVAL '30 days'
        )
      GROUP BY source
    )
    SELECT
      i.source,
      i.active_listings,
      COALESCE(r.price_reductions, 0)::int as price_reductions,
      COALESCE(r.avg_reduction_pct, 0)::float as avg_reduction_pct,
      CASE WHEN i.active_listings > 0
        THEN ROUND((COALESCE(r.price_reductions, 0) * 100.0 / i.active_listings)::numeric, 1)::float
        ELSE 0
      END as price_cut_rate,
      CASE WHEN i.active_listings > 0
        THEN ROUND((COALESCE(t.turned_over, 0) * 100.0 / i.active_listings)::numeric, 1)::float
        ELSE 0
      END as turnover_rate
    FROM inventory i
    LEFT JOIN reductions r ON r.source = i.source
    LEFT JOIN turnover t ON t.source = i.source
    ORDER BY i.active_listings DESC
  `;
  return rows;
}

export async function getAgentTrends(days: number = 30): Promise<AgentTrend[]> {
  const { rows } = await sql<AgentTrend>`
    WITH dates AS (
      SELECT generate_series(
        DATE_TRUNC('day', NOW() - INTERVAL '30 days'),
        DATE_TRUNC('day', NOW()),
        '1 day'::interval
      )::date as date
    ),
    sources AS (
      SELECT DISTINCT source FROM listings WHERE source != 'rightmove'
    ),
    date_source AS (
      SELECT d.date, s.source FROM dates d CROSS JOIN sources s
    ),
    daily_inventory AS (
      SELECT
        ds.date,
        ds.source,
        COUNT(DISTINCT l.id)::int as inventory
      FROM date_source ds
      LEFT JOIN listings l ON l.source = ds.source
        AND l.first_seen::date <= ds.date
        AND (l.is_active = 1 OR l.last_seen::date >= ds.date)
      GROUP BY ds.date, ds.source
    ),
    price_changes AS (
      SELECT
        ph.listing_id,
        ph.price_pcm as current_price,
        LAG(ph.price_pcm) OVER (PARTITION BY ph.listing_id ORDER BY ph.recorded_at) as prev_price,
        ph.recorded_at
      FROM price_history ph
    ),
    cumulative_reductions AS (
      SELECT
        ds.date,
        ds.source,
        COUNT(DISTINCT pc.listing_id)::int as cumulative_reductions
      FROM date_source ds
      LEFT JOIN listings l ON l.source = ds.source
      LEFT JOIN price_changes pc ON pc.listing_id = l.id
        AND pc.prev_price IS NOT NULL
        AND pc.current_price < pc.prev_price
        AND pc.recorded_at::date <= ds.date
        AND pc.recorded_at::timestamp >= NOW() - INTERVAL '30 days'
      GROUP BY ds.date, ds.source
    ),
    cumulative_new AS (
      SELECT
        ds.date,
        ds.source,
        COUNT(DISTINCT l.id)::int as cumulative_new
      FROM date_source ds
      LEFT JOIN listings l ON l.source = ds.source
        AND l.first_seen::date <= ds.date
        AND l.first_seen::timestamp >= NOW() - INTERVAL '30 days'
      GROUP BY ds.date, ds.source
    )
    SELECT
      di.date::text,
      di.source,
      di.inventory,
      COALESCE(cr.cumulative_reductions, 0)::int as cumulative_reductions,
      COALESCE(cn.cumulative_new, 0)::int as cumulative_new
    FROM daily_inventory di
    LEFT JOIN cumulative_reductions cr ON cr.date = di.date AND cr.source = di.source
    LEFT JOIN cumulative_new cn ON cn.date = di.date AND cn.source = di.source
    ORDER BY di.date ASC, di.source
  `;
  return rows;
}

export interface AgentLeaderboard {
  rank: number;
  source: string;
  score: number;
  active_listings: number;
  price_reductions: number;
  avg_change_pct: number;
  turnover_pct: number;
}

export async function getAgentLeaderboard(): Promise<AgentLeaderboard[]> {
  const { rows } = await sql<AgentLeaderboard>`
    WITH price_changes AS (
      SELECT
        ph.listing_id,
        ph.price_pcm as current_price,
        LAG(ph.price_pcm) OVER (PARTITION BY ph.listing_id ORDER BY ph.recorded_at) as prev_price,
        ph.recorded_at
      FROM price_history ph
    ),
    -- Get the FIRST price reduction per listing (avoids counting oscillations multiple times)
    first_reduction_per_listing AS (
      SELECT DISTINCT ON (pc.listing_id)
        pc.listing_id,
        (pc.current_price - pc.prev_price)::numeric / pc.prev_price::numeric * 100 as reduction_pct
      FROM price_changes pc
      WHERE pc.prev_price IS NOT NULL
        AND pc.current_price < pc.prev_price
        AND pc.recorded_at::timestamp >= NOW() - INTERVAL '30 days'
      ORDER BY pc.listing_id, pc.recorded_at ASC
    ),
    metrics AS (
      SELECT
        l.source,
        COUNT(DISTINCT CASE WHEN l.is_active = 1 THEN l.id END)::int as active_listings,
        COUNT(DISTINCT fr.listing_id)::int as price_reductions,
        -- Average per-listing, not per-event (fixes oscillation bug)
        COALESCE(ROUND(AVG(fr.reduction_pct)::numeric, 2), 0)::float as avg_change_pct,
        COUNT(DISTINCT CASE WHEN l.is_active = 0 AND l.last_seen::timestamp >= NOW() - INTERVAL '30 days' THEN l.id END)::int as turned_over
      FROM listings l
      LEFT JOIN first_reduction_per_listing fr ON fr.listing_id = l.id
      WHERE l.source != 'rightmove'
      GROUP BY l.source
    )
    SELECT
      ROW_NUMBER() OVER (ORDER BY
        -- Score: fewer reductions = better, smaller avg cut = better, high turnover = good
        (100 - LEAST(price_reductions, 100)) +
        (100 - LEAST(ABS(avg_change_pct)::numeric * 20, 100)) +
        LEAST((turned_over * 100.0 / NULLIF(active_listings, 0))::numeric, 50)
        DESC
      )::int as rank,
      source,
      ROUND((
        (100 - LEAST(price_reductions, 100)) +
        (100 - LEAST(ABS(avg_change_pct)::numeric * 20, 100)) +
        LEAST((turned_over * 100.0 / NULLIF(active_listings, 0))::numeric, 50)
      )::numeric, 1)::float as score,
      active_listings,
      price_reductions,
      avg_change_pct,
      ROUND((turned_over * 100.0 / NULLIF(active_listings, 0))::numeric, 1)::float as turnover_pct
    FROM metrics
    WHERE active_listings > 0  -- Filter out inactive agents (e.g., johndwood)
    ORDER BY rank
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
