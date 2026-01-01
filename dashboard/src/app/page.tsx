import { getRecentRuns, getListingStats, getHealthStatus, getDailyStats, getModelRuns } from '@/lib/db';
import RunningSpiders from './components/RunningSpiders';

function formatDate(dateValue: string | Date | null | undefined): string {
  if (!dateValue) return 'N/A';
  // Handle both Date objects and strings from @vercel/postgres
  const date = dateValue instanceof Date ? dateValue : new Date(dateValue);
  if (isNaN(date.getTime())) return 'N/A';
  return date.toLocaleString('en-GB', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function formatDuration(seconds: number | null): string {
  if (!seconds) return '-';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.round(seconds / 60)}m`;
}

function StatusBadge({ status }: { status: string }) {
  const colors = {
    completed: 'bg-green-100 text-green-800',
    running: 'bg-blue-100 text-blue-800',
    failed: 'bg-red-100 text-red-800',
  };
  const color = colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${color}`}>
      {status}
    </span>
  );
}

function HealthCard({ title, value, subtitle, status }: { 
  title: string; 
  value: string | number; 
  subtitle?: string;
  status?: 'good' | 'warning' | 'error';
}) {
  const borderColors = {
    good: 'border-l-green-500',
    warning: 'border-l-yellow-500',
    error: 'border-l-red-500',
  };
  const borderColor = status ? borderColors[status] : 'border-l-gray-300';
  
  return (
    <div className={`bg-white rounded-lg shadow p-4 border-l-4 ${borderColor}`}>
      <div className="text-sm text-gray-500">{title}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
      {subtitle && <div className="text-xs text-gray-400 mt-1">{subtitle}</div>}
    </div>
  );
}

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export default async function Dashboard() {
  let health;
  let runs;
  let stats;
  let dailyStats;
  let modelRuns;
  let error = null;

  try {
    [health, runs, stats, dailyStats, modelRuns] = await Promise.all([
      getHealthStatus(),
      getRecentRuns(20),
      getListingStats(),
      getDailyStats(7),
      getModelRuns(10),
    ]);
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to connect to database';
  }

  if (error) {
    return (
      <main className="min-h-screen p-8">
        <h1 className="text-3xl font-bold mb-8">Rental Scraper Dashboard</h1>
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h2 className="text-red-800 font-semibold">Database Connection Error</h2>
          <p className="text-red-600 mt-2">{error}</p>
          <p className="text-sm text-gray-600 mt-4">
            Make sure POSTGRES_URL is set in your environment variables.
          </p>
        </div>
      </main>
    );
  }

  const hoursSinceLastRun = health?.lastRun 
    ? (Date.now() - new Date(health.lastRun).getTime()) / (1000 * 60 * 60)
    : Infinity;
  const overallStatus = health?.lastStatus === 'failed' ? 'error' 
    : hoursSinceLastRun > 26 ? 'warning' 
    : 'good';

  return (
    <main className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Rental Scraper Dashboard</h1>
          <div className="flex items-center gap-4">
            <a
              href="/agencies"
              className="bg-green-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-green-700 transition"
            >
              Agency Leaderboard
            </a>
            <a
              href="/agents"
              className="bg-purple-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-purple-700 transition"
            >
              Agent Performance
            </a>
            <a
              href="/report"
              className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700 transition"
            >
              Property Valuation Report
            </a>
            <span className="text-sm text-gray-500">
              Auto-refreshes on each visit
            </span>
          </div>
        </div>

        <RunningSpiders />

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <HealthCard
            title="Last Scrape"
            value={formatDate(health?.lastRun || null)}
            subtitle={`Status: ${health?.lastStatus || 'unknown'}`}
            status={overallStatus}
          />
          <HealthCard 
            title="Total Listings" 
            value={health?.totalListings?.toLocaleString() || 0}
            status="good"
          />
          <HealthCard 
            title="Items (24h)" 
            value={health?.itemsLast24h?.toLocaleString() || 0}
            status={health?.itemsLast24h && health.itemsLast24h > 0 ? 'good' : 'warning'}
          />
          <HealthCard 
            title="Errors (24h)" 
            value={health?.errorsLast24h || 0}
            status={health?.errorsLast24h && health.errorsLast24h > 10 ? 'error' : 'good'}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 bg-white rounded-lg shadow">
            <div className="p-4 border-b">
              <h2 className="text-lg font-semibold">Recent Scrape Runs</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left">Run ID</th>
                    <th className="px-4 py-2 text-left">Spider</th>
                    <th className="px-4 py-2 text-left">Started</th>
                    <th className="px-4 py-2 text-right">Duration</th>
                    <th className="px-4 py-2 text-right">Items</th>
                    <th className="px-4 py-2 text-right">Errors</th>
                    <th className="px-4 py-2 text-center">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {runs?.map((run) => (
                    <tr key={`${run.run_id}-${run.spider_name}`} className="hover:bg-gray-50">
                      <td className="px-4 py-2 font-mono text-xs">{run.run_id}</td>
                      <td className="px-4 py-2">{run.spider_name}</td>
                      <td className="px-4 py-2">{formatDate(run.started_at)}</td>
                      <td className="px-4 py-2 text-right">{formatDuration(run.duration_seconds)}</td>
                      <td className="px-4 py-2 text-right">{run.items_scraped}</td>
                      <td className="px-4 py-2 text-right">
                        {run.error_count > 0 ? (
                          <span className="text-red-600">{run.error_count}</span>
                        ) : (
                          run.error_count
                        )}
                      </td>
                      <td className="px-4 py-2 text-center">
                        <StatusBadge status={run.status} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow">
            <div className="p-4 border-b">
              <h2 className="text-lg font-semibold">Listings by Source</h2>
            </div>
            <div className="p-4">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500">
                    <th className="text-left pb-2">Source</th>
                    <th className="text-right pb-2">Total</th>
                    <th className="text-right pb-2">Active</th>
                    <th className="text-right pb-2">w/ sqft</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {stats?.map((stat) => (
                    <tr key={stat.source}>
                      <td className="py-2 font-medium">{stat.source}</td>
                      <td className="py-2 text-right">{stat.total?.toLocaleString()}</td>
                      <td className="py-2 text-right">{stat.active?.toLocaleString()}</td>
                      <td className="py-2 text-right">{stat.with_sqft?.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="mt-8 bg-white rounded-lg shadow">
          <div className="p-4 border-b">
            <h2 className="text-lg font-semibold">Daily Scrape History (7 days)</h2>
          </div>
          <div className="p-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500">
                  <th className="text-left pb-2">Date</th>
                  <th className="text-right pb-2">Items Scraped</th>
                  <th className="text-right pb-2">Errors</th>
                  <th className="text-right pb-2">Duration</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {dailyStats?.map((day) => (
                  <tr key={day.date}>
                    <td className="py-2">{day.date}</td>
                    <td className="py-2 text-right">{day.items_scraped?.toLocaleString()}</td>
                    <td className="py-2 text-right">
                      {day.errors > 10 ? (
                        <span className="text-red-600">{day.errors}</span>
                      ) : (
                        day.errors
                      )}
                    </td>
                    <td className="py-2 text-right">{day.duration_minutes}m</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {modelRuns && modelRuns.length > 0 && (
          <div className="mt-8 bg-white rounded-lg shadow">
            <div className="p-4 border-b flex justify-between items-center">
              <h2 className="text-lg font-semibold">Model Performance History</h2>
              <span className="text-xs text-gray-500">XGBoost Price Prediction Model</span>
            </div>
            <div className="p-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-blue-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-blue-700">
                    {(modelRuns[0]?.r2_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500 mt-1">Latest R² Score</div>
                </div>
                <div className="bg-green-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-green-700">
                    {modelRuns[0]?.mape?.toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500 mt-1">MAPE (Mean Abs % Error)</div>
                </div>
                <div className="bg-purple-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-purple-700">
                    {modelRuns[0]?.median_ape?.toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500 mt-1">Median APE</div>
                </div>
                <div className="bg-orange-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-orange-700">
                    {modelRuns[0]?.samples_total?.toLocaleString()}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">Training Samples</div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-500 border-b">
                      <th className="text-left pb-2">Date</th>
                      <th className="text-right pb-2">R²</th>
                      <th className="text-right pb-2">MAE</th>
                      <th className="text-right pb-2">MAPE</th>
                      <th className="text-right pb-2">Median APE</th>
                      <th className="text-right pb-2">Samples</th>
                      <th className="text-right pb-2">Trend</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {modelRuns.map((run, idx) => {
                      const prevRun = modelRuns[idx + 1];
                      const r2Change = prevRun ? run.r2_score - prevRun.r2_score : 0;
                      return (
                        <tr key={run.run_id} className={idx === 0 ? 'bg-blue-50/50' : ''}>
                          <td className="py-2">{run.run_date}</td>
                          <td className="py-2 text-right font-medium">{(run.r2_score * 100).toFixed(1)}%</td>
                          <td className="py-2 text-right">£{run.mae?.toLocaleString()}</td>
                          <td className="py-2 text-right">{run.mape?.toFixed(1)}%</td>
                          <td className="py-2 text-right">{run.median_ape?.toFixed(1)}%</td>
                          <td className="py-2 text-right">{run.samples_total?.toLocaleString()}</td>
                          <td className="py-2 text-right">
                            {r2Change > 0.001 ? (
                              <span className="text-green-600">+{(r2Change * 100).toFixed(2)}%</span>
                            ) : r2Change < -0.001 ? (
                              <span className="text-red-600">{(r2Change * 100).toFixed(2)}%</span>
                            ) : (
                              <span className="text-gray-400">-</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
