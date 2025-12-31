import { getAgentPerformance, getAgentLeaderboard, getAgentTrends } from '@/lib/db';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

const AGENT_COLORS: Record<string, string> = {
  savills: '#4CAF50',
  knightfrank: '#2196F3',
  chestertons: '#E57373',
  foxtons: '#9C27B0',
};

const AGENT_DISPLAY_NAMES: Record<string, string> = {
  savills: 'Savills',
  knightfrank: 'Knight Frank',
  chestertons: 'Chestertons',
  foxtons: 'Foxtons',
};

function formatPercent(value: number | null): string {
  if (value === null || value === undefined) return '-';
  return `${value.toFixed(1)}%`;
}

function MetricCard({
  title,
  value,
  subtitle,
  trend,
  color = 'blue'
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  color?: 'blue' | 'green' | 'red' | 'purple' | 'orange';
}) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-700',
    green: 'bg-green-50 text-green-700',
    red: 'bg-red-50 text-red-700',
    purple: 'bg-purple-50 text-purple-700',
    orange: 'bg-orange-50 text-orange-700',
  };

  return (
    <div className={`rounded-lg p-4 ${colorClasses[color]}`}>
      <div className="text-sm opacity-80">{title}</div>
      <div className="text-2xl font-bold mt-1 flex items-center gap-2">
        {value}
        {trend === 'up' && <span className="text-green-600 text-sm">^</span>}
        {trend === 'down' && <span className="text-red-600 text-sm">v</span>}
      </div>
      {subtitle && <div className="text-xs opacity-60 mt-1">{subtitle}</div>}
    </div>
  );
}

function RankBadge({ rank }: { rank: number }) {
  const colors = {
    1: 'bg-yellow-100 text-yellow-800 border-yellow-300',
    2: 'bg-gray-100 text-gray-700 border-gray-300',
    3: 'bg-orange-100 text-orange-800 border-orange-300',
    4: 'bg-slate-100 text-slate-600 border-slate-300',
  };
  const color = colors[rank as keyof typeof colors] || colors[4];

  return (
    <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full border-2 font-bold ${color}`}>
      {rank}
    </span>
  );
}

export default async function AgentsPage() {
  let performance;
  let leaderboard;
  let trends;
  let error = null;

  try {
    [performance, leaderboard, trends] = await Promise.all([
      getAgentPerformance(),
      getAgentLeaderboard(),
      getAgentTrends(30),
    ]);
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load agent data';
  }

  if (error) {
    return (
      <main className="min-h-screen p-8 bg-gray-50">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold mb-8">Agent Performance</h1>
          <div className="bg-red-50 border border-red-200 rounded-lg p-6">
            <h2 className="text-red-800 font-semibold">Error Loading Data</h2>
            <p className="text-red-600 mt-2">{error}</p>
          </div>
        </div>
      </main>
    );
  }

  // Process trends data for the mini charts
  const trendsByAgent: Record<string, NonNullable<typeof trends>> = {};
  if (trends) {
    for (const t of trends) {
      if (!trendsByAgent[t.source]) trendsByAgent[t.source] = [];
      trendsByAgent[t.source].push(t);
    }
  }

  // Get latest date's data for summary
  const latestDate = trends?.[trends.length - 1]?.date;

  return (
    <main className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">Agent Performance Dashboard</h1>
            <p className="text-gray-500 mt-1">Compare London letting agents - 30 day metrics</p>
          </div>
          <a
            href="/"
            className="bg-gray-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-gray-700 transition"
          >
            Back to Dashboard
          </a>
        </div>

        {/* Leaderboard */}
        <div className="bg-white rounded-lg shadow mb-8">
          <div className="p-4 border-b">
            <h2 className="text-lg font-semibold">Agent Leaderboard</h2>
            <p className="text-sm text-gray-500">Ranked by pricing stability and inventory turnover</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left">Rank</th>
                  <th className="px-4 py-3 text-left">Agent</th>
                  <th className="px-4 py-3 text-right">Score</th>
                  <th className="px-4 py-3 text-right">Active Listings</th>
                  <th className="px-4 py-3 text-right">Price Cuts (30d)</th>
                  <th className="px-4 py-3 text-right">Avg Cut %</th>
                  <th className="px-4 py-3 text-right">Turnover</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {leaderboard?.map((agent) => (
                  <tr
                    key={agent.source}
                    className={agent.rank === 1 ? 'bg-yellow-50/50' : 'hover:bg-gray-50'}
                  >
                    <td className="px-4 py-3">
                      <RankBadge rank={agent.rank} />
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: AGENT_COLORS[agent.source] || '#888' }}
                        />
                        <span className="font-medium">
                          {AGENT_DISPLAY_NAMES[agent.source] || agent.source}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right font-bold">{agent.score.toFixed(1)}</td>
                    <td className="px-4 py-3 text-right">{agent.active_listings.toLocaleString()}</td>
                    <td className="px-4 py-3 text-right">
                      {agent.price_reductions > 0 ? (
                        <span className="text-red-600">{agent.price_reductions}</span>
                      ) : (
                        <span className="text-green-600">0</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className={agent.avg_change_pct < -1 ? 'text-red-600' : 'text-gray-600'}>
                        {agent.avg_change_pct.toFixed(2)}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">{formatPercent(agent.turnover_pct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Performance Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {performance?.map((agent) => (
            <div key={agent.source} className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center gap-2 mb-4">
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: AGENT_COLORS[agent.source] || '#888' }}
                />
                <h3 className="font-semibold text-lg">
                  {AGENT_DISPLAY_NAMES[agent.source] || agent.source}
                </h3>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 rounded p-2 text-center">
                  <div className="text-xl font-bold">{agent.active_listings}</div>
                  <div className="text-xs text-gray-500">Active</div>
                </div>
                <div className="bg-red-50 rounded p-2 text-center">
                  <div className="text-xl font-bold text-red-700">{agent.price_reductions}</div>
                  <div className="text-xs text-gray-500">Price Cuts</div>
                </div>
                <div className="bg-orange-50 rounded p-2 text-center">
                  <div className="text-xl font-bold text-orange-700">{formatPercent(agent.price_cut_rate)}</div>
                  <div className="text-xs text-gray-500">Cut Rate</div>
                </div>
                <div className="bg-blue-50 rounded p-2 text-center">
                  <div className="text-xl font-bold text-blue-700">{formatPercent(agent.turnover_rate)}</div>
                  <div className="text-xs text-gray-500">Turnover</div>
                </div>
              </div>

              {agent.avg_reduction_pct !== 0 && (
                <div className="mt-3 text-center text-sm text-gray-500">
                  Avg reduction: <span className="text-red-600 font-medium">{agent.avg_reduction_pct.toFixed(2)}%</span>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Metrics Explanation */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Understanding the Metrics</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 text-sm">
            <div>
              <h3 className="font-medium text-gray-900 mb-1">Price Cut Rate</h3>
              <p className="text-gray-600">
                Percentage of listings that received a price reduction in the last 30 days.
                Lower is better - indicates accurate initial pricing.
              </p>
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-1">Avg Reduction %</h3>
              <p className="text-gray-600">
                Average size of price reductions. Closer to 0% is better - small adjustments
                suggest minor market corrections rather than overpricing.
              </p>
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-1">Turnover Rate</h3>
              <p className="text-gray-600">
                Percentage of inventory that turned over (let or removed) in 30 days.
                Higher suggests faster lettings and better market fit.
              </p>
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-1">Score</h3>
              <p className="text-gray-600">
                Composite score based on pricing stability (fewer cuts, smaller adjustments)
                and inventory efficiency (good turnover). Higher is better.
              </p>
            </div>
          </div>
        </div>

        {/* Data freshness */}
        <div className="mt-6 text-center text-sm text-gray-500">
          Data updated daily. Excludes Rightmove (aggregator, not an agency).
        </div>
      </div>
    </main>
  );
}
