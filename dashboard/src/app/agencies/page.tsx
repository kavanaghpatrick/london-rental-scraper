import { getAgencyList, AgencySummary } from '@/lib/db';
import Link from 'next/link';

function formatCurrency(value: number | bigint, compact = false): string {
  const num = typeof value === 'bigint' ? Number(value) : value;
  if (compact) {
    if (num >= 1000000) return `£${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `£${(num / 1000).toFixed(0)}k`;
  }
  return `£${num.toLocaleString()}`;
}

function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

function RankBadge({ rank }: { rank: number }) {
  const colors = {
    1: 'bg-yellow-100 text-yellow-800 border-yellow-300',
    2: 'bg-gray-100 text-gray-700 border-gray-300',
    3: 'bg-orange-100 text-orange-800 border-orange-300',
  };
  const color = colors[rank as keyof typeof colors] || 'bg-blue-50 text-blue-700 border-blue-200';
  const medal = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : `#${rank}`;

  return (
    <span className={`px-3 py-1 rounded-full text-sm font-bold border ${color}`}>
      {medal}
    </span>
  );
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
  trend?: number;
  color?: 'blue' | 'green' | 'purple' | 'orange';
}) {
  const bgColors = {
    blue: 'bg-blue-50',
    green: 'bg-green-50',
    purple: 'bg-purple-50',
    orange: 'bg-orange-50',
  };
  const textColors = {
    blue: 'text-blue-700',
    green: 'text-green-700',
    purple: 'text-purple-700',
    orange: 'text-orange-700',
  };

  return (
    <div className={`${bgColors[color]} rounded-lg p-4 text-center`}>
      <div className={`text-2xl font-bold ${textColors[color]}`}>
        {value}
        {trend !== undefined && trend !== 0 && (
          <span className={`text-sm ml-2 ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
            {trend > 0 ? '↑' : '↓'}{Math.abs(trend)}
          </span>
        )}
      </div>
      <div className="text-xs text-gray-500 mt-1">{title}</div>
      {subtitle && <div className="text-xs text-gray-400 mt-0.5">{subtitle}</div>}
    </div>
  );
}

function AgencyRow({ agency, showGmv }: { agency: AgencySummary; showGmv: boolean }) {
  return (
    <tr className="hover:bg-gray-50 transition">
      <td className="px-4 py-3">
        <RankBadge rank={agency.rank} />
      </td>
      <td className="px-4 py-3">
        <Link
          href={`/agencies/${agency.source}`}
          className="font-semibold text-blue-600 hover:text-blue-800 hover:underline"
        >
          {agency.display_name}
        </Link>
      </td>
      <td className="px-4 py-3 text-right font-medium">
        {agency.active_listings.toLocaleString()}
      </td>
      <td className="px-4 py-3 text-right">
        <span className={agency.let_rate > 30 ? 'text-green-600 font-medium' : ''}>
          {formatPercent(agency.let_rate)}
        </span>
      </td>
      <td className="px-4 py-3 text-right">
        {formatCurrency(agency.avg_price_pcm)}/mo
      </td>
      <td className="px-4 py-3 text-right">
        {agency.avg_ppsf > 0 ? `£${agency.avg_ppsf.toFixed(2)}` : '-'}
      </td>
      {showGmv && (
        <>
          <td className="px-4 py-3 text-right font-medium text-green-700">
            {formatCurrency(agency.estimated_annual_gmv, true)}
          </td>
          <td className="px-4 py-3 text-right text-purple-700">
            {formatCurrency(agency.estimated_commission, true)}
          </td>
        </>
      )}
    </tr>
  );
}

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export default async function AgenciesPage() {
  let agencies: AgencySummary[] = [];
  let error = null;

  try {
    agencies = await getAgencyList();
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load agency data';
  }

  if (error) {
    return (
      <main className="min-h-screen p-8 bg-gray-50">
        <h1 className="text-3xl font-bold mb-8">Agency Leaderboard</h1>
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h2 className="text-red-800 font-semibold">Error Loading Data</h2>
          <p className="text-red-600 mt-2">{error}</p>
        </div>
      </main>
    );
  }

  // Calculate market totals
  const totals = agencies.reduce((acc, agency) => ({
    active: acc.active + agency.active_listings,
    inactive: acc.inactive + agency.inactive_listings,
    gmv: acc.gmv + Number(agency.estimated_annual_gmv),
    commission: acc.commission + Number(agency.estimated_commission),
    portfolioValue: acc.portfolioValue + Number(agency.total_portfolio_value),
  }), { active: 0, inactive: 0, gmv: 0, commission: 0, portfolioValue: 0 });

  const avgLetRate = totals.active + totals.inactive > 0
    ? (totals.inactive / (totals.active + totals.inactive)) * 100
    : 0;

  return (
    <main className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">Agency Leaderboard</h1>
            <p className="text-gray-500 mt-1">Competitive intelligence for London rental agents</p>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="text-gray-600 hover:text-gray-800 text-sm"
            >
              ← Back to Dashboard
            </Link>
            <Link
              href="/agents"
              className="bg-purple-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-purple-700 transition"
            >
              Agent Performance
            </Link>
          </div>
        </div>

        {/* Market Overview */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
          <MetricCard
            title="Total Active Listings"
            value={totals.active.toLocaleString()}
            subtitle={`${agencies.length} agencies`}
            color="blue"
          />
          <MetricCard
            title="Total Lets (Inactive)"
            value={totals.inactive.toLocaleString()}
            subtitle={`${avgLetRate.toFixed(1)}% let rate`}
            color="green"
          />
          <MetricCard
            title="Est. Annual Market GMV"
            value={formatCurrency(totals.gmv, true)}
            subtitle="Based on let activity"
            color="purple"
          />
          <MetricCard
            title="Est. Total Commission"
            value={formatCurrency(totals.commission, true)}
            subtitle="10% of GMV"
            color="orange"
          />
          <MetricCard
            title="Portfolio Value"
            value={formatCurrency(totals.portfolioValue, true)}
            subtitle="Annual asking rents"
            color="blue"
          />
        </div>

        {/* GMV Methodology Note */}
        <div className="bg-blue-50 border border-blue-100 rounded-lg p-4 mb-6 text-sm">
          <span className="font-semibold text-blue-800">GMV Methodology:</span>{' '}
          <span className="text-blue-700">
            Inactive listings × 90% fall-through × avg asking rent × 95% (negotiation) × 12 months,
            annualized with seasonality adjustment. Commission estimated at 10% of GMV.
          </span>
        </div>

        {/* Leaderboard Table */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="p-4 border-b flex justify-between items-center">
            <h2 className="text-lg font-semibold">Agency Rankings by GMV</h2>
            <span className="text-xs text-gray-500">
              Excludes Rightmove (aggregator)
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left">Rank</th>
                  <th className="px-4 py-3 text-left">Agency</th>
                  <th className="px-4 py-3 text-right">Active</th>
                  <th className="px-4 py-3 text-right">Let Rate</th>
                  <th className="px-4 py-3 text-right">Avg Rent</th>
                  <th className="px-4 py-3 text-right">£/sqft</th>
                  <th className="px-4 py-3 text-right">Est. GMV</th>
                  <th className="px-4 py-3 text-right">Commission</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {agencies.map((agency) => (
                  <AgencyRow key={agency.source} agency={agency} showGmv={true} />
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Quick Insights */}
        {agencies.length >= 2 && (
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="font-semibold text-gray-800 mb-3">Market Leader</h3>
              <div className="text-3xl font-bold text-blue-600">{agencies[0]?.display_name}</div>
              <p className="text-sm text-gray-500 mt-2">
                {formatCurrency(agencies[0]?.estimated_annual_gmv || 0, true)} estimated annual GMV
                with {agencies[0]?.active_listings.toLocaleString()} active listings
              </p>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="font-semibold text-gray-800 mb-3">Highest Let Rate</h3>
              {(() => {
                const highest = [...agencies].sort((a, b) => b.let_rate - a.let_rate)[0];
                return highest ? (
                  <>
                    <div className="text-3xl font-bold text-green-600">{highest.display_name}</div>
                    <p className="text-sm text-gray-500 mt-2">
                      {formatPercent(highest.let_rate)} of listings successfully let,
                      indicating strong market positioning
                    </p>
                  </>
                ) : null;
              })()}
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="font-semibold text-gray-800 mb-3">Premium Positioning</h3>
              {(() => {
                const premium = [...agencies].filter(a => a.avg_ppsf > 0).sort((a, b) => b.avg_ppsf - a.avg_ppsf)[0];
                return premium ? (
                  <>
                    <div className="text-3xl font-bold text-purple-600">{premium.display_name}</div>
                    <p className="text-sm text-gray-500 mt-2">
                      £{premium.avg_ppsf.toFixed(2)}/sqft average - highest price point
                      in the market
                    </p>
                  </>
                ) : null;
              })()}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
