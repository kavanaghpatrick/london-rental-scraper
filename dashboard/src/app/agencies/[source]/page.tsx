import { getAgencyDetail, getAgencyMarketShare, getAgencyPortfolio, getAgencyList, getAgencyTrends, getAgencyPriceHistory } from '@/lib/db';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { InventoryTrendChart, ActivityChart, CumulativeLetsChart, PriceHistoryTable } from '../components/TrendCharts';

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

function HealthScore({ score }: { score: number }) {
  const getColor = (s: number) => {
    if (s >= 80) return 'text-green-600 bg-green-100';
    if (s >= 60) return 'text-yellow-600 bg-yellow-100';
    if (s >= 40) return 'text-orange-600 bg-orange-100';
    return 'text-red-600 bg-red-100';
  };

  const getLabel = (s: number) => {
    if (s >= 80) return 'Excellent';
    if (s >= 60) return 'Good';
    if (s >= 40) return 'Fair';
    return 'Needs Improvement';
  };

  return (
    <div className="text-center">
      <div className={`inline-block text-4xl font-bold px-6 py-3 rounded-xl ${getColor(score)}`}>
        {score}
      </div>
      <div className="text-sm text-gray-500 mt-2">{getLabel(score)}</div>
    </div>
  );
}

function StatCard({
  label,
  value,
  subValue,
  icon,
}: {
  label: string;
  value: string | number;
  subValue?: string;
  icon?: string;
}) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center gap-2 text-gray-500 text-sm">
        {icon && <span>{icon}</span>}
        {label}
      </div>
      <div className="text-2xl font-bold mt-1">{value}</div>
      {subValue && <div className="text-xs text-gray-400 mt-1">{subValue}</div>}
    </div>
  );
}

function PortfolioBar({
  segments,
  colors,
}: {
  segments: { label: string; value: number; count: number }[];
  colors: string[];
}) {
  const total = segments.reduce((sum, s) => sum + s.value, 0);
  if (total === 0) return null;

  return (
    <div>
      <div className="h-8 flex rounded-lg overflow-hidden">
        {segments.map((seg, i) => (
          <div
            key={seg.label}
            className={`${colors[i % colors.length]} flex items-center justify-center text-xs text-white font-medium`}
            style={{ width: `${seg.value}%` }}
            title={`${seg.label}: ${seg.count} (${seg.value.toFixed(1)}%)`}
          >
            {seg.value > 10 ? seg.label : ''}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-3 mt-3">
        {segments.map((seg, i) => (
          <div key={seg.label} className="flex items-center gap-1.5 text-xs">
            <div className={`w-3 h-3 rounded ${colors[i % colors.length]}`} />
            <span className="text-gray-600">{seg.label}</span>
            <span className="text-gray-400">({seg.count})</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MarketShareRow({
  district,
  share,
  rank,
  count,
  avgPrice,
}: {
  district: string;
  share: number;
  rank: number;
  count: number;
  avgPrice: number;
}) {
  const rankColors = {
    1: 'bg-green-100 text-green-800',
    2: 'bg-blue-100 text-blue-800',
    3: 'bg-purple-100 text-purple-800',
  };
  const rankColor = rankColors[rank as keyof typeof rankColors] || 'bg-gray-100 text-gray-600';

  return (
    <div className="flex items-center justify-between py-2 border-b last:border-0">
      <div className="flex items-center gap-3">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${rankColor}`}>
          #{rank}
        </span>
        <span className="font-medium">{district}</span>
      </div>
      <div className="flex items-center gap-4 text-sm">
        <span className="text-gray-500">{count} listings</span>
        <span className="text-gray-500">{formatCurrency(avgPrice)}/mo avg</span>
        <span className="font-semibold text-blue-600">{formatPercent(share)}</span>
      </div>
    </div>
  );
}

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export default async function AgencyDetailPage({
  params,
}: {
  params: Promise<{ source: string }>;
}) {
  const { source } = await params;

  let error = null;
  let detail = null;
  let marketShare: Awaited<ReturnType<typeof getAgencyMarketShare>> = [];
  let portfolio: Awaited<ReturnType<typeof getAgencyPortfolio>> = {
    by_price_tier: [],
    by_bedrooms: [],
    by_property_type: [],
  };
  let allAgencies: Awaited<ReturnType<typeof getAgencyList>> = [];
  let trends: Awaited<ReturnType<typeof getAgencyTrends>> = [];
  let priceHistory: Awaited<ReturnType<typeof getAgencyPriceHistory>> = [];

  try {
    [detail, marketShare, portfolio, allAgencies, trends, priceHistory] = await Promise.all([
      getAgencyDetail(source),
      getAgencyMarketShare(source),
      getAgencyPortfolio(source),
      getAgencyList(),
      getAgencyTrends(source),
      getAgencyPriceHistory(source),
    ]);
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load agency data';
  }

  if (error) {
    return (
      <main className="min-h-screen p-8 bg-gray-50">
        <h1 className="text-3xl font-bold mb-8">Agency Detail</h1>
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h2 className="text-red-800 font-semibold">Error Loading Data</h2>
          <p className="text-red-600 mt-2">{error}</p>
        </div>
      </main>
    );
  }

  if (!detail) {
    notFound();
  }

  // Find this agency's rank
  const currentRank = allAgencies.findIndex((a) => a.source === source) + 1;

  // Color palettes for charts
  const tierColors = ['bg-blue-400', 'bg-blue-500', 'bg-blue-600', 'bg-blue-700', 'bg-blue-800'];
  const bedColors = ['bg-purple-300', 'bg-purple-400', 'bg-purple-500', 'bg-purple-600', 'bg-purple-700', 'bg-purple-800'];
  const typeColors = ['bg-green-400', 'bg-green-500', 'bg-green-600', 'bg-green-700', 'bg-teal-500', 'bg-teal-600'];

  return (
    <main className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <Link href="/agencies" className="text-sm text-gray-500 hover:text-gray-700">
              ← Back to Leaderboard
            </Link>
            <h1 className="text-3xl font-bold mt-2">{detail.display_name}</h1>
            <p className="text-gray-500 mt-1">
              Rank #{currentRank} of {allAgencies.length} agencies
            </p>
          </div>
          <HealthScore score={detail.health_score} />
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-8">
          <StatCard
            label="Active Listings"
            value={detail.active_listings.toLocaleString()}
            icon="🏠"
          />
          <StatCard
            label="Properties Let"
            value={detail.inactive_listings.toLocaleString()}
            subValue={`${formatPercent(detail.let_rate)} let rate`}
            icon="✅"
          />
          <StatCard
            label="Est. Annual GMV"
            value={formatCurrency(detail.estimated_annual_gmv, true)}
            subValue="Based on lets"
            icon="💰"
          />
          <StatCard
            label="Commission"
            value={formatCurrency(detail.estimated_commission, true)}
            subValue="10% of GMV"
            icon="📊"
          />
          <StatCard
            label="Avg Days on Market"
            value={detail.avg_days_on_market > 0 ? `${detail.avg_days_on_market.toFixed(0)}` : '-'}
            subValue="for let properties"
            icon="📅"
          />
          <StatCard
            label="Data Quality"
            value={`${detail.sqft_coverage.toFixed(0)}%`}
            subValue="have sqft data"
            icon="📐"
          />
        </div>

        {/* Pricing Analysis */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Pricing Analysis</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div>
              <div className="text-sm text-gray-500">Average Rent</div>
              <div className="text-2xl font-bold">{formatCurrency(detail.avg_price_pcm)}/mo</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Median Rent</div>
              <div className="text-2xl font-bold">{formatCurrency(detail.median_price_pcm)}/mo</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Average £/sqft</div>
              <div className="text-2xl font-bold">
                {detail.avg_ppsf > 0 ? `£${detail.avg_ppsf.toFixed(2)}` : '-'}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Median £/sqft</div>
              <div className="text-2xl font-bold">
                {detail.median_ppsf > 0 ? `£${detail.median_ppsf.toFixed(2)}` : '-'}
              </div>
            </div>
          </div>

          <div className="mt-6 pt-6 border-t">
            <div className="flex items-center gap-4">
              <div className="text-sm text-gray-500">Price Reductions (30 days):</div>
              <div className="font-semibold text-orange-600">
                {detail.price_reductions} listings ({formatPercent(detail.price_reduction_rate)} of inventory)
              </div>
            </div>
          </div>
        </div>

        {/* Trend Charts */}
        {trends.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <InventoryTrendChart data={trends} agencyName={detail.display_name} />
            <ActivityChart data={trends} agencyName={detail.display_name} />
          </div>
        )}

        {/* Cumulative Lets + Price History */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {trends.length > 0 && (
            <CumulativeLetsChart data={trends} agencyName={detail.display_name} />
          )}
          <PriceHistoryTable data={priceHistory} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Market Share by District */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Market Share by District</h2>
            {marketShare.length > 0 ? (
              <div className="space-y-1">
                {marketShare.slice(0, 10).map((ms) => (
                  <MarketShareRow
                    key={ms.district}
                    district={ms.district}
                    share={ms.market_share}
                    rank={ms.rank}
                    count={ms.agency_count}
                    avgPrice={ms.avg_price_pcm}
                  />
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-sm">No market share data available</p>
            )}
          </div>

          {/* Portfolio Breakdown */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Portfolio Breakdown</h2>

            {/* By Price Tier */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-600 mb-2">By Price Tier</h3>
              {portfolio.by_price_tier.length > 0 ? (
                <PortfolioBar
                  segments={portfolio.by_price_tier.map((t) => ({
                    label: t.segment,
                    value: t.percentage,
                    count: t.count,
                  }))}
                  colors={tierColors}
                />
              ) : (
                <p className="text-gray-400 text-sm">No data</p>
              )}
            </div>

            {/* By Bedrooms */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-600 mb-2">By Bedrooms</h3>
              {portfolio.by_bedrooms.length > 0 ? (
                <PortfolioBar
                  segments={portfolio.by_bedrooms.map((b) => ({
                    label: b.segment,
                    value: b.percentage,
                    count: b.count,
                  }))}
                  colors={bedColors}
                />
              ) : (
                <p className="text-gray-400 text-sm">No data</p>
              )}
            </div>

            {/* By Property Type */}
            <div>
              <h3 className="text-sm font-medium text-gray-600 mb-2">By Property Type</h3>
              {portfolio.by_property_type.length > 0 ? (
                <PortfolioBar
                  segments={portfolio.by_property_type.map((p) => ({
                    label: p.segment,
                    value: p.percentage,
                    count: p.count,
                  }))}
                  colors={typeColors}
                />
              ) : (
                <p className="text-gray-400 text-sm">No data</p>
              )}
            </div>
          </div>
        </div>

        {/* Competitive Comparison */}
        {allAgencies.length > 1 && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Competitive Comparison</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left">Agency</th>
                    <th className="px-4 py-2 text-right">Active</th>
                    <th className="px-4 py-2 text-right">Let Rate</th>
                    <th className="px-4 py-2 text-right">Avg Rent</th>
                    <th className="px-4 py-2 text-right">£/sqft</th>
                    <th className="px-4 py-2 text-right">Est. GMV</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {allAgencies.map((agency) => (
                    <tr
                      key={agency.source}
                      className={agency.source === source ? 'bg-blue-50' : 'hover:bg-gray-50'}
                    >
                      <td className="px-4 py-2">
                        <Link
                          href={`/agencies/${agency.source}`}
                          className={`font-medium ${
                            agency.source === source
                              ? 'text-blue-700'
                              : 'text-gray-700 hover:text-blue-600'
                          }`}
                        >
                          {agency.display_name}
                          {agency.source === source && (
                            <span className="ml-2 text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                              You
                            </span>
                          )}
                        </Link>
                      </td>
                      <td className="px-4 py-2 text-right">{agency.active_listings}</td>
                      <td className="px-4 py-2 text-right">{formatPercent(agency.let_rate)}</td>
                      <td className="px-4 py-2 text-right">{formatCurrency(agency.avg_price_pcm)}</td>
                      <td className="px-4 py-2 text-right">
                        {agency.avg_ppsf > 0 ? `£${agency.avg_ppsf.toFixed(2)}` : '-'}
                      </td>
                      <td className="px-4 py-2 text-right font-medium">
                        {formatCurrency(agency.estimated_annual_gmv, true)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
