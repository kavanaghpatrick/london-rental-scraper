import { getComparables, getMarketStats, getPpsfDistribution, getPpsfByDistrict, Comparable } from '@/lib/db';
import ReportCharts from './ReportCharts';

// Default subject property - can be made configurable via URL params
const SUBJECT = {
  address: '4 South Eaton Place',
  postcode: 'SW1W',
  size_sqft: 1312,
  bedrooms: 2,
  bathrooms: 2,
  predicted_pcm: 8925,
  range_low: 7318,
  range_high: 10531,
};

// SW1 area districts
const SW1_DISTRICTS = new Set(['SW1A', 'SW1E', 'SW1H', 'SW1P', 'SW1V', 'SW1W', 'SW1X', 'SW1Y']);
const PRIME_ADJACENT = new Set(['SW3', 'SW7', 'W1', 'W8', 'NW1', 'NW3', 'NW8']);

function isSW1District(district: string): boolean {
  if (!district) return false;
  // Handle exact matches (SW1A, SW1W, etc.)
  if (SW1_DISTRICTS.has(district)) return true;
  // Handle partial 'SW1' or 'SW1X' format
  if (district === 'SW1') return true;
  // Handle SW1X format where X is a letter (length 4)
  if (district.startsWith('SW1') && district.length === 4 && /[A-Z]/.test(district[3])) return true;
  return false;
}

function getTier(district: string, subjectDistrict: string): { num: number; label: string } {
  if (!district) return { num: 4, label: 'Tier 4: Broader Market' };

  if (district === subjectDistrict) return { num: 1, label: 'Tier 1: Same District' };

  if (isSW1District(subjectDistrict) && isSW1District(district)) {
    return { num: 2, label: 'Tier 2: SW1 Area' };
  }

  const areaCode = district.match(/^([A-Z]+\d+)/)?.[1] || district;
  if (PRIME_ADJACENT.has(areaCode) || PRIME_ADJACENT.has(district)) {
    return { num: 3, label: 'Tier 3: Prime Central' };
  }

  return { num: 4, label: 'Tier 4: Broader Market' };
}

function formatCurrency(value: number): string {
  return `£${value.toLocaleString()}`;
}

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export default async function ValuationReport() {
  const subjectPpsf = SUBJECT.predicted_pcm / SUBJECT.size_sqft;

  let comparables: Comparable[] = [];
  let marketStats = { total_listings: 0, median_ppsf: 0, p25_ppsf: 0, p75_ppsf: 0, avg_ppsf: 0 };
  let ppsfDistribution: { bucket: number; count: number }[] = [];
  let ppsfByDistrict: { district: string; median_ppsf: number; count: number }[] = [];
  let error = null;

  try {
    [comparables, marketStats, ppsfDistribution, ppsfByDistrict] = await Promise.all([
      getComparables(SUBJECT.size_sqft, SUBJECT.bedrooms),
      getMarketStats(),
      getPpsfDistribution(),
      getPpsfByDistrict(),
    ]);
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load data';
  }

  if (error) {
    return (
      <main className="min-h-screen p-8 bg-gray-100">
        <div className="max-w-4xl mx-auto bg-red-50 border border-red-200 rounded-lg p-6">
          <h2 className="text-red-800 font-semibold">Error Loading Report</h2>
          <p className="text-red-600 mt-2">{error}</p>
        </div>
      </main>
    );
  }

  // Add tier info to comparables
  const comparablesWithTier = comparables.map(comp => ({
    ...comp,
    tier: getTier(comp.district, SUBJECT.postcode),
  }));

  // Group by tier
  const tier1 = comparablesWithTier.filter(c => c.tier.num === 1);
  const tier2 = comparablesWithTier.filter(c => c.tier.num === 2);
  const tier3 = comparablesWithTier.filter(c => c.tier.num === 3);
  const tier4 = comparablesWithTier.filter(c => c.tier.num === 4);

  // Calculate market percentile
  const belowSubject = ppsfDistribution
    .filter(d => d.bucket < subjectPpsf)
    .reduce((sum, d) => sum + d.count, 0);
  const totalInDist = ppsfDistribution.reduce((sum, d) => sum + d.count, 0);
  const marketPercentile = totalInDist > 0 ? Math.round((belowSubject / totalInDist) * 100) : 0;

  // Tier medians
  const getMedianPpsf = (items: typeof comparablesWithTier) => {
    if (items.length === 0) return 0;
    const sorted = [...items].sort((a, b) => a.ppsf - b.ppsf);
    return sorted[Math.floor(sorted.length / 2)].ppsf;
  };

  const tierStats = {
    tier1: { count: tier1.length, median: getMedianPpsf(tier1) },
    tier2: { count: tier2.length, median: getMedianPpsf(tier2) },
    tier3: { count: tier3.length, median: getMedianPpsf(tier3) },
    tier4: { count: tier4.length, median: getMedianPpsf(tier4) },
  };

  const generatedAt = new Date().toLocaleString('en-GB', {
    day: '2-digit',
    month: 'long',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <main className="min-h-screen p-4 md:p-8 bg-gray-100">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-900 to-blue-700 text-white rounded-xl p-6 md:p-8 mb-6 text-center">
          <h1 className="text-2xl md:text-3xl font-bold mb-2">Property Valuation Report</h1>
          <p className="text-lg opacity-90">{SUBJECT.address}, London {SUBJECT.postcode}</p>
          <p className="text-sm opacity-75 mt-4">
            Generated: {generatedAt} | Live Data: {marketStats.total_listings.toLocaleString()} listings
          </p>
        </div>

        {/* Valuation Box */}
        <div className="bg-gradient-to-r from-blue-50 to-blue-100 border-2 border-blue-400 rounded-xl p-6 mb-6 text-center">
          <div className="text-sm text-gray-600 mb-1">ESTIMATED MONTHLY RENT</div>
          <div className="text-4xl md:text-5xl font-bold text-blue-700">{formatCurrency(SUBJECT.predicted_pcm)}</div>
          <div className="text-gray-600 mt-2">
            Confidence Range: {formatCurrency(SUBJECT.range_low)} - {formatCurrency(SUBJECT.range_high)} pcm
          </div>
          <div className="text-sm text-gray-500 mt-2">
            £{subjectPpsf.toFixed(2)}/sqft | {marketPercentile}th percentile of market
          </div>
        </div>

        {/* Property Details */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-blue-900">Property Details</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">{SUBJECT.size_sqft.toLocaleString()}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Square Feet</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">{SUBJECT.bedrooms}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Bedrooms</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">{SUBJECT.bathrooms}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Bathrooms</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">{SUBJECT.postcode}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Postcode</div>
            </div>
          </div>
        </div>

        {/* Market Position */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-blue-900">Market Position</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">£{subjectPpsf.toFixed(2)}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Subject £/sqft</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">£{marketStats.median_ppsf?.toFixed(2) || 'N/A'}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Market Median £/sqft</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">{marketPercentile}th</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Market Percentile</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">{comparables.length}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Comparables</div>
            </div>
          </div>
          <p className="text-sm text-gray-600 text-center mt-4">
            The property&apos;s £/sqft places it in the <strong>{marketPercentile}th percentile</strong> of the London rental market.
            {marketPercentile > 70 && ' This is a premium valuation typical for Belgravia/SW1.'}
          </p>
        </div>

        {/* Charts */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-blue-900">Market Analysis</h2>
          <ReportCharts
            ppsfDistribution={ppsfDistribution}
            ppsfByDistrict={ppsfByDistrict}
            comparables={comparablesWithTier.slice(0, 20)}
            subjectPpsf={subjectPpsf}
            subjectDistrict={SUBJECT.postcode}
            subjectPrice={SUBJECT.predicted_pcm}
            subjectSize={SUBJECT.size_sqft}
          />
        </div>

        {/* Comparables by Tier */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-blue-900">Comparable Properties by Location Tier</h2>
          <p className="text-sm text-gray-600 mb-4">
            Properties similar in size (±20%) with {SUBJECT.bedrooms} bedrooms, ranked by location relevance.
            <span className="ml-2 px-2 py-0.5 bg-green-100 rounded text-green-800 text-xs">Green</span> = lower £/sqft,
            <span className="ml-1 px-2 py-0.5 bg-red-100 rounded text-red-800 text-xs">Red</span> = higher £/sqft
          </p>

          {/* Tier 1 */}
          {tier1.length > 0 && (
            <TierSection
              title={`Tier 1: Same District (${SUBJECT.postcode})`}
              description="Direct comparables in the same postcode district"
              bgColor="bg-teal-50"
              items={tier1}
              stats={tierStats.tier1}
              subjectPpsf={subjectPpsf}
              subjectSize={SUBJECT.size_sqft}
            />
          )}

          {/* Tier 2 */}
          {tier2.length > 0 && (
            <TierSection
              title="Tier 2: SW1 Area"
              description="Other SW1 districts (SW1X, SW1V, SW1A, etc.) - similar prime Belgravia/Westminster market"
              bgColor="bg-blue-50"
              items={tier2}
              stats={tierStats.tier2}
              subjectPpsf={subjectPpsf}
              subjectSize={SUBJECT.size_sqft}
            />
          )}

          {/* Tier 3 */}
          {tier3.length > 0 && (
            <TierSection
              title="Tier 3: Prime Central London"
              description="Adjacent prime areas (SW3 Chelsea, SW7 South Ken, W1 Mayfair, W8 Kensington)"
              bgColor="bg-purple-50"
              items={tier3}
              stats={tierStats.tier3}
              subjectPpsf={subjectPpsf}
              subjectSize={SUBJECT.size_sqft}
            />
          )}

          {/* Tier 4 */}
          {tier4.length > 0 && (
            <TierSection
              title="Tier 4: Broader Market"
              description="Properties of similar size across wider London market"
              bgColor="bg-gray-50"
              items={tier4}
              stats={tierStats.tier4}
              subjectPpsf={subjectPpsf}
              subjectSize={SUBJECT.size_sqft}
            />
          )}

          <p className="text-xs text-gray-500 mt-4">
            Total: {comparables.length} comparables across all tiers
          </p>
        </div>

        {/* Market Statistics */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-blue-900">Market Statistics</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">£{marketStats.p25_ppsf?.toFixed(2) || 'N/A'}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">25th Percentile £/sqft</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">£{marketStats.median_ppsf?.toFixed(2) || 'N/A'}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Median £/sqft</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">£{marketStats.p75_ppsf?.toFixed(2) || 'N/A'}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">75th Percentile £/sqft</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">{marketStats.total_listings?.toLocaleString() || 0}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Total Active Listings</div>
            </div>
          </div>
        </div>

        {/* Methodology */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-blue-900">Methodology</h2>
          <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-600 space-y-2">
            <p><strong>Model:</strong> XGBoost regression with 46+ features including size, location encodings, amenities, and property type indicators.</p>
            <p><strong>Training Data:</strong> {marketStats.total_listings?.toLocaleString() || 0} active London rental listings from multiple agents (Savills, Knight Frank, Foxtons, Chestertons, Rightmove).</p>
            <p><strong>Confidence Range:</strong> Based on model&apos;s ~18% median absolute percentage error.</p>
            <p><strong>Comparables:</strong> Properties within ±20% of subject size, matching bedroom count, from active listings.</p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm py-6">
          <p>Property Valuation Model | Generated {generatedAt}</p>
          <p>{SUBJECT.address}, {SUBJECT.postcode}</p>
          <p className="text-xs mt-2 text-gray-400">
            Disclaimer: This is an automated valuation estimate (AVM). Actual rental values depend on property condition,
            exact fixtures/fittings, lease terms, and current market conditions. This should be used as guidance only.
          </p>
          <p className="mt-4">
            <a href="/" className="text-blue-600 hover:underline">← Back to Dashboard</a>
          </p>
        </div>
      </div>
    </main>
  );
}

// Tier Section Component
function TierSection({
  title,
  description,
  bgColor,
  items,
  stats,
  subjectPpsf,
  subjectSize,
}: {
  title: string;
  description: string;
  bgColor: string;
  items: (Comparable & { tier: { num: number; label: string } })[];
  stats: { count: number; median: number };
  subjectPpsf: number;
  subjectSize: number;
}) {
  return (
    <div className={`${bgColor} rounded-lg p-4 mb-4`}>
      <h4 className="font-semibold text-blue-900 mb-1">{title} ({stats.count} properties)</h4>
      <p className="text-xs text-gray-600 mb-2">{description}</p>
      <p className="text-sm font-medium mb-3">
        Tier Median: £{stats.median.toFixed(2)}/sqft | £{Math.round(stats.median * subjectSize).toLocaleString()}/month
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-left">
              <th className="pb-2">Address</th>
              <th className="pb-2">District</th>
              <th className="pb-2 text-right">Rent</th>
              <th className="pb-2 text-right">Size</th>
              <th className="pb-2 text-right">£/sqft</th>
              <th className="pb-2">Source</th>
            </tr>
          </thead>
          <tbody>
            {items.slice(0, 5).map((comp, idx) => {
              const pctDiff = ((comp.ppsf / subjectPpsf) - 1) * 100;
              const rowBg = pctDiff > 10 ? 'bg-red-100' : pctDiff < -10 ? 'bg-green-100' : '';
              return (
                <tr key={idx} className={rowBg}>
                  <td className="py-1">
                    <a href={comp.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                      {comp.address?.slice(0, 40)}{comp.address?.length > 40 ? '...' : ''}
                    </a>
                  </td>
                  <td className="py-1">{comp.district}</td>
                  <td className="py-1 text-right">£{comp.price_pcm.toLocaleString()}</td>
                  <td className="py-1 text-right">{comp.size_sqft.toLocaleString()}</td>
                  <td className="py-1 text-right">£{comp.ppsf.toFixed(2)}</td>
                  <td className="py-1">{comp.source}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {items.length > 5 && (
        <p className="text-xs text-gray-500 mt-2">Showing 5 of {items.length}</p>
      )}
    </div>
  );
}
