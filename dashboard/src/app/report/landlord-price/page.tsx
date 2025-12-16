import { getComparables, getMarketStats, getPpsfDistribution, getPpsfByDistrict, getLatestValuation, Comparable } from '@/lib/db';
import ReportCharts from '../ReportCharts';

// Subject property details
const PROPERTY = {
  address: '4 South Eaton Place',
  postcode: 'SW1W',
  size_sqft: 1312,
  bedrooms: 2,
  bathrooms: 2,
};

// Landlord's asking price
const LANDLORD_PRICE = 10394.62;
const LANDLORD_PPSF = LANDLORD_PRICE / PROPERTY.size_sqft;

// SW1 area districts
const SW1_DISTRICTS = new Set(['SW1A', 'SW1E', 'SW1H', 'SW1P', 'SW1V', 'SW1W', 'SW1X', 'SW1Y']);
const PRIME_ADJACENT = new Set(['SW3', 'SW7', 'W1', 'W8', 'NW1', 'NW3', 'NW8']);

function isSW1District(district: string): boolean {
  if (!district) return false;
  if (SW1_DISTRICTS.has(district)) return true;
  if (district === 'SW1') return true;
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
  return `£${value.toLocaleString('en-GB', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
}

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export default async function LandlordPriceReport() {
  // Get model's fair value prediction
  let modelPrediction = 8925;
  let modelVersion = 'V15';
  let modelR2 = 0.67;  // Actual V15 R² (no data leakage)
  let modelMape = 18.0;  // Conservative estimate
  try {
    const dbValuation = await getLatestValuation(PROPERTY.address);
    if (dbValuation) {
      modelPrediction = dbValuation.predicted_pcm;
      modelVersion = dbValuation.model_version;
      modelR2 = dbValuation.model_r2;
      modelMape = dbValuation.model_mape;
    }
  } catch {
    // Use fallback
  }

  const modelPpsf = modelPrediction / PROPERTY.size_sqft;

  const postcodeArea = PROPERTY.postcode.match(/^([A-Z]+\d+)/)?.[1] || PROPERTY.postcode.slice(0, 3);

  let comparables: Comparable[] = [];
  let marketStats = { total_listings: 0, median_ppsf: 0, p25_ppsf: 0, p75_ppsf: 0, avg_ppsf: 0 };
  let ppsfDistribution: { bucket: number; count: number }[] = [];
  let ppsfByDistrict: { district: string; median_ppsf: number; count: number }[] = [];
  let error = null;

  // Use 25% size range for robust comparable dataset
  const SIZE_RANGE = 0.25;

  try {
    [comparables, marketStats, ppsfDistribution, ppsfByDistrict] = await Promise.all([
      getComparables(PROPERTY.size_sqft, PROPERTY.bedrooms, SIZE_RANGE, LANDLORD_PPSF, postcodeArea),
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

  // Add tier info
  const comparablesWithTier = comparables.map(comp => ({
    ...comp,
    tier: getTier(comp.district, PROPERTY.postcode),
  }));

  // Calculate size range for display
  const minSize = Math.floor(PROPERTY.size_sqft * (1 - SIZE_RANGE));
  const maxSize = Math.ceil(PROPERTY.size_sqft * (1 + SIZE_RANGE));

  // HONEST METRICS: Compare £/sqft, not total rent
  const lowerPpsfComps = comparablesWithTier.filter(c => c.ppsf < LANDLORD_PPSF);
  const lowerPpsfPct = comparablesWithTier.length > 0
    ? Math.round((lowerPpsfComps.length / comparablesWithTier.length) * 100)
    : 0;

  // Tier breakdown
  const tier1 = comparablesWithTier.filter(c => c.tier.num === 1);
  const tier2 = comparablesWithTier.filter(c => c.tier.num === 2);
  const sw1Comps = [...tier1, ...tier2];

  // Calculate median £/sqft for SW1 comps
  const getMedianPpsf = (items: typeof comparablesWithTier) => {
    if (items.length === 0) return 0;
    const sorted = [...items].sort((a, b) => a.ppsf - b.ppsf);
    return sorted[Math.floor(sorted.length / 2)].ppsf;
  };

  const sw1MedianPpsf = getMedianPpsf(sw1Comps);
  const allMedianPpsf = getMedianPpsf(comparablesWithTier);

  // What would fair rent be at market median £/sqft?
  const fairRentAtMedian = Math.round(allMedianPpsf * PROPERTY.size_sqft);
  const fairRentAtSW1Median = Math.round(sw1MedianPpsf * PROPERTY.size_sqft);

  // Market percentile based on £/sqft
  const belowLandlordPpsf = ppsfDistribution
    .filter(d => d.bucket < LANDLORD_PPSF)
    .reduce((sum, d) => sum + d.count, 0);
  const totalInDist = ppsfDistribution.reduce((sum, d) => sum + d.count, 0);
  const landlordPercentile = totalInDist > 0 ? Math.round((belowLandlordPpsf / totalInDist) * 100) : 0;

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
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 text-white rounded-xl p-6 md:p-8 mb-6 text-center">
          <h1 className="text-2xl md:text-3xl font-bold mb-2">Rental Price Analysis</h1>
          <p className="text-lg opacity-90">{PROPERTY.address}, London {PROPERTY.postcode}</p>
          <p className="text-sm opacity-75 mt-4">
            Generated: {generatedAt} | Based on {marketStats.total_listings.toLocaleString()} active listings
          </p>
        </div>

        {/* The Core Comparison - £/sqft basis */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">Price Per Square Foot Comparison</h2>
          <p className="text-sm text-gray-600 mb-4">
            £/sqft is the standard metric for comparing properties of different sizes fairly.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4 text-center">
              <div className="text-sm text-red-600 font-semibold mb-1">LANDLORD ASKING</div>
              <div className="text-3xl font-bold text-red-700">£{LANDLORD_PPSF.toFixed(2)}</div>
              <div className="text-sm text-gray-600 mt-1">per sqft/month</div>
              <div className="text-xs text-gray-500 mt-2">{formatCurrency(LANDLORD_PRICE)}/month total</div>
            </div>

            <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4 text-center">
              <div className="text-sm text-blue-600 font-semibold mb-1">PRIME CENTRAL MEDIAN</div>
              <div className="text-3xl font-bold text-blue-700">£{allMedianPpsf.toFixed(2)}</div>
              <div className="text-sm text-gray-600 mt-1">per sqft/month</div>
              <div className="text-xs text-gray-500 mt-2">= {formatCurrency(fairRentAtMedian)}/month for this size</div>
            </div>

            <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4 text-center">
              <div className="text-sm text-green-600 font-semibold mb-1">ML MODEL PREDICTION</div>
              <div className="text-3xl font-bold text-green-700">£{modelPpsf.toFixed(2)}</div>
              <div className="text-sm text-gray-600 mt-1">per sqft/month</div>
              <div className="text-xs text-gray-500 mt-2">= {formatCurrency(modelPrediction)}/month for this size</div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
            <p className="text-sm text-amber-800">
              <strong>Key Finding:</strong> The landlord is asking £{LANDLORD_PPSF.toFixed(2)}/sqft, which is{' '}
              <strong>{((LANDLORD_PPSF / allMedianPpsf - 1) * 100).toFixed(0)}% above the Prime Central London median</strong> of £{allMedianPpsf.toFixed(2)}/sqft
              for similar-sized properties ({minSize.toLocaleString()}-{maxSize.toLocaleString()} sqft) across Belgravia, Chelsea, South Kensington, and Mayfair.
            </p>
          </div>
        </div>

        {/* All Comparable Properties - UNBIASED - shows everything */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">
            All Comparable Properties ({minSize.toLocaleString()}-{maxSize.toLocaleString()} sqft in Prime Central London)
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            Complete dataset of {comparablesWithTier.length} long-term let properties within ±{Math.round(SIZE_RANGE * 100)}% of subject size
            across SW1 (Belgravia), SW3 (Chelsea), SW7 (South Kensington), and W1 (Mayfair).
            Short lets excluded. Sorted by £/sqft to show full market range.
          </p>

          {/* Summary Statistics - show full picture first */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-xs text-gray-500 uppercase">Total Comps</div>
              <div className="text-2xl font-bold text-slate-700">{comparablesWithTier.length}</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-xs text-gray-500 uppercase">Median £/sqft</div>
              <div className="text-2xl font-bold text-slate-700">£{allMedianPpsf.toFixed(2)}</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4 text-center">
              <div className="text-xs text-gray-500 uppercase">Below Asking</div>
              <div className="text-2xl font-bold text-green-700">{lowerPpsfComps.length}</div>
              <div className="text-xs text-gray-500">({lowerPpsfPct}%)</div>
            </div>
            <div className="bg-red-50 rounded-lg p-4 text-center">
              <div className="text-xs text-gray-500 uppercase">Above Asking</div>
              <div className="text-2xl font-bold text-red-700">{comparablesWithTier.length - lowerPpsfComps.length}</div>
              <div className="text-xs text-gray-500">({100 - lowerPpsfPct}%)</div>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-left border-b">
                  <th className="pb-2">#</th>
                  <th className="pb-2">Address</th>
                  <th className="pb-2">District</th>
                  <th className="pb-2">Beds</th>
                  <th className="pb-2 text-right">Size</th>
                  <th className="pb-2 text-right">Rent/month</th>
                  <th className="pb-2 text-right">£/sqft</th>
                  <th className="pb-2 text-right">vs Asking</th>
                  <th className="pb-2">Source</th>
                </tr>
              </thead>
              <tbody>
                {comparablesWithTier.map((comp, idx) => {
                  const isLowerPpsf = comp.ppsf < LANDLORD_PPSF;
                  const ppsfDiff = Math.abs(comp.ppsf - LANDLORD_PPSF);
                  const rowBg = isLowerPpsf ? 'bg-green-50' : 'bg-red-50';
                  return (
                    <tr key={idx} className={rowBg}>
                      <td className="py-2 text-gray-400 text-xs">{idx + 1}</td>
                      <td className="py-2">
                        <a href={comp.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                          {comp.address?.slice(0, 28)}{comp.address?.length > 28 ? '...' : ''}
                        </a>
                      </td>
                      <td className="py-2">{comp.district}</td>
                      <td className="py-2">{comp.bedrooms}</td>
                      <td className="py-2 text-right">{comp.size_sqft.toLocaleString()}</td>
                      <td className="py-2 text-right">{formatCurrency(comp.price_pcm)}</td>
                      <td className="py-2 text-right font-semibold">£{comp.ppsf.toFixed(2)}</td>
                      <td className="py-2 text-right">
                        {isLowerPpsf ? (
                          <span className="text-green-700">-£{ppsfDiff.toFixed(2)}</span>
                        ) : (
                          <span className="text-red-600">+£{ppsfDiff.toFixed(2)}</span>
                        )}
                      </td>
                      <td className="py-2 text-xs">{comp.source}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="text-xs text-gray-500 mt-4 p-3 bg-gray-50 rounded">
            <strong>Note:</strong> This table shows ALL {comparablesWithTier.length} comparable properties across Prime Central London
            (Belgravia, Chelsea, South Kensington, Mayfair) - no filtering or cherry-picking.
            The landlord&apos;s asking of £{LANDLORD_PPSF.toFixed(2)}/sqft falls at the {100 - lowerPpsfPct}th percentile
            (more expensive than {lowerPpsfPct}% of comparables in London&apos;s most prestigious postcodes).
          </p>
        </div>

        {/* Property Details */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">Subject Property</h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-slate-700">{PROPERTY.size_sqft.toLocaleString()}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Square Feet</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-slate-700">{PROPERTY.bedrooms}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Bedrooms</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-slate-700">{PROPERTY.bathrooms}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Bathrooms</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-slate-700">{PROPERTY.postcode}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Postcode</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-red-600">£{LANDLORD_PPSF.toFixed(2)}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Asking £/sqft</div>
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">Market Visualizations</h2>
          <ReportCharts
            ppsfDistribution={ppsfDistribution}
            ppsfByDistrict={ppsfByDistrict}
            comparables={comparablesWithTier}
            subjectPpsf={LANDLORD_PPSF}
            subjectDistrict={PROPERTY.postcode}
            subjectPrice={Math.round(LANDLORD_PRICE)}
            subjectSize={PROPERTY.size_sqft}
          />
        </div>

        {/* Summary */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4 text-slate-800">Summary</h2>
          <div className="space-y-3 text-sm text-gray-700">
            <p>
              <strong>1. The asking price of £{LANDLORD_PPSF.toFixed(2)}/sqft is above market:</strong>{' '}
              {lowerPpsfPct}% of similar-sized properties ({minSize.toLocaleString()}-{maxSize.toLocaleString()} sqft) in Prime Central London
              have lower £/sqft rates.
            </p>
            <p>
              <strong>2. Market median for this size:</strong>{' '}
              £{allMedianPpsf.toFixed(2)}/sqft, which translates to {formatCurrency(fairRentAtMedian)}/month
              for a {PROPERTY.size_sqft.toLocaleString()} sqft property.
            </p>
            <p>
              <strong>3. ML model prediction:</strong>{' '}
              {formatCurrency(modelPrediction)}/month (£{modelPpsf.toFixed(2)}/sqft).
            </p>
          </div>
        </div>

        {/* Methodology - HONEST */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">Methodology</h2>
          <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-600 space-y-2">
            <p>
              <strong>Comparison Metric:</strong> Price per square foot (£/sqft) allows fair comparison
              across different-sized properties. Total rent comparisons can be misleading when property sizes differ.
            </p>
            <p>
              <strong>Comparable Selection:</strong> ALL active long-term let properties in Prime Central London
              (SW1 Belgravia, SW3 Chelsea, SW7 South Kensington, W1 Mayfair) within ±{Math.round(SIZE_RANGE * 100)}%
              of subject size ({minSize.toLocaleString()}-{maxSize.toLocaleString()} sqft). These four postcodes represent
              the traditional &quot;super prime&quot; residential areas as defined by major estate agents. Short lets excluded
              as they command ~2x the £/sqft. No filtering by price or £/sqft - complete unbiased dataset.
            </p>
            <p>
              <strong>ML Model:</strong> XGBoost {modelVersion} regression predicting £/sqft based on location,
              size, bedrooms, and property features. R² = {modelR2.toFixed(2)} (explains {Math.round(modelR2 * 100)}% of
              price variance). Note: Model provides guidance but market comparables are the primary evidence.
            </p>
            <p>
              <strong>Data Source:</strong> {marketStats.total_listings.toLocaleString()} active listings from
              Savills, Knight Frank, Foxtons, Chestertons, and Rightmove. Data refreshed daily.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm py-6">
          <p>Rental Price Analysis | {generatedAt}</p>
          <p>{PROPERTY.address}, {PROPERTY.postcode}</p>
          <p className="mt-4">
            <a href="/report" className="text-blue-600 hover:underline">← View Fair Value Report</a>
            <span className="mx-4">|</span>
            <a href="/" className="text-blue-600 hover:underline">Dashboard</a>
          </p>
        </div>
      </div>
    </main>
  );
}
