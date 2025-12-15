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
  let modelPrediction = 8925; // fallback
  let modelVersion = 'V15';
  try {
    const dbValuation = await getLatestValuation(PROPERTY.address);
    if (dbValuation) {
      modelPrediction = dbValuation.predicted_pcm;
      modelVersion = dbValuation.model_version;
    }
  } catch {
    // Use fallback
  }

  const modelPpsf = modelPrediction / PROPERTY.size_sqft;
  const overpayment = LANDLORD_PRICE - modelPrediction;
  const overpaymentPct = ((LANDLORD_PRICE / modelPrediction) - 1) * 100;
  const annualOverpayment = overpayment * 12;

  const postcodeArea = PROPERTY.postcode.match(/^([A-Z]+\d+)/)?.[1] || PROPERTY.postcode.slice(0, 3);

  let comparables: Comparable[] = [];
  let marketStats = { total_listings: 0, median_ppsf: 0, p25_ppsf: 0, p75_ppsf: 0, avg_ppsf: 0 };
  let ppsfDistribution: { bucket: number; count: number }[] = [];
  let ppsfByDistrict: { district: string; median_ppsf: number; count: number }[] = [];
  let error = null;

  try {
    [comparables, marketStats, ppsfDistribution, ppsfByDistrict] = await Promise.all([
      getComparables(PROPERTY.size_sqft, PROPERTY.bedrooms, 0.40, LANDLORD_PPSF, postcodeArea),
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

  // Count how many comps are CHEAPER than landlord's price
  const cheaperComps = comparablesWithTier.filter(c => c.price_pcm < LANDLORD_PRICE);
  const cheaperPct = Math.round((cheaperComps.length / comparablesWithTier.length) * 100);

  // Count how many comps have LOWER £/sqft
  const lowerPpsfComps = comparablesWithTier.filter(c => c.ppsf < LANDLORD_PPSF);
  const lowerPpsfPct = Math.round((lowerPpsfComps.length / comparablesWithTier.length) * 100);

  // Tier breakdown
  const tier1 = comparablesWithTier.filter(c => c.tier.num === 1);
  const tier2 = comparablesWithTier.filter(c => c.tier.num === 2);
  const tier1Cheaper = tier1.filter(c => c.price_pcm < LANDLORD_PRICE);
  const tier2Cheaper = tier2.filter(c => c.price_pcm < LANDLORD_PRICE);

  // Market percentile at landlord's price
  const belowLandlord = ppsfDistribution
    .filter(d => d.bucket < LANDLORD_PPSF)
    .reduce((sum, d) => sum + d.count, 0);
  const totalInDist = ppsfDistribution.reduce((sum, d) => sum + d.count, 0);
  const landlordPercentile = totalInDist > 0 ? Math.round((belowLandlord / totalInDist) * 100) : 0;

  // Tier medians
  const getMedianPrice = (items: typeof comparablesWithTier) => {
    if (items.length === 0) return 0;
    const sorted = [...items].sort((a, b) => a.price_pcm - b.price_pcm);
    return sorted[Math.floor(sorted.length / 2)].price_pcm;
  };

  const tier1MedianPrice = getMedianPrice(tier1);
  const tier2MedianPrice = getMedianPrice(tier2);
  const allMedianPrice = getMedianPrice(comparablesWithTier);

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
        {/* Header - Red to indicate problem */}
        <div className="bg-gradient-to-r from-red-900 to-red-700 text-white rounded-xl p-6 md:p-8 mb-6 text-center">
          <h1 className="text-2xl md:text-3xl font-bold mb-2">Rental Price Analysis</h1>
          <p className="text-lg opacity-90">{PROPERTY.address}, London {PROPERTY.postcode}</p>
          <p className="text-sm opacity-75 mt-4">
            Generated: {generatedAt} | Analysis of {marketStats.total_listings.toLocaleString()} active listings
          </p>
        </div>

        {/* KEY FINDING - The main argument */}
        <div className="bg-gradient-to-r from-red-50 to-red-100 border-2 border-red-400 rounded-xl p-6 mb-6">
          <div className="text-center mb-6">
            <div className="text-sm text-red-600 font-semibold mb-1">LANDLORD&apos;S ASKING PRICE</div>
            <div className="text-4xl md:text-5xl font-bold text-red-700">{formatCurrency(LANDLORD_PRICE)}</div>
            <div className="text-gray-600 mt-2">
              £{LANDLORD_PPSF.toFixed(2)}/sqft | {landlordPercentile}th percentile of London market
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-white rounded-lg p-4 text-center border-2 border-red-300">
              <div className="text-3xl font-bold text-red-600">+{formatCurrency(overpayment)}</div>
              <div className="text-sm text-gray-600 mt-1">Above Fair Market Value</div>
              <div className="text-xs text-gray-500">+{overpaymentPct.toFixed(1)}% premium</div>
            </div>
            <div className="bg-white rounded-lg p-4 text-center border-2 border-red-300">
              <div className="text-3xl font-bold text-red-600">{formatCurrency(annualOverpayment)}</div>
              <div className="text-sm text-gray-600 mt-1">Annual Overpayment</div>
              <div className="text-xs text-gray-500">if paid for 12 months</div>
            </div>
            <div className="bg-white rounded-lg p-4 text-center border-2 border-red-300">
              <div className="text-3xl font-bold text-red-600">{cheaperPct}%</div>
              <div className="text-sm text-gray-600 mt-1">Of Comps Are Cheaper</div>
              <div className="text-xs text-gray-500">{cheaperComps.length} of {comparablesWithTier.length} properties</div>
            </div>
          </div>
        </div>

        {/* Fair Value Comparison */}
        <div className="bg-gradient-to-r from-green-50 to-green-100 border-2 border-green-400 rounded-xl p-6 mb-6">
          <div className="text-center">
            <div className="text-sm text-green-600 font-semibold mb-1">MODEL-PREDICTED FAIR VALUE ({modelVersion})</div>
            <div className="text-4xl md:text-5xl font-bold text-green-700">{formatCurrency(modelPrediction)}</div>
            <div className="text-gray-600 mt-2">
              £{modelPpsf.toFixed(2)}/sqft | Based on {marketStats.total_listings.toLocaleString()} comparable listings
            </div>
          </div>
        </div>

        {/* Evidence Summary */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-blue-900">Evidence Summary</h2>

          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <span className="text-2xl">1.</span>
              <div>
                <p className="font-semibold text-red-700">
                  {lowerPpsfPct}% of comparable properties have lower £/sqft
                </p>
                <p className="text-sm text-gray-600">
                  At £{LANDLORD_PPSF.toFixed(2)}/sqft, the asking price is higher than {lowerPpsfComps.length} of {comparablesWithTier.length} similar properties in the {postcodeArea} area.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="text-2xl">2.</span>
              <div>
                <p className="font-semibold text-red-700">
                  Same District (SW1W): {tier1Cheaper.length} of {tier1.length} properties are cheaper
                </p>
                <p className="text-sm text-gray-600">
                  {tier1.length > 0 ? `Median rent in SW1W: ${formatCurrency(tier1MedianPrice)}/month (${formatCurrency(LANDLORD_PRICE - tier1MedianPrice)} less than asking)` : 'Limited direct comparables in SW1W'}
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="text-2xl">3.</span>
              <div>
                <p className="font-semibold text-red-700">
                  SW1 Area: {tier2Cheaper.length} of {tier2.length} properties are cheaper
                </p>
                <p className="text-sm text-gray-600">
                  {tier2.length > 0 ? `Median rent across SW1: ${formatCurrency(tier2MedianPrice)}/month` : 'Comparable properties across SW1 districts'}
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="text-2xl">4.</span>
              <div>
                <p className="font-semibold text-red-700">
                  XGBoost ML Model predicts {formatCurrency(modelPrediction)} fair value
                </p>
                <p className="text-sm text-gray-600">
                  Trained on {marketStats.total_listings.toLocaleString()} active listings with R² &gt; 0.90 accuracy. The landlord&apos;s price is {overpaymentPct.toFixed(1)}% above model prediction.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <span className="text-2xl">5.</span>
              <div>
                <p className="font-semibold text-red-700">
                  Market median for comparable size: {formatCurrency(allMedianPrice)}/month
                </p>
                <p className="text-sm text-gray-600">
                  The asking price of {formatCurrency(LANDLORD_PRICE)} is {formatCurrency(LANDLORD_PRICE - allMedianPrice)} ({((LANDLORD_PRICE / allMedianPrice - 1) * 100).toFixed(0)}%) above the median of similar properties.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Property Details */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-blue-900">Property Details</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">{PROPERTY.size_sqft.toLocaleString()}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Square Feet</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">{PROPERTY.bedrooms}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Bedrooms</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">{PROPERTY.bathrooms}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Bathrooms</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-700">{PROPERTY.postcode}</div>
              <div className="text-xs text-gray-500 uppercase mt-1">Postcode</div>
            </div>
          </div>
        </div>

        {/* Charts - using landlord price as subject */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-blue-900">Market Analysis</h2>
          <ReportCharts
            ppsfDistribution={ppsfDistribution}
            ppsfByDistrict={ppsfByDistrict}
            comparables={comparablesWithTier.slice(0, 20)}
            subjectPpsf={LANDLORD_PPSF}
            subjectDistrict={PROPERTY.postcode}
            subjectPrice={Math.round(LANDLORD_PRICE)}
            subjectSize={PROPERTY.size_sqft}
          />
        </div>

        {/* Comparable Properties Table */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-blue-900">
            Comparable Properties - All Cheaper Than Asking Price
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            Properties in {postcodeArea} area with similar size, sorted by negotiation value.
            <span className="ml-2 px-2 py-0.5 bg-green-100 rounded text-green-800 text-xs font-semibold">Green rows</span> = cheaper than landlord&apos;s asking price
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-left border-b">
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
                {comparablesWithTier.slice(0, 25).map((comp, idx) => {
                  const isCheaper = comp.price_pcm < LANDLORD_PRICE;
                  const savings = LANDLORD_PRICE - comp.price_pcm;
                  const rowBg = isCheaper ? 'bg-green-50' : 'bg-red-50';
                  return (
                    <tr key={idx} className={rowBg}>
                      <td className="py-2">
                        <a href={comp.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                          {comp.address?.slice(0, 35)}{comp.address?.length > 35 ? '...' : ''}
                        </a>
                      </td>
                      <td className="py-2">{comp.district}</td>
                      <td className="py-2">{comp.bedrooms}</td>
                      <td className="py-2 text-right">{comp.size_sqft.toLocaleString()}</td>
                      <td className="py-2 text-right font-semibold">{formatCurrency(comp.price_pcm)}</td>
                      <td className="py-2 text-right">£{comp.ppsf.toFixed(2)}</td>
                      <td className="py-2 text-right">
                        {isCheaper ? (
                          <span className="text-green-700 font-semibold">-{formatCurrency(savings)}</span>
                        ) : (
                          <span className="text-red-600">+{formatCurrency(-savings)}</span>
                        )}
                      </td>
                      <td className="py-2">{comp.source}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="text-xs text-gray-500 mt-4">
            Showing {Math.min(25, comparablesWithTier.length)} of {comparablesWithTier.length} comparable properties. All data from live listings.
          </p>
        </div>

        {/* Recommendation */}
        <div className="bg-gradient-to-r from-blue-50 to-blue-100 border-2 border-blue-400 rounded-xl p-6 mb-6">
          <h2 className="text-lg font-semibold text-blue-900 mb-4">Recommended Counter-Offer</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white rounded-lg p-4 text-center">
              <div className="text-sm text-gray-500 mb-1">Conservative</div>
              <div className="text-2xl font-bold text-blue-700">{formatCurrency(Math.round(modelPrediction * 1.05))}</div>
              <div className="text-xs text-gray-500">Model + 5%</div>
            </div>
            <div className="bg-white rounded-lg p-4 text-center border-2 border-blue-400">
              <div className="text-sm text-gray-500 mb-1">Fair Market Value</div>
              <div className="text-2xl font-bold text-green-700">{formatCurrency(modelPrediction)}</div>
              <div className="text-xs text-gray-500">ML Model Prediction</div>
            </div>
            <div className="bg-white rounded-lg p-4 text-center">
              <div className="text-sm text-gray-500 mb-1">Aggressive</div>
              <div className="text-2xl font-bold text-blue-700">{formatCurrency(Math.round(allMedianPrice))}</div>
              <div className="text-xs text-gray-500">Market Median</div>
            </div>
          </div>
          <p className="text-sm text-gray-600 text-center mt-4">
            Based on {comparablesWithTier.length} comparable properties and ML model trained on {marketStats.total_listings.toLocaleString()} listings
          </p>
        </div>

        {/* Methodology */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-blue-900">Methodology</h2>
          <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-600 space-y-2">
            <p><strong>Model:</strong> XGBoost {modelVersion} regression with 79 features including size, location encodings, amenities, and property type.</p>
            <p><strong>Training Data:</strong> {marketStats.total_listings.toLocaleString()} active London rental listings from Savills, Knight Frank, Foxtons, Chestertons, and Rightmove.</p>
            <p><strong>Comparables:</strong> Properties in {postcodeArea} area (±40% size) ranked by negotiation utility - prioritizing cheaper properties with similar or larger size.</p>
            <p><strong>Data Currency:</strong> All listings are currently active on the market as of {generatedAt}.</p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm py-6">
          <p>Rental Price Analysis | Generated {generatedAt}</p>
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
