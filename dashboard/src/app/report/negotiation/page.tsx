import { getComparables, getMarketStats, getPpsfDistribution, getPpsfByDistrict, getLatestValuation, Comparable } from '@/lib/db';
import ReportCharts from '../ReportCharts';

// Subject property details - 4 South Eaton Place
const PROPERTY = {
  address: 'Flat 2, 4 South Eaton Place',
  postcode: 'SW1W',
  size_sqft: 1312,
  bedrooms: 2,
  bathrooms: 2,
  propertyType: 'Maisonette (First & Second Floor Duplex)',
  epcRating: 'C',
};

// Landlord's asking price (new bid)
const LANDLORD_PRICE = 10914;
const LANDLORD_PPSF = LANDLORD_PRICE / PROPERTY.size_sqft;

// Subject property amenities - VERIFIED from listing
const SUBJECT_AMENITIES = {
  // Premium Features
  airConditioning: { has: true, label: 'Air Conditioning', premium: '+5%', importance: 'High', note: 'Throughout property' },
  lift: { has: true, label: 'Lift', premium: '+3%', importance: 'High', note: 'Building lift' },

  // Outdoor Space
  balconies: { has: true, label: 'Private Balconies', premium: '+3%', importance: 'Medium', note: '2 balconies (reception + kitchen)' },
  terrace: { has: true, label: 'Private Terrace', premium: '+3%', importance: 'Medium', note: '' },
  communalGarden: { has: true, label: 'Communal Garden', premium: '+2%', importance: 'Medium', note: 'Belgrave Square (by application)' },
  gardenViews: { has: true, label: 'Garden Views', premium: '+1%', importance: 'Low', note: 'From master suite' },

  // Interior Features
  highCeilings: { has: true, label: 'High Ceilings', premium: '+2%', importance: 'Medium', note: 'Period feature' },
  periodFeatures: { has: true, label: 'Period Features', premium: '+2%', importance: 'Medium', note: 'Original details' },
  woodFlooring: { has: true, label: 'Wood Flooring', premium: '+1%', importance: 'Low', note: 'Rich walnut throughout' },
  furnished: { has: true, label: 'Furnished', premium: '0%', importance: 'Standard', note: 'Contemporary style' },

  // NOT Present - Key Missing Features
  porter: { has: false, label: 'Porter/Concierge', premium: '-', importance: 'Very High', note: 'Not mentioned in listing' },
  gym: { has: false, label: 'Building Gym', premium: '-', importance: 'Very High', note: 'Not available' },
  pool: { has: false, label: 'Swimming Pool', premium: '-', importance: 'Very High', note: 'Not available' },
  parking: { has: false, label: 'Parking', premium: '-', importance: 'Medium', note: 'Not included' },
};

// Luxury model feature importance (£6K+ segment)
const LUXURY_FEATURE_IMPORTANCE = [
  { feature: 'Swimming Pool', allMarket: 0.012, luxury: 0.060, change: '+424%', hasIt: false },
  { feature: 'Building Gym', allMarket: 0.006, luxury: 0.027, change: '+318%', hasIt: false },
  { feature: 'Outdoor Space (Prime)', allMarket: 0.010, luxury: 0.029, change: '+176%', hasIt: true },
  { feature: 'Porter/Concierge', allMarket: 0.013, luxury: 0.035, change: '+163%', hasIt: false },
  { feature: 'Lift', allMarket: 0.012, luxury: 0.032, change: '+157%', hasIt: true },
  { feature: 'Air Conditioning', allMarket: 0.011, luxury: 0.026, change: '+145%', hasIt: true },
  { feature: 'Balcony', allMarket: 0.011, luxury: 0.019, change: '+68%', hasIt: true },
];

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

function getPercentile(values: number[], value: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const below = sorted.filter(v => v < value).length;
  return Math.round((below / sorted.length) * 100);
}

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export default async function NegotiationReport() {
  // Get model's fair value prediction
  let modelPrediction = 7908; // Luxury model with amenities
  let modelVersion = 'V15-Luxury';
  let modelR2 = 0.42; // Luxury segment R²

  try {
    const dbValuation = await getLatestValuation(PROPERTY.address);
    if (dbValuation) {
      // Use luxury-adjusted prediction
      modelPrediction = Math.round(dbValuation.predicted_pcm * 1.05); // AC premium
    }
  } catch {
    // Use fallback
  }

  const modelPpsf = modelPrediction / PROPERTY.size_sqft;
  const postcodeArea = 'SW1';

  let comparables: Comparable[] = [];
  let marketStats = { total_listings: 0, median_ppsf: 0, p25_ppsf: 0, p75_ppsf: 0, avg_ppsf: 0 };
  let ppsfDistribution: { bucket: number; count: number }[] = [];
  let ppsfByDistrict: { district: string; median_ppsf: number; count: number }[] = [];
  let error = null;

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

  // Add tier info to comparables
  const comparablesWithTier = comparables.map(comp => ({
    ...comp,
    tier: getTier(comp.district, PROPERTY.postcode),
  }));

  // Calculate statistics
  const minSize = Math.floor(PROPERTY.size_sqft * (1 - SIZE_RANGE));
  const maxSize = Math.ceil(PROPERTY.size_sqft * (1 + SIZE_RANGE));

  const ppsfValues = comparables.map(c => c.ppsf);
  const priceValues = comparables.map(c => c.price_pcm);

  const getMedian = (arr: number[]) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  };

  const getPercentileValue = (arr: number[], pct: number) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = Math.floor(sorted.length * pct / 100);
    return sorted[Math.min(idx, sorted.length - 1)];
  };

  const medianPpsf = getMedian(ppsfValues);
  const p25Ppsf = getPercentileValue(ppsfValues, 25);
  const p75Ppsf = getPercentileValue(ppsfValues, 75);
  const p90Ppsf = getPercentileValue(ppsfValues, 90);

  const medianPrice = getMedian(priceValues);
  const p75Price = getPercentileValue(priceValues, 75);
  const p90Price = getPercentileValue(priceValues, 90);

  const landlordPercentile = getPercentile(ppsfValues, LANDLORD_PPSF);
  const lowerCount = comparables.filter(c => c.ppsf < LANDLORD_PPSF).length;

  // Fair value calculations
  const fairValueMedian = Math.round(medianPpsf * PROPERTY.size_sqft);
  const fairValue75th = Math.round(p75Ppsf * PROPERTY.size_sqft);

  // Amenity-adjusted fair value (considering what we HAVE)
  const amenityPremium = 0.12; // ~12% for AC + lift + outdoor space + period features
  const amenityAdjustedFair = Math.round(fairValueMedian * (1 + amenityPremium));
  const amenityAdjusted75th = Math.round(fairValue75th * (1 + amenityPremium * 0.5)); // Less premium at higher end

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
        <div className="bg-gradient-to-r from-indigo-800 to-indigo-700 text-white rounded-xl p-6 md:p-8 mb-6">
          <div className="text-center">
            <h1 className="text-2xl md:text-3xl font-bold mb-2">Rental Negotiation Analysis</h1>
            <p className="text-lg opacity-90">{PROPERTY.address}, London {PROPERTY.postcode}</p>
            <p className="text-sm opacity-75 mt-2">{PROPERTY.propertyType} | {PROPERTY.size_sqft.toLocaleString()} sqft</p>
          </div>
          <div className="mt-6 grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-3xl font-bold">{formatCurrency(LANDLORD_PRICE)}</div>
              <div className="text-sm opacity-75">Landlord Asking</div>
            </div>
            <div>
              <div className="text-3xl font-bold">{formatCurrency(amenityAdjustedFair)}</div>
              <div className="text-sm opacity-75">Fair Value (Adjusted)</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-amber-300">+{Math.round((LANDLORD_PRICE / amenityAdjustedFair - 1) * 100)}%</div>
              <div className="text-sm opacity-75">Premium Requested</div>
            </div>
          </div>
        </div>

        {/* Subject Property Amenities */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">Subject Property Amenities</h2>
          <p className="text-sm text-gray-600 mb-4">
            Verified features from the property listing that affect valuation.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Features Present */}
            <div>
              <h3 className="font-semibold text-green-700 mb-3 flex items-center">
                <span className="mr-2">✓</span> Features Present
              </h3>
              <div className="space-y-2">
                {Object.entries(SUBJECT_AMENITIES)
                  .filter(([_, v]) => v.has)
                  .map(([key, amenity]) => (
                    <div key={key} className="flex items-center justify-between p-2 bg-green-50 rounded">
                      <div>
                        <span className="font-medium">{amenity.label}</span>
                        {amenity.note && <span className="text-xs text-gray-500 ml-2">({amenity.note})</span>}
                      </div>
                      <span className="text-green-700 font-semibold text-sm">{amenity.premium}</span>
                    </div>
                  ))}
              </div>
            </div>

            {/* Features Missing */}
            <div>
              <h3 className="font-semibold text-red-700 mb-3 flex items-center">
                <span className="mr-2">✗</span> Features Missing (High Impact in Luxury Segment)
              </h3>
              <div className="space-y-2">
                {Object.entries(SUBJECT_AMENITIES)
                  .filter(([_, v]) => !v.has)
                  .map(([key, amenity]) => (
                    <div key={key} className="flex items-center justify-between p-2 bg-red-50 rounded">
                      <div>
                        <span className="font-medium">{amenity.label}</span>
                        {amenity.note && <span className="text-xs text-gray-500 ml-2">({amenity.note})</span>}
                      </div>
                      <span className="text-red-600 font-semibold text-sm">{amenity.importance} importance</span>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
            <p className="text-sm text-amber-800">
              <strong>Key Insight:</strong> This property has excellent features (AC, lift, multiple outdoor spaces, period details)
              but lacks the ultra-premium amenities (pool, gym, porter) that justify top-tier pricing in the £6K+ segment.
            </p>
          </div>
        </div>

        {/* Luxury Market Feature Importance */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">Luxury Market Feature Importance</h2>
          <p className="text-sm text-gray-600 mb-4">
            How feature importance changes in the £6,000+/month segment vs. the general market.
            Features that matter MORE in luxury properties are highlighted.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-left border-b">
                  <th className="pb-2">Feature</th>
                  <th className="pb-2 text-right">All Market</th>
                  <th className="pb-2 text-right">Luxury (£6K+)</th>
                  <th className="pb-2 text-right">Change</th>
                  <th className="pb-2 text-center">This Property</th>
                </tr>
              </thead>
              <tbody>
                {LUXURY_FEATURE_IMPORTANCE.map((item, idx) => (
                  <tr key={idx} className={item.hasIt ? 'bg-green-50' : 'bg-red-50'}>
                    <td className="py-2 font-medium">{item.feature}</td>
                    <td className="py-2 text-right">{item.allMarket.toFixed(3)}</td>
                    <td className="py-2 text-right font-semibold">{item.luxury.toFixed(3)}</td>
                    <td className="py-2 text-right text-indigo-600 font-semibold">{item.change}</td>
                    <td className="py-2 text-center">
                      {item.hasIt ? (
                        <span className="text-green-600 font-bold">✓ YES</span>
                      ) : (
                        <span className="text-red-600 font-bold">✗ NO</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 p-3 bg-slate-50 rounded-lg">
            <p className="text-sm text-gray-700">
              <strong>Model Finding:</strong> Pool importance increases by +424% in the luxury segment.
              Properties WITHOUT pool/gym/porter cannot command the same premium as those with these amenities.
              This property has 4 of 7 key luxury features but is missing the top 3 (pool, gym, porter).
            </p>
          </div>
        </div>

        {/* Price Comparison */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">Price Analysis</h2>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4 text-center">
              <div className="text-sm text-red-600 font-semibold mb-1">LANDLORD ASKING</div>
              <div className="text-2xl font-bold text-red-700">£{LANDLORD_PPSF.toFixed(2)}</div>
              <div className="text-sm text-gray-600">per sqft/month</div>
              <div className="text-lg font-semibold mt-2">{formatCurrency(LANDLORD_PRICE)}/mo</div>
            </div>

            <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4 text-center">
              <div className="text-sm text-blue-600 font-semibold mb-1">MARKET MEDIAN</div>
              <div className="text-2xl font-bold text-blue-700">£{medianPpsf.toFixed(2)}</div>
              <div className="text-sm text-gray-600">per sqft/month</div>
              <div className="text-lg font-semibold mt-2">{formatCurrency(fairValueMedian)}/mo</div>
            </div>

            <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4 text-center">
              <div className="text-sm text-purple-600 font-semibold mb-1">75TH PERCENTILE</div>
              <div className="text-2xl font-bold text-purple-700">£{p75Ppsf.toFixed(2)}</div>
              <div className="text-sm text-gray-600">per sqft/month</div>
              <div className="text-lg font-semibold mt-2">{formatCurrency(fairValue75th)}/mo</div>
            </div>

            <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4 text-center">
              <div className="text-sm text-green-600 font-semibold mb-1">AMENITY-ADJUSTED FAIR</div>
              <div className="text-2xl font-bold text-green-700">£{(amenityAdjustedFair / PROPERTY.size_sqft).toFixed(2)}</div>
              <div className="text-sm text-gray-600">per sqft/month</div>
              <div className="text-lg font-semibold mt-2">{formatCurrency(amenityAdjustedFair)}/mo</div>
            </div>
          </div>

          <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
            <p className="text-sm text-amber-800">
              <strong>Analysis:</strong> The landlord&apos;s asking price of £{LANDLORD_PPSF.toFixed(2)}/sqft places this property
              at the <strong>{100 - landlordPercentile}th percentile</strong> of the market.
              {lowerCount} of {comparables.length} comparable properties ({Math.round(lowerCount/comparables.length*100)}%) have lower £/sqft rates.
              Given the property&apos;s amenities (AC, lift, outdoor space) but lack of pool/gym/porter,
              fair value is estimated at <strong>{formatCurrency(amenityAdjustedFair)}-{formatCurrency(amenityAdjusted75th)}/month</strong>.
            </p>
          </div>
        </div>

        {/* Recommended Negotiation Range */}
        <div className="bg-gradient-to-r from-green-700 to-green-600 text-white rounded-xl p-6 mb-6">
          <h2 className="text-xl font-bold mb-4 text-center">Recommended Negotiation Range</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white/10 rounded-lg p-4 text-center">
              <div className="text-sm opacity-75 mb-1">OPENING OFFER</div>
              <div className="text-3xl font-bold">{formatCurrency(Math.round(amenityAdjustedFair * 0.95))}</div>
              <div className="text-sm opacity-75 mt-1">£{(amenityAdjustedFair * 0.95 / PROPERTY.size_sqft).toFixed(2)}/sqft</div>
            </div>

            <div className="bg-white/20 rounded-lg p-4 text-center border-2 border-white/50">
              <div className="text-sm opacity-75 mb-1">TARGET</div>
              <div className="text-3xl font-bold">{formatCurrency(Math.round((amenityAdjustedFair + amenityAdjusted75th) / 2))}</div>
              <div className="text-sm opacity-75 mt-1">£{((amenityAdjustedFair + amenityAdjusted75th) / 2 / PROPERTY.size_sqft).toFixed(2)}/sqft</div>
            </div>

            <div className="bg-white/10 rounded-lg p-4 text-center">
              <div className="text-sm opacity-75 mb-1">WALK-AWAY MAX</div>
              <div className="text-3xl font-bold">{formatCurrency(amenityAdjusted75th)}</div>
              <div className="text-sm opacity-75 mt-1">£{(amenityAdjusted75th / PROPERTY.size_sqft).toFixed(2)}/sqft</div>
            </div>
          </div>

          <div className="text-center text-sm opacity-90">
            <p>Landlord asking {formatCurrency(LANDLORD_PRICE)} is <strong>+{Math.round((LANDLORD_PRICE / ((amenityAdjustedFair + amenityAdjusted75th) / 2) - 1) * 100)}%</strong> above target.</p>
            <p className="mt-1">Data-backed negotiation room: <strong>{formatCurrency(LANDLORD_PRICE - amenityAdjusted75th)}/month</strong> potential savings.</p>
          </div>
        </div>

        {/* Supporting Comparable Properties */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          {(() => {
            // Fair value £/sqft range (model fair value ± 15%)
            const fairPpsf = amenityAdjustedFair / PROPERTY.size_sqft;
            const upperBoundPpsf = amenityAdjusted75th / PROPERTY.size_sqft;

            // Filter to properties WITHIN or BELOW fair value range (supporting evidence)
            const supportingComps = comparables
              .filter(c => c.ppsf <= upperBoundPpsf * 1.05) // Within 5% of 75th percentile fair value
              .sort((a, b) => b.ppsf - a.ppsf); // Highest £/sqft first (strongest comps near top)

            return (
              <>
                <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">
                  Supporting Comparables - Within Fair Value Range
                </h2>
                <div className="bg-green-50 rounded-lg p-4 mb-4">
                  <p className="text-sm text-green-800">
                    <strong>Fair Value Range:</strong> £{fairPpsf.toFixed(2)} - £{upperBoundPpsf.toFixed(2)}/sqft
                    ({formatCurrency(amenityAdjustedFair)} - {formatCurrency(amenityAdjusted75th)}/month for {PROPERTY.size_sqft} sqft)
                  </p>
                  <p className="text-sm text-green-700 mt-1">
                    <strong>{supportingComps.length}</strong> of {comparables.length} similar properties support this fair value range.
                  </p>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-gray-500 text-left border-b">
                        <th className="pb-2">#</th>
                        <th className="pb-2">Address</th>
                        <th className="pb-2">Area</th>
                        <th className="pb-2 text-right">Size</th>
                        <th className="pb-2 text-right">Rent/month</th>
                        <th className="pb-2 text-right">£/sqft</th>
                        <th className="pb-2 text-right">vs Fair Value</th>
                        <th className="pb-2">Source</th>
                      </tr>
                    </thead>
                    <tbody>
                      {supportingComps.slice(0, 20).map((comp, idx) => {
                        const diffFromFair = comp.ppsf - fairPpsf;
                        const pctFromFair = (comp.ppsf / fairPpsf - 1) * 100;
                        return (
                          <tr key={idx} className="bg-green-50 hover:bg-green-100">
                            <td className="py-2 text-gray-400 text-xs">{idx + 1}</td>
                            <td className="py-2">
                              <a href={comp.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                                {comp.address?.slice(0, 35)}{(comp.address?.length || 0) > 35 ? '...' : ''}
                              </a>
                            </td>
                            <td className="py-2">{comp.district}</td>
                            <td className="py-2 text-right">{comp.size_sqft.toLocaleString()}</td>
                            <td className="py-2 text-right font-semibold">{formatCurrency(comp.price_pcm)}</td>
                            <td className="py-2 text-right font-semibold">£{comp.ppsf.toFixed(2)}</td>
                            <td className="py-2 text-right">
                              <span className={pctFromFair <= 0 ? 'text-green-700' : 'text-amber-600'}>
                                {pctFromFair >= 0 ? '+' : ''}{pctFromFair.toFixed(0)}%
                              </span>
                            </td>
                            <td className="py-2 text-xs">{comp.source}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {supportingComps.length > 20 && (
                  <p className="text-xs text-gray-500 mt-2">
                    Showing top 20 of {supportingComps.length} supporting comparables.
                  </p>
                )}

                <div className="mt-4 p-3 bg-amber-50 rounded-lg border border-amber-200">
                  <p className="text-sm text-amber-800">
                    <strong>Key Point:</strong> The landlord's asking price of {formatCurrency(LANDLORD_PRICE)} (£{LANDLORD_PPSF.toFixed(2)}/sqft)
                    is <strong>£{(LANDLORD_PPSF - upperBoundPpsf).toFixed(2)}/sqft higher</strong> than even the 75th percentile
                    of similar properties in Prime Central London.
                  </p>
                </div>
              </>
            );
          })()}
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

        {/* Methodology */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <h2 className="text-lg font-semibold border-b pb-2 mb-4 text-slate-800">Methodology</h2>
          <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-600 space-y-3">
            <p>
              <strong>Comparable Selection:</strong> {comparables.length} active long-term let properties in Prime Central London
              (SW1 Belgravia, SW3 Chelsea, SW7 South Kensington, W1 Mayfair) within ±{Math.round(SIZE_RANGE * 100)}%
              of subject size ({minSize.toLocaleString()}-{maxSize.toLocaleString()} sqft) with 2 bedrooms.
            </p>
            <p>
              <strong>Luxury Model (V15):</strong> XGBoost regression trained on £6,000+/month properties only.
              R² = {(modelR2).toFixed(2)} for luxury segment (lower than general market due to heterogeneity).
              Key finding: Pool (+424%), Gym (+318%), and Porter (+163%) importance increases dramatically in this segment.
            </p>
            <p>
              <strong>Amenity Adjustment:</strong> +12% premium applied to median for verified features
              (AC, lift, dual balconies, terrace, communal garden, period features).
              No premium for missing ultra-luxury features (pool, gym, porter).
            </p>
            <p>
              <strong>Data Source:</strong> {marketStats.total_listings.toLocaleString()} active listings from
              Savills, Knight Frank, Foxtons, Chestertons, and Rightmove. Refreshed daily.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm py-6">
          <p>Rental Negotiation Analysis | Generated: {generatedAt}</p>
          <p>{PROPERTY.address}, {PROPERTY.postcode} | {PROPERTY.size_sqft.toLocaleString()} sqft {PROPERTY.propertyType}</p>
          <p className="mt-4">
            <a href="/report/landlord-price" className="text-blue-600 hover:underline">← Basic Report</a>
            <span className="mx-4">|</span>
            <a href="/report" className="text-blue-600 hover:underline">Fair Value Report</a>
            <span className="mx-4">|</span>
            <a href="/" className="text-blue-600 hover:underline">Dashboard</a>
          </p>
        </div>
      </div>
    </main>
  );
}
