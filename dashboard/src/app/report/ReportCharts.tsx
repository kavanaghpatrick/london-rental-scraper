'use client';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ScatterChart,
  Scatter,
  Cell,
} from 'recharts';

interface PpsfDistribution {
  bucket: number;
  count: number;
}

interface DistrictData {
  district: string;
  median_ppsf: number;
  count: number;
}

interface Comparable {
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
  ppsf_diff: number;
  size_diff_pct: number;
  tier: { num: number; label: string };
}

interface ReportChartsProps {
  ppsfDistribution: PpsfDistribution[];
  ppsfByDistrict: DistrictData[];
  comparables: Comparable[];
  subjectPpsf: number;
  subjectDistrict: string;
  subjectPrice: number;
  subjectSize: number;
}

// Prime Central London postcodes
const PRIME_DISTRICTS = new Set(['SW1A', 'SW1E', 'SW1H', 'SW1P', 'SW1V', 'SW1W', 'SW1X', 'SW1Y', 'SW3', 'SW7', 'W1']);

function isPrimeDistrict(district: string): boolean {
  if (!district) return false;
  if (PRIME_DISTRICTS.has(district)) return true;
  // Check for W1 variants (W1K, W1H, etc.)
  if (district.startsWith('W1')) return true;
  // Check for SW1 variants
  if (/^SW1[A-Z]$/.test(district)) return true;
  return false;
}

export default function ReportCharts({
  ppsfDistribution,
  ppsfByDistrict,
  comparables,
  subjectPpsf,
  subjectDistrict,
  subjectPrice,
  subjectSize,
}: ReportChartsProps) {
  // Format distribution data for bar chart
  const distributionData = ppsfDistribution.map(d => ({
    ppsf: `£${d.bucket}`,
    bucket: d.bucket,
    count: d.count,
  }));

  // Get Prime Central London districts only
  const primeDistricts = ppsfByDistrict.filter(d => isPrimeDistrict(d.district));

  const districtData = primeDistricts.map(d => ({
    district: d.district,
    median_ppsf: d.median_ppsf,
    count: d.count,
    isSubject: d.district === subjectDistrict || d.district.startsWith(subjectDistrict.slice(0, 3)),
  }));

  // Sort ALL comparables by £/sqft (ascending - cheapest first)
  const sortedByPpsf = [...comparables].sort((a, b) => a.ppsf - b.ppsf);

  // Show up to 20 properties in bar chart for better visualization
  const topComparables = sortedByPpsf
    .slice(0, 20)
    .map(c => ({
      address: c.address?.slice(0, 22) + (c.address?.length > 22 ? '...' : ''),
      fullAddress: c.address,
      district: c.district,
      price: c.price_pcm,
      ppsf: c.ppsf,
      size: c.size_sqft,
      url: c.url,
      isLower: c.ppsf < subjectPpsf,
    }));

  // Scatter data - ALL comparables
  const scatterData = comparables.map(c => ({
    size: c.size_sqft,
    ppsf: c.ppsf,
    price: c.price_pcm,
    address: c.address,
    district: c.district,
  }));

  // Calculate how many are below asking for the caption
  const belowAsking = comparables.filter(c => c.ppsf < subjectPpsf).length;
  const belowPct = Math.round((belowAsking / comparables.length) * 100);

  // Handle bar click to open property URL
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleBarClick = (data: any) => {
    if (data?.url) {
      window.open(data.url, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <div className="space-y-8">
      {/* Chart 1: Comparables by £/sqft - THE KEY CHART */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-1">
          Top 20 Comparable Properties by £/sqft
        </h3>
        <p className="text-xs text-gray-500 mb-3">
          Prime Central London properties sorted by £/sqft (lowest first). Green = below asking, Red = above asking.
        </p>
        <ResponsiveContainer width="100%" height={500}>
          <BarChart data={topComparables} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              fontSize={11}
              tickFormatter={(v) => `£${v.toFixed(2)}`}
              domain={[0, 'auto']}
            />
            <YAxis dataKey="address" type="category" width={130} fontSize={8} />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  const diff = subjectPpsf - data.ppsf;
                  return (
                    <div className="bg-white border shadow-lg p-2 text-xs">
                      <p className="font-semibold">{data.fullAddress || data.address}</p>
                      <p className="text-gray-500">{data.district}</p>
                      <p>Size: {data.size?.toLocaleString()} sqft</p>
                      <p>Rent: £{data.price?.toLocaleString()}/month</p>
                      <p className="font-semibold">£/sqft: £{data.ppsf?.toFixed(2)}</p>
                      {diff > 0 && (
                        <p className="text-green-600 font-semibold mt-1">
                          £{diff.toFixed(2)}/sqft cheaper than asking
                        </p>
                      )}
                      <p className="text-blue-600 mt-1">Click to view listing →</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <ReferenceLine
              x={subjectPpsf}
              stroke="#dc2626"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: `Asking: £${subjectPpsf.toFixed(2)}`, fill: '#dc2626', fontSize: 10 }}
            />
            <Bar dataKey="ppsf" onClick={handleBarClick} style={{ cursor: 'pointer' }}>
              {topComparables.map((entry, index) => (
                <Cell
                  key={index}
                  fill={entry.isLower ? '#22c55e' : '#ef4444'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Click any bar to view the listing. Showing top 20 of {comparables.length} comparables from Belgravia, Chelsea, South Kensington & Mayfair.
        </p>
      </div>

      {/* Chart 2: £/sqft vs Size Scatter - Shows all comparables */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-1">
          £/sqft vs Property Size ({comparables.length} properties)
        </h3>
        <p className="text-xs text-gray-500 mb-3">
          Each dot is a comparable property. {belowPct}% ({belowAsking} of {comparables.length}) have lower £/sqft than asking.
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="size"
              type="number"
              name="Size"
              fontSize={11}
              tickFormatter={(v) => `${v.toLocaleString()}`}
              domain={['auto', 'auto']}
              label={{ value: 'Size (sqft)', position: 'bottom', offset: -5, fontSize: 10 }}
            />
            <YAxis
              dataKey="ppsf"
              type="number"
              name="£/sqft"
              fontSize={11}
              tickFormatter={(v) => `£${v.toFixed(1)}`}
              domain={[0, 'auto']}
              label={{ value: '£/sqft', angle: -90, position: 'insideLeft', fontSize: 10 }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  if (data.isSubject) {
                    return (
                      <div className="bg-white border shadow-lg p-2 text-xs">
                        <p className="font-semibold text-red-600">Landlord&apos;s Asking Price</p>
                        <p>Size: {data.size?.toLocaleString()} sqft</p>
                        <p className="font-semibold">£/sqft: £{data.ppsf?.toFixed(2)}</p>
                      </div>
                    );
                  }
                  return (
                    <div className="bg-white border shadow-lg p-2 text-xs">
                      <p className="font-semibold">{data.address || 'Comparable'}</p>
                      <p className="text-gray-500">{data.district}</p>
                      <p>Size: {data.size?.toLocaleString()} sqft</p>
                      <p>Rent: £{data.price?.toLocaleString()}/month</p>
                      <p className="font-semibold">£/sqft: £{data.ppsf?.toFixed(2)}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <ReferenceLine
              y={subjectPpsf}
              stroke="#dc2626"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: `Asking: £${subjectPpsf.toFixed(2)}`, fill: '#dc2626', fontSize: 10, position: 'right' }}
            />
            <Scatter name="Comparables" data={scatterData}>
              {scatterData.map((entry, index) => (
                <Cell key={index} fill={entry.ppsf < subjectPpsf ? '#22c55e' : '#ef4444'} />
              ))}
            </Scatter>
            {/* Subject property marker */}
            <Scatter
              name="Asking"
              data={[{ size: subjectSize, ppsf: subjectPpsf, isSubject: true }]}
              fill="#dc2626"
              shape="star"
            />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Red star = landlord&apos;s asking (£{subjectPpsf.toFixed(2)}/sqft).
          Green dots = below asking. Red dots = above asking.
        </p>
      </div>

      {/* Chart 3: District £/sqft Comparison */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-1">
          Median £/sqft by District (Prime Central London)
        </h3>
        <p className="text-xs text-gray-500 mb-3">
          District medians across Belgravia, Chelsea, South Kensington, and Mayfair.
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={districtData} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="district" fontSize={9} angle={-45} textAnchor="end" height={60} />
            <YAxis fontSize={11} tickFormatter={(v) => `£${v}`} domain={[0, 'auto']} />
            <Tooltip
              formatter={(value) => [`£${(value as number)?.toFixed(2)}`, 'Median £/sqft']}
              labelFormatter={(label) => `District: ${label}`}
            />
            <ReferenceLine
              y={subjectPpsf}
              stroke="#dc2626"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: `Asking: £${subjectPpsf.toFixed(2)}`, fill: '#dc2626', fontSize: 10 }}
            />
            <Bar dataKey="median_ppsf">
              {districtData.map((entry, index) => (
                <Cell
                  key={index}
                  fill={entry.median_ppsf < subjectPpsf ? '#22c55e' : '#ef4444'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Green = district median below asking. The asking price of £{subjectPpsf.toFixed(2)}/sqft exceeds most Prime Central London district medians.
        </p>
      </div>

      {/* Chart 4: Market Distribution with Context */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-1">
          London-Wide £/sqft Distribution
        </h3>
        <p className="text-xs text-gray-500 mb-3">
          Where the asking price sits relative to all London rentals (not just Prime Central).
        </p>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={distributionData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="ppsf" fontSize={10} />
            <YAxis fontSize={11} />
            <Tooltip
              formatter={(value) => [value ?? 0, 'Properties']}
              labelFormatter={(label) => `${label}/sqft`}
            />
            <ReferenceLine
              x={`£${Math.floor(subjectPpsf)}`}
              stroke="#dc2626"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: `Asking: £${subjectPpsf.toFixed(2)}`, fill: '#dc2626', fontSize: 10 }}
            />
            <Bar dataKey="count" fill="#94a3b8" />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-500 mt-2 text-center">
          This shows ALL of London. Prime Central commands a premium, but even within Prime Central,
          the asking price exceeds {belowPct}% of comparable properties.
        </p>
      </div>
    </div>
  );
}
