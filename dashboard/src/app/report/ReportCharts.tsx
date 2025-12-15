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

const TIER_COLORS: Record<number, string> = {
  1: '#2E7D32',
  2: '#1976D2',
  3: '#F57C00',
  4: '#757575',
};

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

  // Format district data for bar chart
  const districtData = ppsfByDistrict.map(d => ({
    district: d.district,
    median_ppsf: d.median_ppsf,
    count: d.count,
    isPrime: ['SW1', 'SW3', 'SW7', 'W1', 'W8', 'NW1', 'NW3', 'NW8'].some(p => d.district.startsWith(p)),
    isSubject: d.district === subjectDistrict,
  }));

  // Format scatter data for size vs price - only Tier 1 & 2 for relevance
  const scatterData = comparables
    .filter(c => c.tier.num <= 2)
    .map(c => ({
      size: c.size_sqft,
      price: c.price_pcm,
      tier: c.tier.num,
      address: c.address,
      ppsf: c.ppsf,
    }));

  // Top comparables for horizontal bar chart - only Tier 1 & 2 for relevance
  const topComparables = comparables
    .filter(c => c.tier.num <= 2)
    .slice(0, 15)
    .map(c => ({
      address: c.address?.slice(0, 30) + (c.address?.length > 30 ? '...' : ''),
      price: c.price_pcm,
      ppsf: c.ppsf,
      tier: c.tier.num,
    }));

  return (
    <div className="space-y-8">
      {/* PPSF Distribution */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Price per Sqft Distribution - London Rentals</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={distributionData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="ppsf" fontSize={11} />
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
              label={{ value: `Subject: £${subjectPpsf.toFixed(2)}`, fill: '#dc2626', fontSize: 11 }}
            />
            <Bar dataKey="count" fill="#4682B4" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Size vs Price Scatter - Best Comparables Only */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Size vs Monthly Rent - Best Comparables (Tier 1 & 2)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="size"
              type="number"
              name="Size"
              fontSize={11}
              tickFormatter={(v) => `${v} sqft`}
              domain={['auto', 'auto']}
            />
            <YAxis
              dataKey="price"
              type="number"
              name="Rent"
              fontSize={11}
              tickFormatter={(v) => `£${v.toLocaleString()}`}
            />
            <Tooltip
              formatter={(value, name) =>
                name === 'price' ? [`£${(value as number)?.toLocaleString()}`, 'Rent'] : [`${value} sqft`, 'Size']
              }
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  // Handle subject marker (has isSubject flag, no ppsf)
                  if (data.isSubject) {
                    return (
                      <div className="bg-white border shadow-lg p-2 text-xs">
                        <p className="font-semibold text-red-600">Subject Property</p>
                        <p>Size: {data.size} sqft</p>
                        <p>Rent: £{data.price.toLocaleString()}</p>
                        <p>£/sqft: £{(data.price / data.size).toFixed(2)}</p>
                      </div>
                    );
                  }
                  return (
                    <div className="bg-white border shadow-lg p-2 text-xs">
                      <p className="font-semibold">{data.address || 'Unknown'}</p>
                      <p>Size: {data.size} sqft</p>
                      <p>Rent: £{data.price?.toLocaleString() || 'N/A'}</p>
                      <p>£/sqft: £{data.ppsf?.toFixed(2) || (data.price && data.size ? (data.price / data.size).toFixed(2) : 'N/A')}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Scatter name="Comparables" data={scatterData}>
              {scatterData.map((entry, index) => (
                <Cell key={index} fill={TIER_COLORS[entry.tier] || '#757575'} />
              ))}
            </Scatter>
            {/* Subject property marker */}
            <Scatter
              name="Subject"
              data={[{ size: subjectSize, price: subjectPrice, isSubject: true }]}
              fill="#dc2626"
              shape="star"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* District Comparison */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Median £/sqft by District (Prime areas highlighted)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={districtData} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="district" fontSize={10} angle={-45} textAnchor="end" height={60} />
            <YAxis fontSize={11} tickFormatter={(v) => `£${v}`} />
            <Tooltip
              formatter={(value) => [`£${(value as number)?.toFixed(2)}`, 'Median £/sqft']}
              labelFormatter={(label) => `District: ${label}`}
            />
            <ReferenceLine
              y={subjectPpsf}
              stroke="#dc2626"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: 'Subject', fill: '#dc2626', fontSize: 10 }}
            />
            <Bar dataKey="median_ppsf">
              {districtData.map((entry, index) => (
                <Cell
                  key={index}
                  fill={entry.isSubject ? '#dc2626' : entry.isPrime ? '#f87171' : '#93c5fd'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Top Comparables */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Top Comparables by Price</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={topComparables} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" fontSize={11} tickFormatter={(v) => `£${v.toLocaleString()}`} />
            <YAxis dataKey="address" type="category" width={150} fontSize={10} />
            <Tooltip
              formatter={(value) => [`£${(value as number)?.toLocaleString()}`, 'Monthly Rent']}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white border shadow-lg p-2 text-xs">
                      <p className="font-semibold">{data.address}</p>
                      <p>Rent: £{data.price.toLocaleString()}</p>
                      <p>£/sqft: £{data.ppsf.toFixed(2)}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <ReferenceLine
              x={subjectPrice}
              stroke="#dc2626"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: `Valuation: £${subjectPrice.toLocaleString()}`, fill: '#dc2626', fontSize: 10 }}
            />
            <Bar dataKey="price">
              {topComparables.map((entry, index) => (
                <Cell key={index} fill={TIER_COLORS[entry.tier] || '#757575'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="flex gap-4 justify-center mt-2 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded" style={{ backgroundColor: TIER_COLORS[1] }}></span>
            T1: Same District
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded" style={{ backgroundColor: TIER_COLORS[2] }}></span>
            T2: SW1 Area
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded" style={{ backgroundColor: TIER_COLORS[3] }}></span>
            T3: Prime Central
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded" style={{ backgroundColor: TIER_COLORS[4] }}></span>
            T4: Broader Market
          </span>
        </div>
      </div>
    </div>
  );
}
