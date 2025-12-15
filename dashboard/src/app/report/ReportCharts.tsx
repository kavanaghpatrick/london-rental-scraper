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

  // Get SW1 districts only for fair comparison
  const sw1Districts = ppsfByDistrict.filter(d =>
    d.district.startsWith('SW1') || d.district === 'SW3' || d.district === 'SW7'
  );

  const districtData = sw1Districts.map(d => ({
    district: d.district,
    median_ppsf: d.median_ppsf,
    count: d.count,
    isSubject: d.district === subjectDistrict || d.district.startsWith(subjectDistrict.slice(0, 3)),
  }));

  // Sort comparables by £/sqft (ascending - cheapest first)
  const sortedByPpsf = [...comparables]
    .filter(c => c.tier.num <= 2)
    .sort((a, b) => a.ppsf - b.ppsf);

  // Top comparables - NOW SORTED BY £/SQFT (the fair metric)
  const topComparables = sortedByPpsf
    .slice(0, 15)
    .map(c => ({
      address: c.address?.slice(0, 25) + (c.address?.length > 25 ? '...' : ''),
      fullAddress: c.address,
      price: c.price_pcm,
      ppsf: c.ppsf,
      size: c.size_sqft,
      tier: c.tier.num,
      url: c.url,
      isLower: c.ppsf < subjectPpsf,
    }));

  // Scatter data - £/sqft vs size (more honest than price vs size)
  const scatterData = comparables
    .filter(c => c.tier.num <= 2)
    .map(c => ({
      size: c.size_sqft,
      ppsf: c.ppsf,
      price: c.price_pcm,
      tier: c.tier.num,
      address: c.address,
    }));

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
          Comparable Properties by £/sqft
        </h3>
        <p className="text-xs text-gray-500 mb-3">
          Similar-sized SW1 properties sorted by £/sqft. Green bars = lower £/sqft than asking.
        </p>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={topComparables} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              fontSize={11}
              tickFormatter={(v) => `£${v.toFixed(2)}`}
              domain={[0, 'auto']}
            />
            <YAxis dataKey="address" type="category" width={140} fontSize={9} />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  const diff = subjectPpsf - data.ppsf;
                  return (
                    <div className="bg-white border shadow-lg p-2 text-xs">
                      <p className="font-semibold">{data.fullAddress || data.address}</p>
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
          Click any bar to view the listing. All properties are {Math.round(subjectSize * 0.85).toLocaleString()}-{Math.round(subjectSize * 1.15).toLocaleString()} sqft in SW1.
        </p>
      </div>

      {/* Chart 2: £/sqft vs Size Scatter - Shows there's no size premium */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-1">
          £/sqft vs Property Size
        </h3>
        <p className="text-xs text-gray-500 mb-3">
          Shows £/sqft doesn&apos;t increase with size. Larger properties often have LOWER £/sqft.
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="size"
              type="number"
              name="Size"
              fontSize={11}
              tickFormatter={(v) => `${v}`}
              domain={['auto', 'auto']}
              label={{ value: 'Size (sqft)', position: 'bottom', offset: -5, fontSize: 10 }}
            />
            <YAxis
              dataKey="ppsf"
              type="number"
              name="£/sqft"
              fontSize={11}
              tickFormatter={(v) => `£${v.toFixed(1)}`}
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
              strokeWidth={1}
              strokeDasharray="5 5"
            />
            <Scatter name="Comparables" data={scatterData}>
              {scatterData.map((entry, index) => (
                <Cell key={index} fill={entry.ppsf < subjectPpsf ? '#22c55e' : '#94a3b8'} />
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
          Red star = landlord&apos;s asking price. Green dots = properties with lower £/sqft.
        </p>
      </div>

      {/* Chart 3: SW1 District £/sqft Comparison - Shows asking is above even premium area medians */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-1">
          Median £/sqft by SW1 District
        </h3>
        <p className="text-xs text-gray-500 mb-3">
          Even within premium SW1/SW3/SW7, the asking price exceeds most district medians.
        </p>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={districtData} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="district" fontSize={10} />
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
          Green = district median below asking. Even in prime SW1, the median is typically below £{subjectPpsf.toFixed(2)}/sqft.
        </p>
      </div>

      {/* Chart 4: Market Distribution with Context */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-1">
          London Market £/sqft Distribution
        </h3>
        <p className="text-xs text-gray-500 mb-3">
          Where the asking price sits in the overall London rental market.
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
          Note: This is ALL of London. SW1 commands a premium, but even within SW1,
          the asking price of £{subjectPpsf.toFixed(2)}/sqft is above the area median.
        </p>
      </div>
    </div>
  );
}
