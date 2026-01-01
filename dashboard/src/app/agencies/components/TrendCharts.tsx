'use client';

import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
} from 'recharts';

interface AgencyDailyMetric {
  date: string;
  active_listings: number;
  new_listings: number;
  lets: number;
  price_reductions: number;
  cumulative_lets: number;
}

interface TrendChartsProps {
  data: AgencyDailyMetric[];
  agencyName: string;
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-GB', { month: 'short', day: 'numeric' });
}

export function InventoryTrendChart({ data, agencyName }: TrendChartsProps) {
  const chartData = data.map(d => ({
    ...d,
    date: formatDate(d.date),
  }));

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Inventory Trend (30 Days)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <YAxis
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            domain={['auto', 'auto']}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            }}
          />
          <Area
            type="monotone"
            dataKey="active_listings"
            name="Active Listings"
            stroke="#3b82f6"
            fill="#93c5fd"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ActivityChart({ data, agencyName }: TrendChartsProps) {
  const chartData = data.map(d => ({
    ...d,
    date: formatDate(d.date),
  }));

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Daily Activity</h3>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <YAxis
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            }}
          />
          <Legend />
          <Bar
            dataKey="new_listings"
            name="New Listings"
            fill="#22c55e"
            radius={[4, 4, 0, 0]}
          />
          <Bar
            dataKey="lets"
            name="Properties Let"
            fill="#8b5cf6"
            radius={[4, 4, 0, 0]}
          />
          <Bar
            dataKey="price_reductions"
            name="Price Cuts"
            fill="#f97316"
            radius={[4, 4, 0, 0]}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CumulativeLetsChart({ data, agencyName }: TrendChartsProps) {
  const chartData = data.map(d => ({
    ...d,
    date: formatDate(d.date),
  }));

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Cumulative Lets</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <YAxis
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            }}
          />
          <Line
            type="monotone"
            dataKey="cumulative_lets"
            name="Total Lets (30 days)"
            stroke="#8b5cf6"
            strokeWidth={3}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

interface PriceEvent {
  date: string;
  address: string;
  old_price: number;
  new_price: number;
  change_pct: number;
  bedrooms: number;
  postcode: string;
}

interface PriceHistoryProps {
  data: PriceEvent[];
}

export function PriceHistoryTable({ data }: PriceHistoryProps) {
  if (!data || data.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Recent Price Changes</h3>
        <p className="text-gray-500 text-sm">No price changes recorded in the last 30 days</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Recent Price Changes</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-3 py-2 text-left">Date</th>
              <th className="px-3 py-2 text-left">Address</th>
              <th className="px-3 py-2 text-right">Beds</th>
              <th className="px-3 py-2 text-right">Old Price</th>
              <th className="px-3 py-2 text-right">New Price</th>
              <th className="px-3 py-2 text-right">Change</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {data.slice(0, 15).map((event, idx) => (
              <tr key={`${event.date}-${idx}`} className="hover:bg-gray-50">
                <td className="px-3 py-2 text-gray-500">{event.date}</td>
                <td className="px-3 py-2 max-w-[200px] truncate" title={event.address}>
                  {event.address}
                </td>
                <td className="px-3 py-2 text-right">{event.bedrooms}</td>
                <td className="px-3 py-2 text-right text-gray-500">
                  £{event.old_price.toLocaleString()}
                </td>
                <td className="px-3 py-2 text-right font-medium">
                  £{event.new_price.toLocaleString()}
                </td>
                <td className={`px-3 py-2 text-right font-medium ${
                  event.change_pct < 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {event.change_pct > 0 ? '+' : ''}{event.change_pct.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Multi-agency comparison chart
interface ComparisonData {
  metric: string;
  [key: string]: string | number;
}

interface ComparisonChartProps {
  data: ComparisonData[];
  agencies: string[];
  colors: string[];
}

export function AgencyComparisonChart({ data, agencies, colors }: ComparisonChartProps) {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Agency Comparison</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis type="number" tick={{ fontSize: 12 }} />
          <YAxis
            type="category"
            dataKey="metric"
            tick={{ fontSize: 12 }}
            width={90}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            }}
          />
          <Legend />
          {agencies.map((agency, idx) => (
            <Bar
              key={agency}
              dataKey={agency}
              fill={colors[idx % colors.length]}
              radius={[0, 4, 4, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
