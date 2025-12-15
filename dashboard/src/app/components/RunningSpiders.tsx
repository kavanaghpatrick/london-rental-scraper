'use client';

import { useEffect, useState } from 'react';

interface RunningSpider {
  run_id: string;
  spider_name: string;
  started_at: string;
  items_scraped: number;
  error_count: number;
  elapsed_minutes: number;
}

export default function RunningSpiders() {
  const [running, setRunning] = useState<RunningSpider[]>([]);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  useEffect(() => {
    const fetchRunning = async () => {
      try {
        const res = await fetch('/api/running');
        const data = await res.json();
        setRunning(data);
        setLastUpdated(new Date());
      } catch (e) {
        console.error('Failed to fetch running spiders:', e);
      }
    };

    fetchRunning();
    const interval = setInterval(fetchRunning, 10000); // Poll every 10 seconds

    return () => clearInterval(interval);
  }, []);

  if (running.length === 0) {
    return null;
  }

  return (
    <div className="mb-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
          <h2 className="text-lg font-semibold text-blue-900">Scrape In Progress</h2>
        </div>
        <div className="text-xs text-blue-600">
          Updates every 10s {lastUpdated && `(${lastUpdated.toLocaleTimeString()})`}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {running.map((spider) => (
          <div
            key={`${spider.run_id}-${spider.spider_name}`}
            className="bg-white rounded-lg p-4 border border-blue-100"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-blue-900 capitalize">{spider.spider_name}</span>
              <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                {spider.elapsed_minutes}m elapsed
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-500">Items:</span>
                <span className="ml-2 font-medium">{spider.items_scraped.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-gray-500">Errors:</span>
                <span className={`ml-2 font-medium ${spider.error_count > 0 ? 'text-red-600' : ''}`}>
                  {spider.error_count}
                </span>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-400">
              Run ID: {spider.run_id}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
