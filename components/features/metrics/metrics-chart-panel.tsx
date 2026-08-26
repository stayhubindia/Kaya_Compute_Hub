'use client';

import React from 'react';
import { TrainingMetric } from '../../../lib/schemas/dashboard-schemas';

interface MetricsChartPanelProps {
  metrics: TrainingMetric[];
  totalPoints?: number;
  returnedPoints?: number;
}

export const MetricsChartPanel: React.FC<MetricsChartPanelProps> = ({
  metrics,
  totalPoints = 0,
  returnedPoints = 0
}) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="p-8 text-center rounded-xl bg-slate-900/40 border border-slate-800 text-slate-500">
        <p className="text-sm">No training scalar metrics recorded yet.</p>
      </div>
    );
  }

  // Group metrics by name
  const metricsByName: Record<string, TrainingMetric[]> = {};
  (metrics || []).forEach((m) => {
    if (!m || !m.name) return;
    if (!metricsByName[m.name]) metricsByName[m.name] = [];
    metricsByName[m.name].push(m);
  });

  const renderSimpleChart = (metricName: string, data: TrainingMetric[], color: string) => {
    if (!data || data.length < 2) {
      return (
        <div className="h-32 flex items-center justify-center text-xs text-slate-500">
          Insufficient data points for trend chart.
        </div>
      );
    }

    const values = data.map((d) => (typeof d?.value === 'number' ? d.value : 0));
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const range = maxVal - minVal || 1;

    const width = 300;
    const height = 120;
    const padding = 15;

    const points = data.map((d, index) => {
      const val = typeof d?.value === 'number' ? d.value : 0;
      const x = padding + (index / (data.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((val - minVal) / range) * (height - 2 * padding);
      return `${x},${y}`;
    }).join(' ');

    const lastVal = typeof data[data.length - 1]?.value === 'number' ? data[data.length - 1].value : 0;

    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase text-slate-300 tracking-wider">{metricName}</span>
          <span className="text-sm font-bold text-slate-100">{lastVal.toFixed(4)}</span>
        </div>
        <div className="relative w-full overflow-hidden">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-28 overflow-visible">
            <polyline
              fill="none"
              stroke={color}
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={points}
            />
          </svg>
        </div>
        <div className="flex justify-between text-[10px] text-slate-500 font-mono">
          <span>Min: {minVal.toFixed(4)}</span>
          <span>Max: {maxVal.toFixed(4)}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {totalPoints > returnedPoints && (
        <div className="text-[11px] text-amber-400 bg-amber-950/40 border border-amber-800/60 rounded-lg px-3 py-1.5 flex items-center justify-between">
          <span>📊 Metric Downsampling Active ({returnedPoints} points rendered from {totalPoints} total records)</span>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(metricsByName).map(([name, data]) => {
          let color = '#38bdf8'; // cyan default
          if (name.includes('loss')) color = '#f43f5e'; // rose
          if (name.includes('acc')) color = '#10b981'; // emerald
          if (name.includes('rate')) color = '#f59e0b'; // amber

          return (
            <div key={name}>
              {renderSimpleChart(name, data, color)}
            </div>
          );
        })}
      </div>
    </div>
  );
};
