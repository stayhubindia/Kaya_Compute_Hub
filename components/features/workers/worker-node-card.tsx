'use client';

import React from 'react';
import { WorkerNode } from '../../../lib/schemas/dashboard-schemas';
import { StatusBadge } from '../../shared/status-badge';

interface WorkerNodeCardProps {
  worker: WorkerNode;
}

export const WorkerNodeCard: React.FC<WorkerNodeCardProps> = ({ worker }) => {
  const memoryGb = (worker.memory_bytes / (1024 * 1024 * 1024)).toFixed(1);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col space-y-4 hover:border-slate-700 transition-all">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h4 className="text-base font-bold text-slate-100 flex items-center space-x-2">
            <span>💻 {worker.name}</span>
          </h4>
          <p className="text-xs text-slate-400 font-mono mt-0.5">Host: {worker.hostname_label}</p>
        </div>
        <StatusBadge status={worker.status} />
      </div>

      {/* Stale Alert */}
      {worker.is_stale && (
        <div className="bg-rose-950/40 border border-rose-800/80 rounded-lg p-2.5 flex items-center space-x-2 text-rose-300 text-xs">
          <span>⚠️ Stale Heartbeat Warning (No heartbeat in &gt;60s)</span>
        </div>
      )}

      {/* Capacity Grid */}
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
          <span className="text-slate-500 font-medium block">CPU Cores</span>
          <span className="text-slate-200 font-bold text-sm">{worker.cpu_count} vCPUs</span>
        </div>
        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
          <span className="text-slate-500 font-medium block">RAM</span>
          <span className="text-slate-200 font-bold text-sm">{memoryGb} GB</span>
        </div>
        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
          <span className="text-slate-500 font-medium block">GPU Slots</span>
          <span className="text-cyan-400 font-bold text-sm">
            {worker.available_gpu_slots} / {worker.gpu_count} Available
          </span>
        </div>
        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
          <span className="text-slate-500 font-medium block">GPU Model</span>
          <span className="text-slate-300 font-medium truncate block">{worker.gpu_model}</span>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between text-[11px] text-slate-500 border-t border-slate-800 pt-3">
        <span>Active Jobs: <strong className="text-slate-300">{worker.active_jobs_count}</strong></span>
        <span>
          Heartbeat: {worker.last_heartbeat_at ? new Date(worker.last_heartbeat_at).toLocaleTimeString() : 'N/A'}
        </span>
      </div>
    </div>
  );
};
