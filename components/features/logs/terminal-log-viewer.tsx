'use client';

import React, { useState, useRef, useEffect } from 'react';
import { JobLog } from '../../../lib/schemas/dashboard-schemas';

interface TerminalLogViewerProps {
  logs: JobLog[];
  isConnected?: boolean;
  loading?: boolean;
  onRefresh?: () => void;
}

export const TerminalLogViewer: React.FC<TerminalLogViewerProps> = ({
  logs,
  isConnected = true,
  loading = false,
  onRefresh
}) => {
  const [levelFilter, setLevelFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [autoScroll, setAutoScroll] = useState<boolean>(true);
  const logEndRef = useRef<HTMLDivElement>(null);

  const safeLogs = Array.isArray(logs) ? logs : [];

  const filteredLogs = safeLogs.filter((log) => {
    if (!log) return false;
    const lvl = (log.level || 'info').toLowerCase();
    const msg = (log.message || '').toLowerCase();
    if (levelFilter !== 'all' && lvl !== levelFilter) return false;
    if (searchQuery && !msg.includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  useEffect(() => {
    if (autoScroll && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [filteredLogs, autoScroll]);

  const getLevelColor = (level?: string) => {
    switch ((level || 'info').toLowerCase()) {
      case 'error': return 'text-rose-400 font-semibold';
      case 'warning': return 'text-amber-300';
      case 'debug': return 'text-slate-400';
      default: return 'text-cyan-400';
    }
  };

  return (
    <div className="flex flex-col h-96 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs overflow-hidden shadow-2xl">
      {/* Terminal Toolbar */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2.5 bg-slate-900 border-b border-slate-800 gap-2">
        <div className="flex items-center space-x-2">
          <span className="w-3 h-3 rounded-full bg-rose-500/80 inline-block" />
          <span className="w-3 h-3 rounded-full bg-amber-500/80 inline-block" />
          <span className="w-3 h-3 rounded-full bg-emerald-500/80 inline-block" />
          <span className="ml-2 font-bold text-slate-300 text-xs tracking-wider uppercase">Live Job Terminal Logs</span>
        </div>

        {/* Controls */}
        <div className="flex items-center space-x-2">
          {/* Connection status */}
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] bg-slate-800 border border-slate-700">
            <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`} />
            {isConnected ? 'LIVE SSE' : 'DISCONNECTED'}
          </span>

          {/* Level Filter */}
          <select
            value={levelFilter}
            onChange={(e) => setLevelFilter(e.target.value)}
            className="bg-slate-800 border border-slate-700 text-slate-300 text-[11px] rounded px-2 py-1 outline-none"
          >
            <option value="all">ALL LEVELS</option>
            <option value="info">INFO</option>
            <option value="warning">WARNING</option>
            <option value="error">ERROR</option>
            <option value="debug">DEBUG</option>
          </select>

          {/* Search */}
          <input
            type="text"
            placeholder="Search logs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-slate-800 border border-slate-700 text-slate-300 text-[11px] rounded px-2 py-1 outline-none w-28 focus:w-36 transition-all"
          />

          {/* Auto Scroll Toggle */}
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`px-2 py-1 text-[10px] rounded font-semibold border transition-colors ${
              autoScroll
                ? 'bg-cyan-950/60 text-cyan-400 border-cyan-800'
                : 'bg-slate-800 text-slate-400 border-slate-700'
            }`}
          >
            {autoScroll ? 'AUTOSCROLL ON' : 'PAUSED'}
          </button>

          {/* Refresh */}
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="px-2 py-1 text-[10px] rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            >
              🔄
            </button>
          )}
        </div>
      </div>

      {/* Log Stream Body */}
      <div className="flex-1 p-4 overflow-y-auto space-y-1 text-slate-300 select-text">
        {filteredLogs.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-600 italic">
            {loading ? 'Fetching logs...' : 'No logs recorded for this execution.'}
          </div>
        ) : (
          filteredLogs.map((log, idx) => (
            <div key={log.id || idx} className="flex items-start space-x-2 leading-relaxed hover:bg-slate-900/60 rounded px-1 -mx-1">
              <span className="text-slate-500 shrink-0">[{log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : '--:--:--'}]</span>
              <span className={`shrink-0 uppercase text-[10px] px-1 py-0.2 rounded bg-slate-900 border border-slate-800 ${getLevelColor(log.level)}`}>
                {log.level || 'INFO'}
              </span>
              <span className="text-slate-400 shrink-0">[{log.module || 'system'}]:</span>
              <span className="text-slate-200 break-all">{log.message || ''}</span>
            </div>
          ))
        )}
        <div ref={logEndRef} />
      </div>
    </div>
  );
};
