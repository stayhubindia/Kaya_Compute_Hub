import React from 'react';

interface StatusBadgeProps {
  status: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  const s = status.toLowerCase();

  let colorClasses = 'bg-gray-800 text-gray-300 border-gray-700';

  if (['succeeded', 'online', 'approved', 'idle', 'completed', 'available'].includes(s)) {
    colorClasses = 'bg-emerald-950/60 text-emerald-400 border-emerald-800/80';
  } else if (['running', 'busy', 'preparing', 'checkpointing', 'leased'].includes(s)) {
    colorClasses = 'bg-amber-950/60 text-amber-400 border-amber-800/80 animate-pulse';
  } else if (['failed', 'unhealthy', 'offline', 'rejected', 'error'].includes(s)) {
    colorClasses = 'bg-rose-950/60 text-rose-400 border-rose-800/80';
  } else if (['queued', 'registered', 'scheduled', 'validating'].includes(s)) {
    colorClasses = 'bg-cyan-950/60 text-cyan-400 border-cyan-800/80';
  } else if (['paused', 'draining', 'archived', 'cancelled'].includes(s)) {
    colorClasses = 'bg-slate-800 text-slate-400 border-slate-700';
  }

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${colorClasses}`}>
      <span className="w-1.5 h-1.5 mr-1.5 rounded-full bg-current" />
      {status.toUpperCase()}
    </span>
  );
};
