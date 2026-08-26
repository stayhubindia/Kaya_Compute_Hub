'use client';

import React, { useState } from 'react';
import { api } from '../../../lib/api/client';

interface JobActionButtonsProps {
  jobId: string;
  status: string;
  onActionComplete?: () => void;
}

export const JobActionButtons: React.FC<JobActionButtonsProps> = ({ jobId, status, onActionComplete }) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [confirmCancel, setConfirmCancel] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const s = status.toLowerCase();

  const handleAction = async (action: 'cancel' | 'pause' | 'resume' | 'retry') => {
    try {
      setLoading(true);
      setError(null);
      await api.post(`/training-runs/${jobId}/${action}/`, {});
      if (onActionComplete) onActionComplete();
    } catch (err: any) {
      setError(err?.message || `Failed to ${action} job.`);
    } finally {
      setLoading(false);
      setConfirmCancel(false);
    }
  };

  const isTerminal = ['succeeded', 'failed', 'cancelled'].includes(s);

  return (
    <div className="flex flex-col space-y-2">
      {error && <p className="text-xs text-rose-400 font-medium">{error}</p>}
      <div className="flex items-center space-x-2">
        {/* Pause Action */}
        {['running', 'queued', 'preparing'].includes(s) && (
          <button
            onClick={() => handleAction('pause')}
            disabled={loading}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-amber-950/80 hover:bg-amber-900 border border-amber-800 text-amber-300 transition-colors disabled:opacity-50"
          >
            ⏸️ Pause
          </button>
        )}

        {/* Resume Action */}
        {['paused'].includes(s) && (
          <button
            onClick={() => handleAction('resume')}
            disabled={loading}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-800 text-emerald-300 transition-colors disabled:opacity-50"
          >
            ▶️ Resume
          </button>
        )}

        {/* Retry Action */}
        {['failed', 'cancelled'].includes(s) && (
          <button
            onClick={() => handleAction('retry')}
            disabled={loading}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-cyan-950/80 hover:bg-cyan-900 border border-cyan-800 text-cyan-300 transition-colors disabled:opacity-50"
          >
            🔄 Retry
          </button>
        )}

        {/* Cancel Action */}
        {!isTerminal && (
          confirmCancel ? (
            <div className="flex items-center space-x-1">
              <span className="text-xs text-rose-400 font-medium mr-1">Confirm Cancel?</span>
              <button
                onClick={() => handleAction('cancel')}
                disabled={loading}
                className="px-2.5 py-1 text-xs font-bold rounded bg-rose-600 hover:bg-rose-500 text-white transition-colors"
              >
                Yes
              </button>
              <button
                onClick={() => setConfirmCancel(false)}
                className="px-2.5 py-1 text-xs rounded bg-slate-800 text-slate-400 hover:bg-slate-700"
              >
                No
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirmCancel(true)}
              disabled={loading}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-rose-950/80 hover:bg-rose-900 border border-rose-800 text-rose-300 transition-colors disabled:opacity-50"
            >
              🛑 Cancel
            </button>
          )
        )}
      </div>
    </div>
  );
};
