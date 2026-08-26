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
  const [confirmDelete, setConfirmDelete] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const s = (status || '').toLowerCase();

  const handleAction = async (action: 'cancel' | 'pause' | 'resume' | 'retry') => {
    try {
      setLoading(true);
      setError(null);
      try {
        await api.post(`/training-runs/${jobId}/${action}/`, {});
      } catch {
        await api.post(`/jobs/${jobId}/${action}/`, {});
      }
      if (onActionComplete) onActionComplete();
    } catch (err: any) {
      setError(err?.message || `Failed to ${action} job.`);
    } finally {
      setLoading(false);
      setConfirmCancel(false);
    }
  };

  const handleDelete = async () => {
    try {
      setLoading(true);
      setError(null);
      await api.delete(`/jobs/${jobId}/`);
      if (onActionComplete) onActionComplete();
    } catch (err: any) {
      setError(err?.message || 'Failed to delete job.');
    } finally {
      setLoading(false);
      setConfirmDelete(false);
    }
  };

  const isTerminal = ['succeeded', 'failed', 'cancelled'].includes(s);

  return (
    <div className="flex flex-col space-y-2">
      {error && <p className="text-xs text-rose-400 font-medium">{error}</p>}
      <div className="flex items-center space-x-2 flex-wrap gap-y-2">
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
            <div className="flex items-center space-x-1 bg-slate-900 p-1 rounded border border-slate-700">
              <span className="text-xs text-rose-400 font-medium mr-1">Cancel?</span>
              <button
                onClick={() => handleAction('cancel')}
                disabled={loading}
                className="px-2 py-0.5 text-xs font-bold rounded bg-rose-600 hover:bg-rose-500 text-white transition-colors"
              >
                Yes
              </button>
              <button
                onClick={() => setConfirmCancel(false)}
                className="px-2 py-0.5 text-xs rounded bg-slate-800 text-slate-400 hover:bg-slate-700"
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

        {/* Delete Action */}
        {confirmDelete ? (
          <div className="flex items-center space-x-1 bg-slate-900 p-1 rounded border border-slate-700">
            <span className="text-xs text-rose-400 font-medium mr-1">Delete permanently?</span>
            <button
              onClick={handleDelete}
              disabled={loading}
              className="px-2 py-0.5 text-xs font-bold rounded bg-rose-700 hover:bg-rose-600 text-white transition-colors"
            >
              Confirm
            </button>
            <button
              onClick={() => setConfirmDelete(false)}
              className="px-2 py-0.5 text-xs rounded bg-slate-800 text-slate-400 hover:bg-slate-700"
            >
              No
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirmDelete(true)}
            disabled={loading}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-400 hover:text-rose-400 transition-colors disabled:opacity-50"
          >
            🗑️ Delete
          </button>
        )}
      </div>
    </div>
  );
};
