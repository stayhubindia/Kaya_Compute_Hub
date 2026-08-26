'use client';

import { useEffect, useState, useCallback } from 'react';
import { JobLog } from '../schemas/dashboard-schemas';
import { api } from '../api/client';

export function useJobLogs(jobId: string, levelFilter?: string) {
  const [logs, setLogs] = useState<JobLog[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [autoScroll, setAutoScroll] = useState<boolean>(true);

  const fetchLogs = useCallback(async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (levelFilter && levelFilter !== 'all') {
        params.append('level', levelFilter);
      }
      const resp = await api.get<{ logs: JobLog[] }>(`/jobs/${jobId}/logs/?${params.toString()}`);
      setLogs(resp.logs || []);
      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Failed to load logs');
    } finally {
      setLoading(false);
    }
  }, [jobId, levelFilter]);

  useEffect(() => {
    if (jobId) fetchLogs();
  }, [jobId, fetchLogs]);

  return { logs, loading, error, autoScroll, setAutoScroll, refetch: fetchLogs };
}
