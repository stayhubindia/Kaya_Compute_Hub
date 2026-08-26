'use client';

import { useEffect, useState, useCallback } from 'react';
import { WorkerNode } from '../schemas/dashboard-schemas';
import { api } from '../api/client';

export function useWorkerStatus() {
  const [workers, setWorkers] = useState<WorkerNode[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkers = useCallback(async () => {
    try {
      setLoading(true);
      const resp = await api.get<{ workers: WorkerNode[] }>('/workers/');
      setWorkers(resp.workers || []);
      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Failed to load workers');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWorkers();
    const interval = setInterval(fetchWorkers, 10000);
    return () => clearInterval(interval);
  }, [fetchWorkers]);

  return { workers, loading, error, refetch: fetchWorkers };
}
