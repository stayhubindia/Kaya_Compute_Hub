import { apiClient } from './client';
import { PaginatedList } from './jobsClient';

export interface WorkerNode {
  id: string;
  name: string;
  hostname: string;
  status: 'offline' | 'idle' | 'busy' | 'draining' | 'unhealthy';
  capabilities: Record<string, any>;
  cpu_count: number;
  memory_bytes: number;
  gpu_count: number;
  last_heartbeat_at: string | null;
  created_at: string;
}

export const workersClient = {
  async listWorkers(signal?: AbortSignal): Promise<PaginatedList<WorkerNode>> {
    return apiClient<PaginatedList<WorkerNode>>('/workers/', {
      method: 'GET',
      signal,
    });
  },

  async getWorker(id: string, signal?: AbortSignal): Promise<WorkerNode> {
    return apiClient<WorkerNode>(`/workers/${id}/`, {
      method: 'GET',
      signal,
    });
  },
};
