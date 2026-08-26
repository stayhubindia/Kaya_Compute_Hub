import { apiClient } from './client';
import { PaginatedList } from './jobsClient';

export interface Dataset {
  id: string;
  name: string;
  description: string;
  source_type: string;
  storage_path: string;
  size_bytes: number;
  row_count: number | null;
  schema_metadata: Record<string, any>;
  created_by: string;
  created_at: string;
}

export const datasetsClient = {
  async listDatasets(signal?: AbortSignal): Promise<PaginatedList<Dataset>> {
    return apiClient<PaginatedList<Dataset>>('/datasets/', {
      method: 'GET',
      signal,
    });
  },

  async registerDataset(payload: { name: string; description?: string; source_type?: string; storage_path: string }, signal?: AbortSignal): Promise<Dataset> {
    return apiClient<Dataset>('/datasets/', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
    });
  },
};
