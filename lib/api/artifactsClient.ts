import { apiClient } from './client';
import { PaginatedList } from './jobsClient';

export interface Artifact {
  id: string;
  job: string;
  name: string;
  artifact_type: string;
  file_path: string;
  file_size_bytes: number;
  checksum_sha256: string;
  metadata: Record<string, any>;
  created_at: string;
}

export const artifactsClient = {
  async listArtifacts(signal?: AbortSignal): Promise<PaginatedList<Artifact>> {
    return apiClient<PaginatedList<Artifact>>('/artifacts/', {
      method: 'GET',
      signal,
    });
  },
};
