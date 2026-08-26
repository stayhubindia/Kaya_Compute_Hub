import { apiClient } from './client';

export interface Download {
  id: string;
  created_by: string;
  source_url: string;
  provider: string;
  original_filename: string;
  storage_uri?: string;
  content_type?: string;
  expected_size_bytes?: number;
  downloaded_size_bytes: number;
  checksum_algorithm: string;
  expected_checksum?: string;
  actual_checksum?: string;
  extract: boolean;
  status: 'pending' | 'queued' | 'resolving' | 'downloading' | 'validating' | 'extracting' | 'completed' | 'failed' | 'cancelled' | 'paused';
  progress_percent: number;
  current_speed_bytes: number;
  retry_count: number;
  error_code?: string;
  error_message?: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;
  updated_at: string;
}

export interface CreateDownloadRequest {
  url: string;
  expected_checksum?: string;
  checksum_algorithm?: 'sha256' | 'sha512' | 'md5';
  extract?: boolean;
}

export interface DownloadActionResponse {
  id: string;
  status: string;
  message: string;
}

export const downloadsClient = {
  async list(signal?: AbortSignal): Promise<{ results: Download[] }> {
    return apiClient<{ results: Download[] }>('/downloads/', {
      method: 'GET',
      signal,
    });
  },

  async get(id: string, signal?: AbortSignal): Promise<Download> {
    return apiClient<Download>(`/downloads/${id}/`, {
      method: 'GET',
      signal,
    });
  },

  async create(payload: CreateDownloadRequest, signal?: AbortSignal): Promise<DownloadActionResponse> {
    return apiClient<DownloadActionResponse>('/downloads/', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
    });
  },

  async cancel(id: string, signal?: AbortSignal): Promise<DownloadActionResponse> {
    return apiClient<DownloadActionResponse>(`/downloads/${id}/cancel/`, {
      method: 'POST',
      signal,
    });
  },

  async pause(id: string, signal?: AbortSignal): Promise<DownloadActionResponse> {
    return apiClient<DownloadActionResponse>(`/downloads/${id}/pause/`, {
      method: 'POST',
      signal,
    });
  },

  async resume(id: string, signal?: AbortSignal): Promise<DownloadActionResponse> {
    return apiClient<DownloadActionResponse>(`/downloads/${id}/resume/`, {
      method: 'POST',
      signal,
    });
  },

  async verify(id: string, signal?: AbortSignal): Promise<{ id: string; status: string; verified: boolean }> {
    return apiClient<{ id: string; status: string; verified: boolean }>(`/downloads/${id}/verify/`, {
      method: 'POST',
      signal,
    });
  },
};
