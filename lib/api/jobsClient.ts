import { apiClient } from './client';

export interface Job {
  id: string;
  name: string;
  description: string;
  job_type: 'download' | 'extraction' | 'preprocessing' | 'ingest_documents' | 'generate_candidates' | 'run_quality_audit' | 'freeze_dataset' | 'train_qlora' | 'evaluate_model' | 'sync_to_drive' | 'custom_script';
  status: 'draft' | 'queued' | 'leased' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'retrying';
  priority: number;
  progress_percentage: number;
  current_stage: string;
  progress_message: string;
  assigned_worker: string | null;
  retry_count: number;
  max_retries: number;
  error_code: string | null;
  error_message: string | null;
  created_by: string;
  created_at: string;
  updated_at: string;
  payload?: Record<string, any>;
  selected_google_account?: string | null;
}

export interface PaginatedList<T> {
  count: number;
  next: string | null;
  previous: string | null;
  results: T[];
}

export const jobsClient = {
  async listJobs(params: { status?: string; job_type?: string; search?: string } = {}, signal?: AbortSignal): Promise<PaginatedList<Job>> {
    const query = new URLSearchParams(params as Record<string, string>).toString();
    return apiClient<PaginatedList<Job>>(`/jobs/${query ? `?${query}` : ''}`, {
      method: 'GET',
      signal,
    });
  },

  async getJob(id: string, signal?: AbortSignal): Promise<Job> {
    return apiClient<Job>(`/jobs/${id}/`, {
      method: 'GET',
      signal,
    });
  },

  async createJob(payload: { name: string; description?: string; job_type: string; payload?: Record<string, any>; priority?: number; selected_google_account_id?: string }, signal?: AbortSignal): Promise<{ id: string; status: string; message: string }> {
    return apiClient<{ id: string; status: string; message: string }>('/jobs/', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
    });
  },

  async cancelJob(id: string, signal?: AbortSignal): Promise<{ id: string; status: string; message: string }> {
    return apiClient<{ id: string; status: string; message: string }>(`/jobs/${id}/cancel/`, {
      method: 'POST',
      signal,
    });
  },

  async retryJob(id: string, signal?: AbortSignal): Promise<{ id: string; status: string; message: string }> {
    return apiClient<{ id: string; status: string; message: string }>(`/jobs/${id}/retry/`, {
      method: 'POST',
      signal,
    });
  },

  async deleteJob(id: string, signal?: AbortSignal): Promise<{ id: string; message: string }> {
    return apiClient<{ id: string; message: string }>(`/jobs/${id}/`, {
      method: 'DELETE',
      signal,
    });
  },
};
