import { apiClient } from './client';

export interface PipelineStageConfig {
  name: string;
  params?: Record<string, any>;
}

export interface ResourcePolicy {
  max_cpu_cores?: number;
  max_memory_mb?: number;
  max_disk_mb?: number;
  timeout_seconds?: number;
  network_enabled?: boolean;
  run_as_non_root?: boolean;
}

export interface PipelineDefinition {
  id: string;
  name: string;
  description?: string;
  version: string;
  enabled: boolean;
  stages: PipelineStageConfig[];
  resource_policy: ResourcePolicy;
  created_by?: string;
  created_at: string;
  updated_at: string;
}

export interface ProcessingRun {
  id: string;
  pipeline: string;
  source_dataset: string;
  output_dataset?: string;
  status: 'draft' | 'queued' | 'validating' | 'running' | 'checkpointing' | 'succeeded' | 'failed' | 'cancelled' | 'paused' | 'retrying';
  current_stage?: string;
  progress_percent: number;
  input_manifest_uri?: string;
  output_manifest_uri?: string;
  error_code?: string;
  error_message?: string;
  started_at?: string;
  completed_at?: string;
  created_by?: string;
  created_at: string;
  updated_at: string;
}

export interface CreatePipelineRequest {
  name: string;
  description?: string;
  version?: string;
  stages: PipelineStageConfig[];
  resource_policy?: ResourcePolicy;
}

export interface CreateProcessingRunRequest {
  pipeline_id: string;
  source_dataset_id: string;
}

export interface ActionResponse {
  id: string;
  status: string;
  message: string;
}

export const pipelinesClient = {
  async listPipelines(signal?: AbortSignal): Promise<{ results: PipelineDefinition[] }> {
    return apiClient<{ results: PipelineDefinition[] }>('/pipelines/', {
      method: 'GET',
      signal,
    });
  },

  async getPipeline(id: string, signal?: AbortSignal): Promise<PipelineDefinition> {
    return apiClient<PipelineDefinition>(`/pipelines/${id}/`, {
      method: 'GET',
      signal,
    });
  },

  async createPipeline(payload: CreatePipelineRequest, signal?: AbortSignal): Promise<PipelineDefinition> {
    return apiClient<PipelineDefinition>('/pipelines/', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
    });
  },

  async listProcessingRuns(signal?: AbortSignal): Promise<{ results: ProcessingRun[] }> {
    return apiClient<{ results: ProcessingRun[] }>('/processing-runs/', {
      method: 'GET',
      signal,
    });
  },

  async getProcessingRun(id: string, signal?: AbortSignal): Promise<ProcessingRun> {
    return apiClient<ProcessingRun>(`/processing-runs/${id}/`, {
      method: 'GET',
      signal,
    });
  },

  async createProcessingRun(payload: CreateProcessingRunRequest, signal?: AbortSignal): Promise<ActionResponse> {
    return apiClient<ActionResponse>('/processing-runs/', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
    });
  },

  async cancelProcessingRun(id: string, signal?: AbortSignal): Promise<ActionResponse> {
    return apiClient<ActionResponse>(`/processing-runs/${id}/cancel/`, {
      method: 'POST',
      signal,
    });
  },

  async pauseProcessingRun(id: string, signal?: AbortSignal): Promise<ActionResponse> {
    return apiClient<ActionResponse>(`/processing-runs/${id}/pause/`, {
      method: 'POST',
      signal,
    });
  },

  async resumeProcessingRun(id: string, signal?: AbortSignal): Promise<ActionResponse> {
    return apiClient<ActionResponse>(`/processing-runs/${id}/resume/`, {
      method: 'POST',
      signal,
    });
  },
};
