import { api } from './client';

export interface ExternalNotebook {
  id: string;
  provider: string;
  project_id: string;
  region: string;
  notebook_resource_name: string;
  display_name: string;
  status: string;
  created_at: string;
}

export interface ExternalRun {
  id: string;
  notebook?: string;
  provider: string;
  external_run_id: string;
  local_job_id?: string;
  status: 'requested' | 'authorizing' | 'submitted' | 'running' | 'completed' | 'failed' | 'cancelled' | 'timed_out';
  output_uri: string;
  started_at?: string;
  finished_at?: string;
  error_code?: string;
  error_message?: string;
  metadata?: Record<string, any>;
  created_at: string;
}

export const colabClient = {
  listNotebooks: async (): Promise<ExternalNotebook[]> => {
    const res = await api.get<{ results: ExternalNotebook[] }>('/integrations/colab/notebooks/');
    return res.results || [];
  },

  runNotebook: async (notebookId: string, outputUri?: string, selectedGoogleAccountId?: string): Promise<ExternalRun> => {
    return api.post<ExternalRun>(`/integrations/colab/notebooks/${notebookId}/run/`, {
      output_uri: outputUri,
      selected_google_account_id: selectedGoogleAccountId
    });
  },

  getRunDetails: async (runId: string): Promise<ExternalRun> => {
    return api.get<ExternalRun>(`/integrations/colab/runs/${runId}/`);
  },

  cancelRun: async (runId: string): Promise<ExternalRun> => {
    return api.post<ExternalRun>(`/integrations/colab/runs/${runId}/cancel/`);
  },
};
