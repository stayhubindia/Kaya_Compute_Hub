import { api } from './client';

export interface ConnectedAccount {
  id: string;
  provider: string;
  provider_account_id: string;
  email: string;
  display_name: string;
  token_expiry: string | null;
  scopes: string[];
  status: 'active' | 'expired' | 'revoked' | 'disconnected' | 'quota_exhausted' | 'error';
  last_verified_at: string | null;
  connected_at: string;
  disconnected_at: string | null;
}

export interface DriveFile {
  id: string;
  name: string;
  mimeType: string;
  size?: string;
  createdTime?: string;
  modifiedTime?: string;
  md5Checksum?: string;
}

export interface ColabSession {
  name: string;
  endpoint: string;
  accelerator: string;
  variant: string;
  status: string;
  drive_mounted?: boolean | null;
}

export const integrationsClient = {
  listConnectedAccounts: async (): Promise<ConnectedAccount[]> => {
    const res = await api.get<{ results: ConnectedAccount[] }>('/integrations/google/accounts/');
    return res.results || [];
  },

  directConnectAccount: async (data: { email: string; display_name?: string; access_token?: string; refresh_token?: string; raw_json?: string }): Promise<ConnectedAccount> => {
    return api.post<ConnectedAccount>('/integrations/google/direct-connect/', data);
  },

  verifyAccount: async (accountId: string): Promise<{ status: string; last_verified_at?: string; message?: string }> => {
    return api.post<{ status: string; last_verified_at?: string; message?: string }>(`/integrations/google/${accountId}/verify/`);
  },

  disconnectAccount: async (accountId: string): Promise<{ status: string; id: string }> => {
    return api.post<{ status: string; id: string }>(`/integrations/google/${accountId}/disconnect/`);
  },

  revokeAccount: async (accountId: string): Promise<{ status: string; id: string }> => {
    return api.post<{ status: string; id: string }>(`/integrations/google/${accountId}/revoke/`);
  },

  listDriveFiles: async (accountId: string, query?: string): Promise<{ files: DriveFile[] }> => {
    const qParam = query ? `?query=${encodeURIComponent(query)}` : '';
    return api.get<{ files: DriveFile[] }>(`/integrations/google/${accountId}/drive/files/${qParam}`);
  },

  importDriveFile: async (accountId: string, fileId: string): Promise<{ status: string; task_id: string }> => {
    return api.post<{ status: string; task_id: string }>(`/integrations/google/${accountId}/drive/import/`, { file_id: fileId });
  },

  exportArtifactToDrive: async (accountId: string, artifactId: string): Promise<{ status: string; task_id: string }> => {
    return api.post<{ status: string; task_id: string }>(`/integrations/google/${accountId}/drive/export/`, { artifact_id: artifactId });
  },

  createColabSession: async (data: { account_id?: string; session_name?: string; gpu_variant?: string }): Promise<{
    status: string;
    session_name: string;
    account_email: string;
    gpu_variant: string;
    created_at: string;
    kernel_ready: boolean;
    message: string;
  }> => {
    return api.post('/integrations/colab/sessions/create/', data);
  },

  startColabAuthorization: async (): Promise<{ authorization_id: string; authorization_url: string; expires_in_seconds: number; instruction: string }> => {
    return api.post('/integrations/colab/authorize/start/', {});
  },

  pendingColabAuthorization: async (): Promise<{ pending: boolean; authorization_id?: string; authorization_url?: string; expires_at?: number }> => {
    return api.get('/integrations/colab/authorize/pending/');
  },

  completeColabAuthorization: async (authorizationId: string, callbackUrl: string): Promise<{ status: string; account: ConnectedAccount }> => {
    return api.post('/integrations/colab/authorize/complete/', {
      authorization_id: authorizationId,
      callback_url: callbackUrl,
    });
  },

  listColabSessions: async (): Promise<{ output_raw: string; sessions: ColabSession[]; active_count: number; cli_error?: string }> => {
    return api.get('/integrations/colab/sessions/?include_drive_status=1');
  },

  stopColabSession: async (sessionName: string): Promise<{ status: string; session_name: string; message: string }> => {
    return api.post('/integrations/colab/sessions/stop/', { session_name: sessionName });
  },
};
