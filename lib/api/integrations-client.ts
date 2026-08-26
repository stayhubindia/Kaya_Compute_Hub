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

export const integrationsClient = {
  startGoogleOAuth: async (redirectUri?: string): Promise<{ authorization_url: string; state: string }> => {
    const query = redirectUri ? `?redirect_uri=${encodeURIComponent(redirectUri)}` : '';
    return api.get<{ authorization_url: string; state: string }>(`/integrations/google/start/${query}`);
  },

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

  reconnectAccount: async (accountId: string, redirectUri?: string): Promise<{ authorization_url: string; state: string }> => {
    const query = redirectUri ? `?redirect_uri=${encodeURIComponent(redirectUri)}` : '';
    return api.post<{ authorization_url: string; state: string }>(`/integrations/google/${accountId}/reconnect/${query}`);
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
};
