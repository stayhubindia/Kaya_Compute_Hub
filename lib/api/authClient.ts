import { apiClient } from './client';

export interface User {
  id: string;
  email: string;
  is_active: boolean;
  last_login?: string | null;
  created_at: string;
  updated_at?: string;
}

export interface LoginResponse {
  user: User;
}

export const authClient = {
  async login(payload: { email: string; password: string }, signal?: AbortSignal): Promise<LoginResponse> {
    return apiClient<LoginResponse>('/auth/login/', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
    });
  },

  async logout(signal?: AbortSignal): Promise<{ message: string }> {
    return apiClient<{ message: string }>('/auth/logout/', {
      method: 'POST',
      signal,
    });
  },

  async getCurrentUser(signal?: AbortSignal): Promise<User> {
    return apiClient<User>('/auth/me/', {
      method: 'GET',
      signal,
    });
  },
};
