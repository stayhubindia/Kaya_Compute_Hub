export interface APIErrorDetail {
  status_code: number;
  message: string;
  details?: Record<string, any>;
}

export interface APIErrorResponse {
  error: APIErrorDetail;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '/api/v1';

export function getCookie(name: string): string | null {
  if (typeof document === 'undefined') return null;
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop()?.split(';').shift() || null;
  return null;
}

export async function apiClient<T>(
  endpoint: string,
  options: RequestInit & { signal?: AbortSignal } = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
  
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };

  // Attach CSRF token for state-changing HTTP methods
  const csrfToken = getCookie('csrftoken');
  if (csrfToken && ['POST', 'PUT', 'PATCH', 'DELETE'].includes((options.method || 'GET').toUpperCase())) {
    headers['X-CSRFToken'] = csrfToken;
  }

  const requestOptions: RequestInit = {
    ...options,
    headers,
    credentials: 'include',
  };

  try {
    const response = await fetch(url, requestOptions);

    if (response.status === 401) {
      if (typeof window !== 'undefined' && !window.location.pathname.startsWith('/login')) {
        window.location.href = '/login?expired=true';
      }
      const errorJson = await response.json().catch(() => ({ error: { status_code: 401, message: 'Authentication required' } }));
      throw new Error(errorJson.error?.message || 'Authentication credentials were not provided.');
    }

    if (!response.ok) {
      const errorJson: APIErrorResponse = await response.json().catch(() => ({
        error: {
          status_code: response.status,
          message: `HTTP Error ${response.status}: ${response.statusText}`,
        },
      }));
      throw new Error(errorJson.error?.message || `Request failed with status ${response.status}`);
    }

    // Handle empty 204 No Content
    if (response.status === 204) {
      return {} as T;
    }

    return await response.json();
  } catch (error: any) {
    if (error.name === 'AbortError') {
      throw error;
    }
    throw error;
  }
}

export const api = {
  get: <T>(endpoint: string) => apiClient<T>(endpoint, { method: 'GET' }),
  post: <T>(endpoint: string, body?: any) => apiClient<T>(endpoint, { method: 'POST', body: body ? JSON.stringify(body) : undefined }),
};
