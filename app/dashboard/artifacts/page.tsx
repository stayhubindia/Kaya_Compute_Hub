'use client';

import React, { useEffect, useState } from 'react';
import DashboardNavbar from '@/components/DashboardNavbar';
import { User, authClient } from '@/lib/api/authClient';
import { api } from '@/lib/api/client';
import { ArtifactListTable } from '@/components/features/artifacts/artifact-list-table';
import { LoadingState } from '@/components/shared/loading-state';
import { EmptyState } from '@/components/shared/empty-state';
import { ErrorState } from '@/components/shared/error-state';
import { ArtifactItem } from '@/lib/schemas/dashboard-schemas';

export default function DashboardArtifactsPage() {
  const [user, setUser] = useState<User | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchArtifacts = async () => {
    try {
      setIsLoading(true);
      const u = await authClient.getCurrentUser();
      setUser(u);
      const res = await api.get<any>('/artifacts/');
      setArtifacts(res.results || (Array.isArray(res) ? res : []));
      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Failed to load artifacts.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchArtifacts();
  }, []);

  return (
    <div style={{ background: '#090d16', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui' }}>
      <DashboardNavbar user={user} />
      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <div>
            <h1 style={{ fontSize: '24px', fontWeight: '700' }}>Persistent Execution Artifacts</h1>
            <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '4px' }}>
              Browse, inspect checksums, and trigger secure authenticated downloads of job outputs.
            </p>
          </div>

          <button
            onClick={fetchArtifacts}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
          >
            🔄 Refresh List
          </button>
        </div>

        {error && <ErrorState message={error} onRetry={fetchArtifacts} />}

        {isLoading && artifacts.length === 0 ? (
          <LoadingState message="Fetching persistent artifact catalog..." />
        ) : artifacts.length === 0 ? (
          <EmptyState
            title="No Artifacts Recorded"
            description="Artifacts created during downloads, pipeline preprocessing, and training runs will appear here."
          />
        ) : (
          <ArtifactListTable artifacts={artifacts} />
        )}
      </main>
    </div>
  );
}
