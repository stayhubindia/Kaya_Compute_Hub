'use client';

import React, { useEffect, useState } from 'react';
import DashboardNavbar from '@/components/DashboardNavbar';
import { User, authClient } from '@/lib/api/authClient';
import { useWorkerStatus } from '@/lib/hooks/use-worker-status';
import { WorkerNodeCard } from '@/components/features/workers/worker-node-card';
import { LoadingState } from '@/components/shared/loading-state';
import { EmptyState } from '@/components/shared/empty-state';
import { ErrorState } from '@/components/shared/error-state';
import ColabSessionLauncher from '@/components/ColabSessionLauncher';

export default function DashboardWorkersPage() {
  const [user, setUser] = useState<User | null>(null);
  const { workers, loading, error, refetch } = useWorkerStatus();

  useEffect(() => {
    authClient.getCurrentUser().then(setUser).catch(() => {});
  }, []);

  return (
    <div style={{ background: '#090d16', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui' }}>
      <DashboardNavbar user={user} />

      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <div>
            <h1 style={{ fontSize: '24px', fontWeight: '700' }}>Worker Fleet Operations</h1>
            <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '4px' }}>
              Real-time monitoring of CPU/GPU capacity, slot allocation, and node heartbeat status.
            </p>
          </div>

          <button
            onClick={() => refetch()}
            style={{ padding: '8px 16px', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', fontSize: '13px' }}
          >
            🔄 Refresh Status
          </button>
        </div>

        {/* Colab VM Session Launcher Panel */}
        <ColabSessionLauncher onSessionCreated={() => refetch()} />

        {error && <ErrorState message={error} onRetry={refetch} />}

        {loading && workers.length === 0 ? (
          <LoadingState message="Connecting to Worker Fleet Stream..." />
        ) : workers.length === 0 ? (
          <EmptyState
            title="No Worker Nodes Online"
            description="Start Celery workers to register execution nodes with the control plane."
          />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {workers.map((w) => (
              <WorkerNodeCard key={w.id} worker={w} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
