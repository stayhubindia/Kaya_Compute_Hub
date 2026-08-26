'use client';

import React, { useEffect, useState } from 'react';
import DashboardNavbar from '@/components/DashboardNavbar';
import { User, authClient } from '@/lib/api/authClient';
import { api } from '@/lib/api/client';
import { StatusBadge } from '@/components/shared/status-badge';
import { TerminalLogViewer } from '@/components/features/logs/terminal-log-viewer';
import { MetricsChartPanel } from '@/components/features/metrics/metrics-chart-panel';
import { JobActionButtons } from '@/components/features/jobs/job-action-buttons';
import { ArtifactListTable } from '@/components/features/artifacts/artifact-list-table';
import { useJobLogs } from '@/lib/hooks/use-job-logs';
import { useJobEvents } from '@/lib/hooks/use-job-events';

export default function JobDetailPage({ params }: { params: { id: string } }) {
  const jobId = params?.id;

  const [user, setUser] = useState<User | null>(null);
  const [job, setJob] = useState<any>(null);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [checkpoints, setCheckpoints] = useState<any[]>([]);
  const [artifacts, setArtifacts] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<'logs' | 'metrics' | 'artifacts'>('logs');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { logs, refetch: refetchLogs } = useJobLogs(jobId);
  const { isConnected } = useJobEvents(jobId);

  const loadJobDetails = React.useCallback(async () => {
    try {
      setIsLoading(true);
      const userData = await authClient.getCurrentUser();
      setUser(userData);

      // Fetch training run details if training job, else fallback to standard job endpoint
      try {
        const tr = await api.get<any>(`/training-runs/${jobId}/`);
        setJob(tr);
      } catch {
        const j = await api.get<any>(`/jobs/${jobId}/`);
        setJob(j);
      }

      // Fetch metrics
      try {
        const mResp = await api.get<any>(`/training-runs/${jobId}/metrics/`);
        setMetrics(mResp.metrics || (Array.isArray(mResp) ? mResp : []));
      } catch {}

      // Fetch checkpoints
      try {
        const cResp = await api.get<any>(`/training-runs/${jobId}/checkpoints/`);
        setCheckpoints(Array.isArray(cResp) ? cResp : []);
      } catch {}

      // Fetch artifacts
      try {
        const aResp = await api.get<any>(`/artifacts/?job=${jobId}`);
        setArtifacts(aResp.results || (Array.isArray(aResp) ? aResp : []));
      } catch {}

      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Failed to load job operations detail.');
    } finally {
      setIsLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    loadJobDetails();
  }, [loadJobDetails]);

  if (isLoading) {
    return (
      <div style={{ background: '#090d16', minHeight: '100vh', color: '#fff', padding: '40px' }}>
        Loading job execution details...
      </div>
    );
  }

  if (error || !job) {
    return (
      <div style={{ background: '#090d16', minHeight: '100vh', color: '#fff', padding: '40px' }}>
        <p style={{ color: '#f87171' }}>Error: {error || 'Job not found'}</p>
      </div>
    );
  }

  return (
    <div style={{ background: '#090d16', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui' }}>
      <DashboardNavbar user={user} />

      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
        {/* Header Operations Card */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '24px', marginBottom: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '6px' }}>
                <h1 style={{ fontSize: '22px', fontWeight: '700' }}>{job.name || `Job #${jobId.substring(0, 8)}`}</h1>
                <StatusBadge status={job.status || 'unknown'} />
              </div>
              <p style={{ color: '#94a3b8', fontSize: '13px', fontFamily: 'monospace' }}>ID: {jobId}</p>
            </div>

            <JobActionButtons jobId={jobId} status={job.status || ''} onActionComplete={loadJobDetails} />
          </div>

          {/* Progress Bar */}
          <div style={{ marginTop: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8', marginBottom: '6px' }}>
              <span>Execution Progress</span>
              <span>{job.progress_percent ?? job.progress_percentage ?? 0}%</span>
            </div>
            <div style={{ width: '100%', height: '8px', background: '#0f172a', borderRadius: '4px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${job.progress_percent ?? job.progress_percentage ?? 0}%`,
                  height: '100%',
                  background: '#0284c7',
                  transition: 'width 0.3s'
                }}
              />
            </div>
          </div>
        </div>

        {/* Operations Tabs */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '20px', borderBottom: '1px solid #334155', paddingBottom: '8px' }}>
          <button
            onClick={() => setActiveTab('logs')}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: '600',
              border: 'none',
              background: activeTab === 'logs' ? '#0284c7' : 'transparent',
              color: activeTab === 'logs' ? '#fff' : '#94a3b8',
              cursor: 'pointer'
            }}
          >
            🖥️ Terminal Logs ({logs?.length || 0})
          </button>
          <button
            onClick={() => setActiveTab('metrics')}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: '600',
              border: 'none',
              background: activeTab === 'metrics' ? '#0284c7' : 'transparent',
              color: activeTab === 'metrics' ? '#fff' : '#94a3b8',
              cursor: 'pointer'
            }}
          >
            📊 Training Metrics ({metrics?.length || 0})
          </button>
          <button
            onClick={() => setActiveTab('artifacts')}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: '600',
              border: 'none',
              background: activeTab === 'artifacts' ? '#0284c7' : 'transparent',
              color: activeTab === 'artifacts' ? '#fff' : '#94a3b8',
              cursor: 'pointer'
            }}
          >
            📦 Artifacts ({artifacts?.length || 0})
          </button>
        </div>

        {/* Tab Contents */}
        {activeTab === 'logs' && (
          <TerminalLogViewer logs={logs} isConnected={isConnected} onRefresh={refetchLogs} />
        )}

        {activeTab === 'metrics' && (
          <MetricsChartPanel metrics={metrics} />
        )}

        {activeTab === 'artifacts' && (
          <ArtifactListTable artifacts={artifacts} />
        )}
      </main>
    </div>
  );
}
