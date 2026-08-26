'use client';

import React, { useEffect, useState } from 'react';
import DashboardNavbar from '@/components/DashboardNavbar';
import { User, authClient } from '@/lib/api/authClient';
import { Job, jobsClient } from '@/lib/api/jobsClient';
import { WorkerNode, workersClient } from '@/lib/api/workersClient';

export default function DashboardOverviewPage() {
  const [user, setUser] = useState<User | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [workers, setWorkers] = useState<WorkerNode[]>([]);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadData() {
      try {
        const userData = await authClient.getCurrentUser(controller.signal);
        setUser(userData);

        const jobsData = await jobsClient.listJobs({}, controller.signal);
        setJobs(jobsData.results || []);

        const workersData = await workersClient.listWorkers(controller.signal);
        setWorkers(workersData.results || []);
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          setError(err.message || 'Failed to load control plane metrics.');
        }
      } finally {
        setIsLoading(false);
      }
    }

    loadData();
    return () => controller.abort();
  }, []);

  if (isLoading) {
    return (
      <div style={{ background: '#090d16', minHeight: '100vh', color: '#fff', padding: '40px', fontFamily: 'system-ui' }}>
        Loading control plane dashboard...
      </div>
    );
  }

  return (
    <div style={{ background: '#090d16', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui' }}>
      <DashboardNavbar user={user} />

      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
        <div style={{ marginBottom: '28px' }}>
          <h1 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '8px' }}>VM Control Plane Overview</h1>
          <p style={{ color: '#94a3b8', fontSize: '14px' }}>
            Asynchronous task orchestration, VM worker nodes, persistent storage tracking.
          </p>
        </div>

        {error && (
          <div style={{ background: '#451a1a', border: '1px solid #991b1b', color: '#fca5a5', padding: '12px 16px', borderRadius: '8px', marginBottom: '24px' }}>
            {error}
          </div>
        )}

        {/* Metrics Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '20px', marginBottom: '32px' }}>
          <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '20px' }}>
            <div style={{ fontSize: '13px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '600' }}>Active Jobs</div>
            <div style={{ fontSize: '32px', fontWeight: '700', color: '#38bdf8', marginTop: '8px' }}>
              {jobs.filter(j => ['queued', 'leased', 'running'].includes(j.status)).length}
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>Total Jobs Recorded: {jobs.length}</div>
          </div>

          <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '20px' }}>
            <div style={{ fontSize: '13px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '600' }}>Worker Nodes</div>
            <div style={{ fontSize: '32px', fontWeight: '700', color: '#4ade80', marginTop: '8px' }}>
              {workers.filter(w => w.status === 'idle' || w.status === 'busy').length}
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>Total Workers Registered: {workers.length}</div>
          </div>

          <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '20px' }}>
            <div style={{ fontSize: '13px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '600' }}>Admin Identity</div>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#4ade80', marginTop: '12px', wordBreak: 'break-all' }}>
              {user?.email || 'Authenticated Admin'}
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>Private Admin Account</div>
          </div>
        </div>

        {/* Recent Jobs Table */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '24px' }}>
          <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px' }}>Recent Compute Jobs</h2>
          
          {jobs.length === 0 ? (
            <div style={{ color: '#94a3b8', fontSize: '14px', textAlign: 'center', padding: '24px' }}>
              No compute jobs found. Submit a job from the Jobs tab.
            </div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #334155', textAlign: 'left', color: '#94a3b8' }}>
                  <th style={{ padding: '10px 12px' }}>Name</th>
                  <th style={{ padding: '10px 12px' }}>Type</th>
                  <th style={{ padding: '10px 12px' }}>Status</th>
                  <th style={{ padding: '10px 12px' }}>Progress</th>
                </tr>
              </thead>
              <tbody>
                {jobs.slice(0, 5).map((job) => (
                  <tr key={job.id} style={{ borderBottom: '1px solid #0f172a' }}>
                    <td style={{ padding: '12px' }}>{job.name}</td>
                    <td style={{ padding: '12px', textTransform: 'capitalize' }}>{job.job_type}</td>
                    <td style={{ padding: '12px' }}>
                      <span style={{
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: '12px',
                        fontWeight: '600',
                        background: job.status === 'succeeded' ? '#14532d' : job.status === 'failed' ? '#7f1d1d' : '#1e3a8a',
                        color: job.status === 'succeeded' ? '#86efac' : job.status === 'failed' ? '#fca5a5' : '#93c5fd'
                      }}>
                        {job.status}
                      </span>
                    </td>
                    <td style={{ padding: '12px' }}>{job.progress_percentage}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </main>
    </div>
  );
}
