'use client';

import React, { useEffect, useState } from 'react';
import DashboardNavbar from '@/components/DashboardNavbar';
import { User, authClient } from '@/lib/api/authClient';
import { Job, jobsClient } from '@/lib/api/jobsClient';

export default function DashboardJobsPage() {
  const [user, setUser] = useState<User | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  
  const [name, setName] = useState('');
  const [jobType, setJobType] = useState('download');
  const [description, setDescription] = useState('');

  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);

  const fetchJobs = async () => {
    try {
      const data = await jobsClient.listJobs();
      setJobs(data.results || []);
    } catch (err: any) {
      setError(err.message || 'Failed to list jobs.');
    }
  };

  useEffect(() => {
    async function init() {
      try {
        const u = await authClient.getCurrentUser();
        setUser(u);
        await fetchJobs();
      } catch (err: any) {
        setError(err.message || 'Failed to initialize.');
      } finally {
        setIsLoading(false);
      }
    }
    init();
  }, []);

  const handleCreateJob = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setActionSuccess(null);

    try {
      await jobsClient.createJob({
        name,
        job_type: jobType,
        description,
        payload: { demo: true }
      });
      setName('');
      setDescription('');
      setActionSuccess('Job submitted and enqueued successfully!');
      await fetchJobs();
    } catch (err: any) {
      setError(err.message || 'Failed to submit job.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = async (id: string) => {
    try {
      await jobsClient.cancelJob(id);
      setActionSuccess(`Job ${id.slice(0, 8)} cancelled.`);
      await fetchJobs();
    } catch (err: any) {
      setError(err.message || 'Failed to cancel job.');
    }
  };

  const handleRetry = async (id: string) => {
    try {
      await jobsClient.retryJob(id);
      setActionSuccess(`Job ${id.slice(0, 8)} retried.`);
      await fetchJobs();
    } catch (err: any) {
      setError(err.message || 'Failed to retry job.');
    }
  };

  const handleDelete = async (id: string) => {
    if (!window.confirm("Are you sure you want to delete this job permanently?")) return;
    try {
      await jobsClient.deleteJob(id);
      setActionSuccess(`Job ${id.slice(0, 8)} deleted permanently.`);
      await fetchJobs();
    } catch (err: any) {
      setError(err.message || 'Failed to delete job.');
    }
  };

  if (isLoading) {
    return <div style={{ background: '#090d16', minHeight: '100vh', color: '#fff', padding: '40px' }}>Loading jobs...</div>;
  }

  const canCreate = Boolean(user);

  return (
    <div style={{ background: '#090d16', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui' }}>
      <DashboardNavbar user={user} />

      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '24px' }}>Compute Job Orchestration</h1>

        {actionSuccess && (
          <div style={{ background: '#14532d', border: '1px solid #166534', color: '#86efac', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px' }}>
            {actionSuccess}
          </div>
        )}

        {error && (
          <div style={{ background: '#451a1a', border: '1px solid #991b1b', color: '#fca5a5', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px' }}>
            {error}
          </div>
        )}

        {canCreate && (
          <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '24px', marginBottom: '32px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px' }}>Submit New Approved Job</h2>
            <form onSubmit={handleCreateJob} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px' }}>Job Name</label>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Ingest Image Dataset"
                  style={{ width: '100%', padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px' }}>Job Type</label>
                <select
                  value={jobType}
                  onChange={(e) => setJobType(e.target.value)}
                  style={{ width: '100%', padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff' }}
                >
                  <option value="download">Download (Approved Demo)</option>
                  <option value="extraction">Extraction (Approved Demo)</option>
                  <option value="preprocessing">Preprocessing (Approved Demo)</option>
                  <option value="notebook">Notebook (Safe Demo)</option>
                  <option value="training">Training (Safe Demo)</option>
                </select>
              </div>

              <div style={{ gridColumn: '1 / -1' }}>
                <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px' }}>Description</label>
                <input
                  type="text"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Optional description of workload"
                  style={{ width: '100%', padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff' }}
                />
              </div>

              <div style={{ gridColumn: '1 / -1' }}>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  style={{ padding: '10px 20px', background: '#0284c7', color: '#fff', border: 'none', borderRadius: '6px', fontWeight: '600', cursor: 'pointer' }}
                >
                  {isSubmitting ? 'Enqueueing Task...' : 'Submit & Enqueue Task'}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Jobs List */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600' }}>Job Queue</h2>
            <button
              onClick={fetchJobs}
              style={{ background: '#0f172a', border: '1px solid #334155', color: '#94a3b8', padding: '6px 12px', borderRadius: '6px', fontSize: '12px', cursor: 'pointer' }}
            >
              🔄 Refresh
            </button>
          </div>

          {jobs.length === 0 ? (
            <div style={{ color: '#94a3b8', textAlign: 'center', padding: '24px' }}>No jobs enqueued.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {jobs.map((j) => (
                <div key={j.id} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
                  <div>
                    <div style={{ fontWeight: '600', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {j.name} <span style={{ fontSize: '12px', color: '#94a3b8' }}>({j.job_type})</span>
                    </div>
                    <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>Stage: {j.current_stage || j.status} • {j.progress_message || 'In queue'}</div>
                    <div style={{ marginTop: '8px', width: '240px', background: '#334155', borderRadius: '4px', height: '6px', overflow: 'hidden' }}>
                      <div style={{ width: `${j.progress_percentage || 0}%`, background: '#38bdf8', height: '100%' }} />
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ padding: '4px 8px', borderRadius: '4px', fontSize: '12px', background: '#1e3a8a', color: '#93c5fd' }}>
                      {j.status} ({j.progress_percentage || 0}%)
                    </span>

                    <a
                      href={`/dashboard/jobs/${j.id}`}
                      style={{ padding: '6px 12px', background: '#0284c7', color: '#fff', borderRadius: '6px', fontSize: '12px', textDecoration: 'none', fontWeight: '600' }}
                    >
                      🖥️ View Logs
                    </a>

                    {['queued', 'running', 'leased', 'draft'].includes(j.status) && (
                      <button
                        onClick={() => handleCancel(j.id)}
                        style={{ padding: '6px 12px', background: '#7f1d1d', color: '#fca5a5', border: 'none', borderRadius: '6px', fontSize: '12px', cursor: 'pointer' }}
                      >
                        🛑 Cancel
                      </button>
                    )}

                    {j.status === 'failed' && canCreate && (
                      <button
                        onClick={() => handleRetry(j.id)}
                        style={{ padding: '6px 12px', background: '#1e3a8a', color: '#93c5fd', border: 'none', borderRadius: '6px', fontSize: '12px', cursor: 'pointer' }}
                      >
                        🔄 Retry
                      </button>
                    )}

                    <button
                      onClick={() => handleDelete(j.id)}
                      style={{ padding: '6px 12px', background: '#0f172a', color: '#f87171', border: '1px solid #991b1b', borderRadius: '6px', fontSize: '12px', cursor: 'pointer' }}
                    >
                      🗑️ Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
