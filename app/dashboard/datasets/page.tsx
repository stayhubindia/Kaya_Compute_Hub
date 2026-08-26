'use client';

import React, { useEffect, useState } from 'react';
import DashboardNavbar from '@/components/DashboardNavbar';
import { User, authClient } from '@/lib/api/authClient';
import { Dataset, datasetsClient } from '@/lib/api/datasetsClient';

export default function DashboardDatasetsPage() {
  const [user, setUser] = useState<User | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [name, setName] = useState('');
  const [storagePath, setStoragePath] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDatasets = async () => {
    try {
      const res = await datasetsClient.listDatasets();
      setDatasets(res.results || []);
    } catch (err: any) {
      setError(err.message || 'Failed to list datasets.');
    }
  };

  useEffect(() => {
    async function init() {
      try {
        const u = await authClient.getCurrentUser();
        setUser(u);
        await fetchDatasets();
      } catch (err: any) {
        setError(err.message || 'Failed to load datasets.');
      } finally {
        setIsLoading(false);
      }
    }
    init();
  }, []);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await datasetsClient.registerDataset({ name, storage_path: storagePath });
      setName('');
      setStoragePath('');
      await fetchDatasets();
    } catch (err: any) {
      setError(err.message || 'Failed to register dataset.');
    }
  };

  if (isLoading) return <div style={{ background: '#090d16', minHeight: '100vh', color: '#fff', padding: '40px' }}>Loading datasets...</div>;

  const canRegister = Boolean(user);

  return (
    <div style={{ background: '#090d16', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui' }}>
      <DashboardNavbar user={user} />
      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '24px' }}>Persistent Datasets</h1>

        {error && <div style={{ background: '#451a1a', border: '1px solid #991b1b', color: '#fca5a5', padding: '12px', borderRadius: '8px', marginBottom: '20px' }}>{error}</div>}

        {canRegister && (
          <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '24px', marginBottom: '32px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px' }}>Register Dataset</h2>
            <form onSubmit={handleRegister} style={{ display: 'flex', gap: '16px' }}>
              <input type="text" required placeholder="Dataset Name" value={name} onChange={(e) => setName(e.target.value)} style={{ padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff', flex: 1 }} />
              <input type="text" required placeholder="Storage Path (/app/storage/...)" value={storagePath} onChange={(e) => setStoragePath(e.target.value)} style={{ padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff', flex: 1 }} />
              <button type="submit" style={{ padding: '10px 20px', background: '#0284c7', color: '#fff', border: 'none', borderRadius: '6px', fontWeight: '600', cursor: 'pointer' }}>Register</button>
            </form>
          </div>
        )}

        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '24px' }}>
          {datasets.length === 0 ? <div style={{ color: '#94a3b8', textAlign: 'center' }}>No datasets registered.</div> : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {datasets.map(d => (
                <div key={d.id} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '16px' }}>
                  <div style={{ fontWeight: '600' }}>{d.name}</div>
                  <div style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>Path: {d.storage_path}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
