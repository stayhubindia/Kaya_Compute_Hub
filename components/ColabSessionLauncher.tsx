'use client';

import React, { useState, useEffect } from 'react';
import { integrationsClient, ConnectedAccount, ColabSession } from '@/lib/api/integrations-client';

interface ColabSessionLauncherProps {
  onSessionCreated?: (sessionName: string) => void;
}

export default function ColabSessionLauncher({ onSessionCreated }: ColabSessionLauncherProps) {
  const [accounts, setAccounts] = useState<ConnectedAccount[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState<string>('');
  const [sessionName, setSessionName] = useState<string>('colab-t4-worker');
  const [gpuVariant, setGpuVariant] = useState<string>('T4');
  const [isAllocating, setIsAllocating] = useState<boolean>(false);
  const [statusLogs, setStatusLogs] = useState<string[]>([]);
  const [activeSessions, setActiveSessions] = useState<ColabSession[]>([]);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fetchAccountsAndSessions = async () => {
    try {
      const accs = await integrationsClient.listConnectedAccounts();
      setAccounts(accs);
      if (accs.length > 0) {
        setSelectedAccountId(prev => prev || accs[0].id);
      }

      const sessRes = await integrationsClient.listColabSessions();
      setActiveSessions(sessRes.sessions || []);
    } catch (err) {
      console.error('Failed to load accounts or sessions:', err);
    }
  };

  useEffect(() => { fetchAccountsAndSessions(); }, []);

  const handleLaunchSession = async () => {
    if (!selectedAccountId && accounts.length > 0) {
      setErrorMessage('Please select an authenticated Vault account.');
      return;
    }

    setIsAllocating(true);
    setSuccessMessage(null);
    setErrorMessage(null);
    setStatusLogs([
      `[1/4] Syncing Vault credentials for selected account...`,
      `[2/4] Initializing Google Colab CLI worker...`,
      `[3/4] Requesting '${gpuVariant}' GPU accelerator runtime for '${sessionName}'...`,
    ]);

    try {
      const res = await integrationsClient.createColabSession({
        account_id: selectedAccountId,
        session_name: sessionName,
        gpu_variant: gpuVariant,
      });

      setStatusLogs(prev => [...prev, `[4/4] ✅ Kernel Readiness Verified! Colab session online.`]);
      setSuccessMessage(res.message || `Colab ${gpuVariant} VM session '${sessionName}' created successfully!`);
      
      fetchAccountsAndSessions();

      if (onSessionCreated) {
        onSessionCreated(sessionName);
      }
    } catch (err: any) {
      const errText = err?.message || 'Failed to allocate Colab session.';
      setErrorMessage(errText);
      setStatusLogs(prev => [...prev, `❌ Allocation error: ${errText}`]);
    } finally {
      setIsAllocating(false);
    }
  };

  const handleStopSession = async (sName: string) => {
    try {
      await integrationsClient.stopColabSession(sName);
      setSuccessMessage(`Session '${sName}' terminated successfully.`);
      fetchAccountsAndSessions();
    } catch (err: any) {
      setErrorMessage(err?.message || 'Failed to stop session.');
    }
  };

  const variants = [
    { id: 'T4', name: 'NVIDIA T4 GPU', desc: 'Standard Free Tier GPU', color: '#0284c7' },
    { id: 'High-RAM', name: 'High-RAM T4', desc: 'High-RAM Compute Node', color: '#4f46e5' },
    { id: 'TPU', name: 'Google TPU v5e1', desc: 'Tensor Processing Unit', color: '#9333ea' },
    { id: 'A100', name: 'NVIDIA A100', desc: 'Pro SXM4 Accelerator', color: '#d97706' },
    { id: 'CPU', name: 'Standard CPU', desc: 'Basic Runtime', color: '#64748b' },
  ];

  return (
    <div style={{
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      border: '1px solid #334155',
      borderRadius: '16px',
      padding: '24px',
      boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.4)',
      marginBottom: '28px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#f8fafc', margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
            ⚡ Google Colab Session Creator & VM Allocator
          </h2>
          <p style={{ color: '#94a3b8', fontSize: '13px', margin: '4px 0 0 0' }}>
            Allocate and launch stateless Colab T4/TPU/RAM GPU compute sessions using authenticated Vault accounts.
          </p>
        </div>

        <div style={{ background: '#0284c718', border: '1px solid #0284c744', padding: '6px 14px', borderRadius: '20px', color: '#38bdf8', fontSize: '12px', fontWeight: '600' }}>
          Authenticated Colab-CLI Native Engine
        </div>
      </div>

      {successMessage && (
        <div style={{ background: '#05966922', border: '1px solid #10b981', color: '#34d399', padding: '12px 16px', borderRadius: '10px', marginBottom: '16px', fontSize: '13px', fontWeight: '500' }}>
          ✅ {successMessage}
        </div>
      )}

      {errorMessage && (
        <div style={{ background: '#991b1b22', border: '1px solid #ef4444', color: '#f87171', padding: '12px 16px', borderRadius: '10px', marginBottom: '16px', fontSize: '13px', fontWeight: '500' }}>
          ⚠️ {errorMessage}
        </div>
      )}

      {/* Inputs Section */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '20px' }}>
        {/* Vault Account Selector */}
        <div>
          <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
            🔑 Select Authenticated Vault Account:
          </label>
          <select
            value={selectedAccountId}
            onChange={(e) => setSelectedAccountId(e.target.value)}
            style={{ width: '100%', background: '#0f172a', color: '#f8fafc', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', fontSize: '14px', outline: 'none' }}
          >
            {accounts.length === 0 ? (
              <option value="">-- No Vault Accounts Found (Authenticate in Connections) --</option>
            ) : (
              accounts.map((acc) => (
                <option key={acc.id} value={acc.id}>
                  {acc.email || acc.display_name} [{acc.status.toUpperCase()}]
                </option>
              ))
            )}
          </select>
        </div>

        {/* Session Name */}
        <div>
          <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
            🏷️ Colab Session Identifier Name:
          </label>
          <input
            type="text"
            value={sessionName}
            onChange={(e) => setSessionName(e.target.value)}
            placeholder="e.g. colab-t4-worker"
            style={{ width: '100%', background: '#0f172a', color: '#f8fafc', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', fontSize: '14px', fontFamily: 'monospace', outline: 'none' }}
          />
        </div>
      </div>

      {/* Hardware Variant Selection */}
      <div style={{ marginBottom: '24px' }}>
        <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#cbd5e1', marginBottom: '10px' }}>
          🎮 Select Hardware Accelerator Runtime:
        </label>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '10px' }}>
          {variants.map((v) => {
            const isSelected = gpuVariant === v.id;
            return (
              <div
                key={v.id}
                onClick={() => setGpuVariant(v.id)}
                style={{
                  background: isSelected ? `${v.color}22` : '#0f172a',
                  border: isSelected ? `2px solid ${v.color}` : '1px solid #334155',
                  borderRadius: '10px',
                  padding: '12px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px'
                }}
              >
                <div style={{ fontSize: '14px', fontWeight: '700', color: isSelected ? '#fff' : '#cbd5e1' }}>
                  {v.name}
                </div>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>
                  {v.desc}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Action Button */}
      <button
        onClick={handleLaunchSession}
        disabled={isAllocating || accounts.length === 0}
        style={{
          width: '100%',
          background: isAllocating ? '#334155' : 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
          color: '#ffffff',
          border: 'none',
          padding: '14px 24px',
          borderRadius: '10px',
          fontWeight: '700',
          fontSize: '15px',
          cursor: isAllocating || accounts.length === 0 ? 'not-allowed' : 'pointer',
          boxShadow: isAllocating ? 'none' : '0 4px 14px rgba(2, 132, 199, 0.4)',
          transition: 'all 0.2s ease',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '10px'
        }}
      >
        {isAllocating ? (
          <>⏳ Allocating Colab {gpuVariant} VM Session...</>
        ) : (
          <>🚀 Create & Allocate Colab Session Now</>
        )}
      </button>

      {/* Live Status Logs */}
      {statusLogs.length > 0 && (
        <div style={{ marginTop: '20px', background: '#090d16', border: '1px solid #1e293b', borderRadius: '10px', padding: '14px', fontFamily: 'monospace', fontSize: '12px', color: '#38bdf8' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px', marginBottom: '6px', fontWeight: '700' }}>LIVE ALLOCATION LOGS:</div>
          {statusLogs.map((log, idx) => (
            <div key={idx} style={{ margin: '3px 0' }}>{log}</div>
          ))}
        </div>
      )}

      {/* Active Sessions List */}
      {activeSessions.length > 0 && (
        <div style={{ marginTop: '24px', borderTop: '1px solid #334155', paddingTop: '16px' }}>
          <h3 style={{ fontSize: '14px', fontWeight: '700', color: '#cbd5e1', marginBottom: '10px' }}>
            🟢 Active Colab Sessions ({activeSessions.length}):
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {activeSessions.map((sess) => (
              <div key={`${sess.name}-${sess.endpoint}`} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '10px 14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontFamily: 'monospace', fontSize: '13px', color: '#34d399', fontWeight: '600' }}>
                  ✓ {sess.name} · {sess.accelerator} · {sess.status || 'ACTIVE'} · Drive: {sess.drive_mounted === true ? 'mounted' : sess.drive_mounted === false ? 'not mounted' : 'not verified'}
                </span>

                <button
                  onClick={() => {
                    handleStopSession(sess.name);
                  }}
                  style={{ background: '#991b1b', color: '#fff', border: 'none', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer', fontWeight: '600' }}
                >
                  Terminate Session
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeSessions.length === 0 && (
        <div style={{ marginTop: '24px', borderTop: '1px solid #334155', paddingTop: '16px', color: '#94a3b8', fontSize: '13px' }}>
          No active Colab sessions. Google Drive is not mounted because there is no live Colab kernel to mount it in.
        </div>
      )}
    </div>
  );
}
