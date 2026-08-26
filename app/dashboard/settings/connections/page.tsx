'use client';

import React, { useEffect, useState } from 'react';
import DashboardNavbar from '@/components/DashboardNavbar';
import { User, authClient } from '@/lib/api/authClient';
import { integrationsClient, ConnectedAccount } from '@/lib/api/integrations-client';
import { StatusBadge } from '@/components/shared/status-badge';
import { LoadingState } from '@/components/shared/loading-state';
import { EmptyState } from '@/components/shared/empty-state';
import { ErrorState } from '@/components/shared/error-state';

export default function ConnectionsSettingsPage() {
  const [user, setUser] = useState<User | null>(null);
  const [accounts, setAccounts] = useState<ConnectedAccount[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setIsLoading(true);
      const userData = await authClient.getCurrentUser();
      setUser(userData);

      const accs = await integrationsClient.listConnectedAccounts();
      setAccounts(accs);
      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Failed to load connected accounts.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleConnectGoogle = async () => {
    try {
      setActionMessage('Initiating Google OAuth authorization flow...');
      const res = await integrationsClient.startGoogleOAuth();
      if (res.authorization_url) {
        window.location.href = res.authorization_url;
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to initiate Google OAuth.');
    }
  };

  const handleReconnect = async (id: string) => {
    try {
      setActionMessage('Initiating re-authorization flow for account...');
      const res = await integrationsClient.reconnectAccount(id);
      if (res.authorization_url) {
        window.location.href = res.authorization_url;
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to initiate account reconnection.');
    }
  };

  const handleVerify = async (id: string) => {
    try {
      setActionMessage('Verifying token status with Google...');
      const res = await integrationsClient.verifyAccount(id);
      setActionMessage(`Account status: ${res.status}`);
      await loadData();
    } catch (err: any) {
      setError(err?.message || 'Failed to verify account.');
    }
  };

  const handleDisconnect = async (id: string) => {
    if (!window.confirm('Disconnect Google Account? Tokens will be cleared from Kaya.')) return;
    try {
      await integrationsClient.disconnectAccount(id);
      setActionMessage('Account disconnected.');
      await loadData();
    } catch (err: any) {
      setError(err?.message || 'Failed to disconnect account.');
    }
  };

  const handleRevoke = async (id: string) => {
    if (!window.confirm('Revoke access with Google? This will permanently invalidate access and refresh tokens.')) return;
    try {
      await integrationsClient.revokeAccount(id);
      setActionMessage('Account access revoked with Google.');
      await loadData();
    } catch (err: any) {
      setError(err?.message || 'Failed to revoke account.');
    }
  };

  if (isLoading) return <LoadingState message="Loading connected accounts..." />;

  return (
    <div style={{ background: '#090d16', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui' }}>
      <DashboardNavbar user={user} />

      <main style={{ maxWidth: '1000px', margin: '0 auto', padding: '32px 24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <div>
            <h1 style={{ fontSize: '24px', fontWeight: '700' }}>Connected Accounts & Integrations</h1>
            <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '4px' }}>
              Manage independently authorized Google accounts for Drive file access and Colab Enterprise connectors.
            </p>
          </div>

          <button
            onClick={handleConnectGoogle}
            style={{
              background: '#0284c7',
              color: '#fff',
              padding: '10px 18px',
              borderRadius: '8px',
              fontWeight: '600',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            🔑 Add Google/Colab Account
          </button>
        </div>

        {actionMessage && (
          <div style={{ background: '#0369a122', border: '1px solid #0284c7', color: '#38bdf8', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', fontSize: '14px' }}>
            {actionMessage}
          </div>
        )}

        {error && <ErrorState message={error} />}

        {/* Anti-Quota Evasion Policy Banner */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '14px 18px', marginBottom: '24px', fontSize: '13px', color: '#cbd5e1' }}>
          🔒 <strong>Security Policy Note:</strong> Multiple connected accounts are supported for authorized account management and explicit selection, not for bypassing Google Colab access limits or resource quotas.
        </div>

        {/* Account Cards */}
        {accounts.length === 0 ? (
          <EmptyState
            title="No Accounts Connected"
            description="Click 'Add Google/Colab Account' to connect an independently authorized Google account."
          />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {accounts.map((acc) => (
              <div
                key={acc.id}
                style={{
                  background: '#1e293b',
                  border: acc.status === 'quota_exhausted' ? '1px solid #f97316' : '1px solid #334155',
                  borderRadius: '12px',
                  padding: '20px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '6px' }}>
                    <span style={{ fontSize: '18px', fontWeight: '600' }}>{acc.email || acc.display_name || acc.provider_account_id}</span>
                    <StatusBadge status={acc.status} />
                  </div>

                  <p style={{ color: '#94a3b8', fontSize: '13px', margin: '4px 0' }}>
                    Provider: <strong style={{ color: '#e2e8f0' }}>{acc.provider.toUpperCase()}</strong> | Connected: {new Date(acc.connected_at).toLocaleString()}
                    {acc.last_verified_at && ` | Verified: ${new Date(acc.last_verified_at).toLocaleTimeString()}`}
                  </p>

                  <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '10px' }}>
                    {acc.scopes.map((scope, idx) => (
                      <span key={idx} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', padding: '2px 8px', fontSize: '11px', color: '#cbd5e1' }}>
                        {scope.replace('https://www.googleapis.com/auth/', '')}
                      </span>
                    ))}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={() => handleVerify(acc.id)}
                    style={{ background: '#334155', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: '6px', fontSize: '13px', cursor: 'pointer' }}
                  >
                    Verify
                  </button>
                  <button
                    onClick={() => handleReconnect(acc.id)}
                    style={{ background: '#0284c7', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: '6px', fontSize: '13px', cursor: 'pointer' }}
                  >
                    Reconnect
                  </button>
                  <button
                    onClick={() => handleDisconnect(acc.id)}
                    style={{ background: '#475569', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: '6px', fontSize: '13px', cursor: 'pointer' }}
                  >
                    Disconnect
                  </button>
                  <button
                    onClick={() => handleRevoke(acc.id)}
                    style={{ background: '#991b1b', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: '6px', fontSize: '13px', cursor: 'pointer' }}
                  >
                    Revoke
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
