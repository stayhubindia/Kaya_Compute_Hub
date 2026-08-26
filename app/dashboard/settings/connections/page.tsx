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

  // Form State for Direct Token/Vault Entry
  const [showAddForm, setShowAddForm] = useState(false);
  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [accessToken, setAccessToken] = useState('');
  const [refreshToken, setRefreshToken] = useState('');
  const [rawJson, setRawJson] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

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

  const handleDirectConnectSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) {
      setError('Google Account Email is required.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setActionMessage(null);

    try {
      await integrationsClient.directConnectAccount({
        email,
        display_name: displayName || email,
        access_token: accessToken,
        refresh_token: refreshToken,
        raw_json: rawJson,
      });

      setActionMessage(`🎉 Google Account [${email}] successfully registered in Vault!`);
      setEmail('');
      setDisplayName('');
      setAccessToken('');
      setRefreshToken('');
      setRawJson('');
      setShowAddForm(false);
      await loadData();
    } catch (err: any) {
      setError(err?.message || 'Failed to register Google Account token.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleVerify = async (id: string) => {
    try {
      setActionMessage('Verifying token status with Google Drive API...');
      const res = await integrationsClient.verifyAccount(id);
      setActionMessage(`Account status: ${res.status}`);
      await loadData();
    } catch (err: any) {
      setError(err?.message || 'Failed to verify account.');
    }
  };

  const handleDisconnect = async (id: string) => {
    if (!window.confirm('Disconnect Google Account? Account will be unlinked.')) return;
    try {
      await integrationsClient.disconnectAccount(id);
      setActionMessage('Account disconnected.');
      await loadData();
    } catch (err: any) {
      setError(err?.message || 'Failed to disconnect account.');
    }
  };

  const handleRevoke = async (id: string) => {
    if (!window.confirm('Revoke access with Google? This will remove token credentials.')) return;
    try {
      await integrationsClient.revokeAccount(id);
      setActionMessage('Account access revoked.');
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
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <div>
            <h1 style={{ fontSize: '24px', fontWeight: '700' }}>Google & Colab Account Vault</h1>
            <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '4px' }}>
              Connect Google accounts directly for Drive file syncing and Colab GPU failover.
            </p>
          </div>

          <button
            onClick={() => setShowAddForm(!showAddForm)}
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
            {showAddForm ? '✕ Close Form' : '🔑 Register Google Account Token'}
          </button>
        </div>

        {actionMessage && (
          <div style={{ background: '#0369a122', border: '1px solid #0284c7', color: '#38bdf8', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', fontSize: '14px' }}>
            {actionMessage}
          </div>
        )}

        {error && <ErrorState message={error} />}

        {/* Direct Account Entry Form */}
        {showAddForm && (
          <div style={{ background: '#1e293b', border: '1px solid #38bdf8', borderRadius: '12px', padding: '24px', marginBottom: '28px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px', color: '#38bdf8' }}>
              🔐 Register Google / Colab Account into Vault
            </h2>
            <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '20px' }}>
              No browser OAuth popup required! Input your Google Account email and optional credentials to store in the stateless token vault.
            </p>

            <form onSubmit={handleDirectConnectSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px' }}>Google Email *</label>
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="e.g. stayhubindia@gmail.com"
                    style={{ width: '100%', padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff' }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px' }}>Account Label / Alias</label>
                  <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="e.g. Primary Drive Account"
                    style={{ width: '100%', padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff' }}
                  />
                </div>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px' }}>Access Token / Authorization Code (Optional)</label>
                <input
                  type="text"
                  value={accessToken}
                  onChange={(e) => setAccessToken(e.target.value)}
                  placeholder="e.g. ya29.a0... or 4/1AY0e..."
                  style={{ width: '100%', padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff', fontFamily: 'monospace' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px' }}>Refresh Token or JSON Token Object (Optional)</label>
                <textarea
                  rows={2}
                  value={refreshToken}
                  onChange={(e) => setRefreshToken(e.target.value)}
                  placeholder="e.g. 1//0g..."
                  style={{ width: '100%', padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff', fontFamily: 'monospace' }}
                />
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  style={{ background: '#0284c7', color: '#fff', padding: '10px 24px', borderRadius: '6px', fontWeight: '600', border: 'none', cursor: 'pointer' }}
                >
                  {isSubmitting ? 'Saving to Vault...' : '💾 Save Account to Vault'}
                </button>
                <button
                  type="button"
                  onClick={() => setShowAddForm(false)}
                  style={{ background: '#475569', color: '#fff', padding: '10px 16px', borderRadius: '6px', border: 'none', cursor: 'pointer' }}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Anti-Quota Evasion & Setup Note */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '16px 20px', marginBottom: '24px', fontSize: '13px', color: '#cbd5e1' }}>
          💡 <strong>Multi-Account Drive Setup:</strong> Share your primary Google Drive folder with secondary accounts and click <em>&quot;Add shortcut to Drive&quot;</em> in secondary accounts. Any connected account in this Vault will access your models and datasets automatically without losing progress!
        </div>

        {/* Account Cards List */}
        {accounts.length === 0 ? (
          <EmptyState
            title="No Accounts Registered in Vault"
            description="Click 'Register Google Account Token' above to add stayhubindia@gmail.com or other accounts."
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
                    <span style={{ fontSize: '18px', fontWeight: '600' }}>{acc.email || acc.display_name}</span>
                    <StatusBadge status={acc.status} />
                  </div>

                  <p style={{ color: '#94a3b8', fontSize: '13px', margin: '4px 0' }}>
                    Vault Account ID: <strong style={{ color: '#e2e8f0' }}>{acc.id.slice(0, 8)}</strong> | Registered: {new Date(acc.connected_at).toLocaleString()}
                  </p>

                  <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '10px' }}>
                    {acc.scopes.map((scope, idx) => (
                      <span key={idx} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', padding: '2px 8px', fontSize: '11px', color: '#cbd5e1' }}>
                        {scope}
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
                    onClick={() => handleDisconnect(acc.id)}
                    style={{ background: '#475569', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: '6px', fontSize: '13px', cursor: 'pointer' }}
                  >
                    Disconnect
                  </button>
                  <button
                    onClick={() => handleRevoke(acc.id)}
                    style={{ background: '#991b1b', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: '6px', fontSize: '13px', cursor: 'pointer' }}
                  >
                    Remove
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
