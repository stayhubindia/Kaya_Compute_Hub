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

  // Link & Verification Code Flow State
  const [generatedUrl, setGeneratedUrl] = useState<string | null>(null);
  const [oauthState, setOauthState] = useState<string | null>(null);
  const [authCodeInput, setAuthCodeInput] = useState('');
  const [accountEmailInput, setAccountEmailInput] = useState('stayhubindia@gmail.com');
  const [isGeneratingLink, setIsGeneratingLink] = useState(false);
  const [isVerifyingCode, setIsVerifyingCode] = useState(false);

  // Direct Entry Form Toggle State
  const [showAddForm, setShowAddForm] = useState(false);
  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [accessToken, setAccessToken] = useState('');
  const [refreshToken, setRefreshToken] = useState('');
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

  // 1. Server generates Google Verification Link
  const handleGenerateAuthLink = async () => {
    setIsGeneratingLink(true);
    setError(null);
    setActionMessage(null);
    try {
      const res = await integrationsClient.startGoogleOAuth();
      setGeneratedUrl(res.authorization_url);
      setOauthState(res.state);
      setActionMessage('🔗 Google Authorization link generated! Click below to open in Google, then paste the returned authorization code.');
    } catch (err: any) {
      setError(err?.message || 'Failed to generate Google Authorization URL.');
    } finally {
      setIsGeneratingLink(false);
    }
  };

  // 2. User submits Google Code to complete verification
  const handleVerifyAuthCode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!authCodeInput.trim()) {
      setError('Please paste the Google Authorization Code first.');
      return;
    }

    setIsVerifyingCode(true);
    setError(null);
    setActionMessage(null);

    try {
      const res = await integrationsClient.verifyGoogleAuthCode({
        code: authCodeInput.trim(),
        state: oauthState || undefined,
        email: accountEmailInput.trim() || undefined,
      });

      setActionMessage(`🎉 Google Account [${res.email || accountEmailInput}] successfully verified and saved to Vault!`);
      setAuthCodeInput('');
      setGeneratedUrl(null);
      setOauthState(null);
      await loadData();
    } catch (err: any) {
      setError(err?.message || 'Failed to verify Google Authorization Code.');
    } finally {
      setIsVerifyingCode(false);
    }
  };

  // Direct Token Form Submit
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
      });

      setActionMessage(`🎉 Google Account [${email}] saved to Vault!`);
      setEmail('');
      setDisplayName('');
      setAccessToken('');
      setRefreshToken('');
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
      setActionMessage('Verifying account token status...');
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
    if (!window.confirm('Remove Google Account from Vault?')) return;
    try {
      await integrationsClient.revokeAccount(id);
      setActionMessage('Account removed from Vault.');
      await loadData();
    } catch (err: any) {
      setError(err?.message || 'Failed to remove account.');
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
              Connect and verify Google accounts for Drive dataset sync and Colab GPU workers.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              onClick={handleGenerateAuthLink}
              disabled={isGeneratingLink}
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
              {isGeneratingLink ? 'Generating Link...' : '🔗 Generate Google Auth Link'}
            </button>

            <button
              onClick={() => setShowAddForm(!showAddForm)}
              style={{
                background: '#334155',
                color: '#fff',
                padding: '10px 16px',
                borderRadius: '8px',
                fontWeight: '600',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              {showAddForm ? '✕ Close' : '⚙️ Direct Vault Token Input'}
            </button>
          </div>
        </div>

        {actionMessage && (
          <div style={{ background: '#0369a122', border: '1px solid #0284c7', color: '#38bdf8', padding: '14px 18px', borderRadius: '10px', marginBottom: '20px', fontSize: '14px' }}>
            {actionMessage}
          </div>
        )}

        {error && <ErrorState message={error} />}

        {/* 1. SERVER-GENERATED GOOGLE AUTH LINK & VERIFICATION CARD */}
        {generatedUrl && (
          <div style={{ background: '#0f172a', border: '2px solid #0284c7', borderRadius: '14px', padding: '24px', marginBottom: '28px', boxShadow: '0 10px 30px rgba(2,132,199,0.2)' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '700', color: '#38bdf8', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>🌐</span> Step 1: Open Google Authorization Link
            </h2>
            <p style={{ fontSize: '13px', color: '#cbd5e1', marginBottom: '16px' }}>
              Click the generated button below to open Google in a new tab, sign in with your account (e.g., <strong>stayhubindia@gmail.com</strong>), approve permissions, and copy the Google Authorization Code.
            </p>

            <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: '20px', background: '#1e293b', padding: '12px', borderRadius: '8px', border: '1px solid #334155' }}>
              <a
                href={generatedUrl}
                target="_blank"
                rel="noreferrer"
                style={{
                  background: '#0284c7',
                  color: '#fff',
                  padding: '10px 20px',
                  borderRadius: '6px',
                  fontWeight: '700',
                  textDecoration: 'none',
                  fontSize: '14px',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                🔗 Open Google Authorization Page ↗
              </a>
              <button
                onClick={() => {
                  navigator.clipboard.writeText(generatedUrl);
                  alert('Authorization URL copied to clipboard!');
                }}
                style={{ background: '#334155', color: '#cbd5e1', border: 'none', padding: '10px 14px', borderRadius: '6px', cursor: 'pointer', fontSize: '13px' }}
              >
                📋 Copy Link
              </button>
            </div>

            <h2 style={{ fontSize: '18px', fontWeight: '700', color: '#38bdf8', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>✅</span> Step 2: Paste Code & Verify Account
            </h2>

            <form onSubmit={handleVerifyAuthCode} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '14px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}>Google Account Email</label>
                  <input
                    type="email"
                    required
                    value={accountEmailInput}
                    onChange={(e) => setAccountEmailInput(e.target.value)}
                    placeholder="stayhubindia@gmail.com"
                    style={{ width: '100%', padding: '10px', background: '#090d16', border: '1px solid #475569', borderRadius: '6px', color: '#fff' }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}>Google Authorization Code *</label>
                  <input
                    type="text"
                    required
                    value={authCodeInput}
                    onChange={(e) => setAuthCodeInput(e.target.value)}
                    placeholder="Paste code from Google here (e.g., 4/1AY0e-g...)"
                    style={{ width: '100%', padding: '10px', background: '#090d16', border: '1px solid #0284c7', borderRadius: '6px', color: '#86efac', fontFamily: 'monospace', fontWeight: '600' }}
                  />
                </div>
              </div>

              <div>
                <button
                  type="submit"
                  disabled={isVerifyingCode}
                  style={{
                    background: '#16a34a',
                    color: '#fff',
                    padding: '12px 28px',
                    borderRadius: '8px',
                    fontWeight: '700',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: '14px',
                    boxShadow: '0 4px 14px rgba(22,163,74,0.3)'
                  }}
                >
                  {isVerifyingCode ? 'Verifying Code...' : '✅ Submit Code & Complete Verification'}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Direct Token Form */}
        {showAddForm && (
          <div style={{ background: '#1e293b', border: '1px solid #38bdf8', borderRadius: '12px', padding: '24px', marginBottom: '28px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px', color: '#38bdf8' }}>
              🔐 Direct Vault Token Entry
            </h2>
            <form onSubmit={handleDirectConnectSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px' }}>Google Email *</label>
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="stayhubindia@gmail.com"
                    style={{ width: '100%', padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff' }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px' }}>Account Alias</label>
                  <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="Primary Storage"
                    style={{ width: '100%', padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff' }}
                  />
                </div>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px' }}>Access Token / Auth Code (Optional)</label>
                <input
                  type="text"
                  value={accessToken}
                  onChange={(e) => setAccessToken(e.target.value)}
                  placeholder="ya29... or auth code"
                  style={{ width: '100%', padding: '10px', background: '#0f172a', border: '1px solid #475569', borderRadius: '6px', color: '#fff', fontFamily: 'monospace' }}
                />
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  style={{ background: '#0284c7', color: '#fff', padding: '10px 24px', borderRadius: '6px', fontWeight: '600', border: 'none', cursor: 'pointer' }}
                >
                  {isSubmitting ? 'Saving...' : '💾 Save to Vault'}
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

        {/* Account Cards List */}
        {accounts.length === 0 ? (
          <EmptyState
            title="No Verified Google Accounts in Vault"
            description="Click '🔗 Generate Google Auth Link' above to open Google verification, or add your account directly."
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
                    {acc.last_verified_at && ` | Verified: ${new Date(acc.last_verified_at).toLocaleTimeString()}`}
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
