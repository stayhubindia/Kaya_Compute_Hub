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

  // Colab Auth Panel State
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authLink, setAuthLink] = useState<string | null>(null);
  const [authCode, setAuthCode] = useState('');
  const [authEmail, setAuthEmail] = useState('stayhubindia@gmail.com');
  const [isVerifying, setIsVerifying] = useState(false);
  const [isFetchingLink, setIsFetchingLink] = useState(false);

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

  // Fetch Colab Auth Link
  const handleOpenAuthModal = async (presetEmail?: string) => {
    if (presetEmail) setAuthEmail(presetEmail);
    setShowAuthModal(true);
    setError(null);

    if (!authLink) {
      try {
        setIsFetchingLink(true);
        const res = await integrationsClient.getColabAuthLink();
        setAuthLink(res.auth_url);
      } catch (err: any) {
        setError('Failed to generate Colab authentication link.');
      } finally {
        setIsFetchingLink(false);
      }
    }
  };

  // Submit Authorization Code
  const handleVerifyCodeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!authCode.trim()) {
      setError('Please paste the authorization code from Google.');
      return;
    }

    setIsVerifying(true);
    setError(null);
    setActionMessage(null);

    try {
      const res = await integrationsClient.verifyColabCode({
        code: authCode.trim(),
        email: authEmail.trim() || undefined,
      });

      setActionMessage(`🎉 Colab Account [${res.email || authEmail}] verified & saved to Vault!`);
      setAuthCode('');
      setShowAuthModal(false);
      await loadData();
    } catch (err: any) {
      setError(err?.message || 'Failed to verify Colab authorization code.');
    } finally {
      setIsVerifying(false);
    }
  };

  const handleVerifyStatus = async (acc: ConnectedAccount) => {
    try {
      setActionMessage(`Verifying Vault status for [${acc.email}]...`);
      const res = await integrationsClient.verifyAccount(acc.id);
      if (res.status === 'active') {
        setActionMessage(`✅ Account [${acc.email}] is active in Vault!`);
      } else {
        // Open Auth modal directly if verification needed
        handleOpenAuthModal(acc.email);
      }
      await loadData();
    } catch (err: any) {
      handleOpenAuthModal(acc.email);
    }
  };

  const handleDisconnect = async (id: string) => {
    if (!window.confirm('Unlink this Colab account?')) return;
    try {
      await integrationsClient.disconnectAccount(id);
      setActionMessage('Account disconnected.');
      await loadData();
    } catch (err: any) {
      setError(err?.message || 'Failed to disconnect account.');
    }
  };

  const handleRevoke = async (id: string) => {
    if (!window.confirm('Remove Colab Account permanently from Vault?')) return;
    try {
      await integrationsClient.revokeAccount(id);
      setActionMessage('Account permanently removed from Vault.');
      await loadData();
    } catch (err: any) {
      setError(err?.message || 'Failed to remove account.');
    }
  };

  if (isLoading) return <LoadingState message="Loading connected Colab accounts..." />;

  return (
    <div style={{ background: '#090d16', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui' }}>
      <DashboardNavbar user={user} />

      <main style={{ maxWidth: '1000px', margin: '0 auto', padding: '32px 24px' }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <div>
            <h1 style={{ fontSize: '24px', fontWeight: '700' }}>Colab Account Vault</h1>
            <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '4px' }}>
              Authenticate & manage Colab GPU Worker accounts via Google Colab verification link.
            </p>
          </div>

          <button
            onClick={() => handleOpenAuthModal()}
            style={{
              background: '#0284c7',
              color: '#fff',
              padding: '12px 22px',
              borderRadius: '8px',
              fontWeight: '700',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: '0 4px 14px rgba(2,132,199,0.3)'
            }}
          >
            🔑 Authenticate Colab Account
          </button>
        </div>

        {actionMessage && (
          <div style={{ background: '#0369a122', border: '1px solid #0284c7', color: '#38bdf8', padding: '14px 18px', borderRadius: '10px', marginBottom: '20px', fontSize: '14px' }}>
            {actionMessage}
          </div>
        )}

        {error && <ErrorState message={error} />}

        {/* Colab Interactive Link & Verification Modal/Panel */}
        {showAuthModal && (
          <div style={{ background: '#0f172a', border: '2px solid #0284c7', borderRadius: '14px', padding: '24px', marginBottom: '28px', boxShadow: '0 10px 30px rgba(2,132,199,0.2)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: '700', color: '#38bdf8' }}>
                ⚡ Colab Google Authentication (Link & Code Verification)
              </h2>
              <button
                onClick={() => setShowAuthModal(false)}
                style={{ background: 'transparent', border: 'none', color: '#94a3b8', fontSize: '18px', cursor: 'pointer' }}
              >
                ✕
              </button>
            </div>

            <p style={{ fontSize: '13px', color: '#cbd5e1', marginBottom: '20px' }}>
              Jaise new Google Colab notebook require karta hai: Link par click karke Google account approve karein, phir screen par aane wala <strong>Authorization Code</strong> yaha paste karke verify karein!
            </p>

            {/* Step 1: Auth Link */}
            <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '16px', marginBottom: '20px' }}>
              <span style={{ fontSize: '12px', fontWeight: '700', color: '#0284c7', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Step 1: Open Google Auth Link</span>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginTop: '10px' }}>
                {isFetchingLink ? (
                  <span style={{ color: '#94a3b8', fontSize: '13px' }}>Generating Colab auth link...</span>
                ) : authLink ? (
                  <>
                    <a
                      href={authLink}
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
                      🔗 Click Here to Open Google Authorization Page ↗
                    </a>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(authLink);
                        alert('Auth link copied to clipboard!');
                      }}
                      style={{ background: '#334155', color: '#cbd5e1', border: 'none', padding: '10px 14px', borderRadius: '6px', cursor: 'pointer', fontSize: '13px' }}
                    >
                      📋 Copy Link
                    </button>
                  </>
                ) : null}
              </div>
            </div>

            {/* Step 2: Form */}
            <form onSubmit={handleVerifyCodeSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <span style={{ fontSize: '12px', fontWeight: '700', color: '#16a34a', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Step 2: Paste Authorization Code & Verify</span>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '16px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>Google / Colab Email *</label>
                  <input
                    type="email"
                    required
                    value={authEmail}
                    onChange={(e) => setAuthEmail(e.target.value)}
                    placeholder="stayhubindia@gmail.com"
                    style={{ width: '100%', padding: '10px', background: '#090d16', border: '1px solid #475569', borderRadius: '6px', color: '#fff' }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px' }}>Google Authorization Code *</label>
                  <input
                    type="text"
                    required
                    value={authCode}
                    onChange={(e) => setAuthCode(e.target.value)}
                    placeholder="Paste code from Google here (e.g., 4/1AY0e-g...)"
                    style={{ width: '100%', padding: '10px', background: '#090d16', border: '1px solid #0284c7', borderRadius: '6px', color: '#86efac', fontFamily: 'monospace', fontWeight: '600' }}
                  />
                </div>
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  type="submit"
                  disabled={isVerifying}
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
                  {isVerifying ? 'Verifying & Saving...' : '✅ Verify Code & Save to Vault'}
                </button>
                <button
                  type="button"
                  onClick={() => setShowAuthModal(false)}
                  style={{ background: '#475569', color: '#fff', padding: '12px 18px', borderRadius: '8px', border: 'none', cursor: 'pointer' }}
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
            title="No Colab Accounts Registered in Vault"
            description="Click '🔑 Authenticate Colab Account' above to generate the Google auth link and register your account."
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
                    {(acc.scopes || []).map((scope, idx) => (
                      <span key={idx} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '4px', padding: '2px 8px', fontSize: '11px', color: '#cbd5e1' }}>
                        {scope}
                      </span>
                    ))}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={() => handleVerifyStatus(acc)}
                    style={{ background: '#0284c7', color: '#fff', border: 'none', padding: '8px 14px', borderRadius: '6px', fontSize: '13px', fontWeight: '600', cursor: 'pointer' }}
                  >
                    Verify Account
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
