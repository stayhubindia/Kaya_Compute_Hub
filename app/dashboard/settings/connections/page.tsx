'use client';

import React, { useEffect, useState } from 'react';
import DashboardNavbar from '@/components/DashboardNavbar';
import { User, authClient } from '@/lib/api/authClient';
import { integrationsClient, ConnectedAccount, DriveFile } from '@/lib/api/integrations-client';
import { StatusBadge } from '@/components/shared/status-badge';
import { LoadingState } from '@/components/shared/loading-state';
import { EmptyState } from '@/components/shared/empty-state';
import { ErrorState } from '@/components/shared/error-state';
import ColabSessionLauncher from '@/components/ColabSessionLauncher';

export default function ConnectionsSettingsPage() {
  const [user, setUser] = useState<User | null>(null);
  const [accounts, setAccounts] = useState<ConnectedAccount[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showConnect, setShowConnect] = useState(false);
  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [tokenJson, setTokenJson] = useState('');
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [driveFiles, setDriveFiles] = useState<Record<string, DriveFile[]>>({});

  const loadData = async () => {
    try {
      setIsLoading(true);
      const [userData, accs] = await Promise.all([authClient.getCurrentUser(), integrationsClient.listConnectedAccounts()]);
      setUser(userData); setAccounts(accs); setError(null);
    } catch (err: any) { setError(err?.message || 'Failed to load connected accounts.'); }
    finally { setIsLoading(false); }
  };
  useEffect(() => { loadData(); }, []);

  const handleConnect = async (event: React.FormEvent) => {
    event.preventDefault(); setSaving(true); setMessage(null); setError(null);
    try {
      const account = await integrationsClient.directConnectAccount({ email, display_name: displayName, raw_json: tokenJson });
      setMessage(`Connected ${account.email || email}. Drive and Colab credentials are stored in the VM vault.`);
      setEmail(''); setDisplayName(''); setTokenJson(''); setShowConnect(false); await loadData();
    } catch (err: any) { setError(err?.message || 'Could not import the Colab token.json.'); }
    finally { setSaving(false); }
  };

  const verify = async (account: ConnectedAccount) => {
    try { const result = await integrationsClient.verifyAccount(account.id); setMessage(result.message || `Drive verified for ${account.email}.`); await loadData(); }
    catch (err: any) { setError(err?.message || 'Drive verification failed. Import a fresh token.json.'); }
  };
  const testDrive = async (account: ConnectedAccount) => {
    try { const result = await integrationsClient.listDriveFiles(account.id); setDriveFiles((current) => ({ ...current, [account.id]: result.files || [] })); setMessage(`Drive connection is working for ${account.email}.`); }
    catch (err: any) { setError(err?.message || 'Drive listing failed.'); }
  };
  const disconnect = async (id: string, remove = false) => {
    if (!window.confirm(remove ? 'Remove this account from the VM vault?' : 'Disconnect this account?')) return;
    try { await (remove ? integrationsClient.revokeAccount(id) : integrationsClient.disconnectAccount(id)); setMessage(remove ? 'Account removed from the VM vault.' : 'Account disconnected.'); await loadData(); }
    catch (err: any) { setError(err?.message || 'Account action failed.'); }
  };

  if (isLoading) return <LoadingState message="Loading Drive and Colab accounts..." />;
  return (
    <div style={{ background: '#090d16', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui' }}>
      <DashboardNavbar user={user} />
      <main style={{ maxWidth: 1000, margin: '0 auto', padding: '32px 24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, marginBottom: 24 }}>
          <div><h1 style={{ fontSize: 24, fontWeight: 700 }}>Drive &amp; Colab Account Vault</h1><p style={{ color: '#94a3b8', fontSize: 14 }}>Import the official Colab CLI <code>token.json</code> directly. The VM uses the same account for Drive files and Colab sessions.</p></div>
          <button onClick={() => setShowConnect(true)} style={{ ...buttonStyle, background: '#0284c7', padding: '12px 18px' }}>+ Connect account</button>
        </div>
        {message && <div style={{ background: '#0369a122', border: '1px solid #0284c7', color: '#7dd3fc', padding: 14, borderRadius: 10, marginBottom: 16 }}>{message}</div>}
        {error && <ErrorState message={error} />}
        {showConnect && <form onSubmit={handleConnect} style={{ background: '#0f172a', border: '1px solid #0284c7', borderRadius: 12, padding: 20, marginBottom: 24 }}>
          <h2 style={{ fontSize: 18, marginTop: 0 }}>Import account credentials</h2>
          <p style={{ color: '#94a3b8', fontSize: 13 }}>On the target Google account, run the Colab CLI login once, then paste its <code>~/.config/colab-cli/token.json</code> contents below. The VM encrypts the credential and writes a 0600 vault copy; it never needs an app client secret.</p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}><input required type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Google account email" style={inputStyle} /><input value={displayName} onChange={(e) => setDisplayName(e.target.value)} placeholder="Display name (optional)" style={inputStyle} /></div>
          <textarea required value={tokenJson} onChange={(e) => setTokenJson(e.target.value)} placeholder={'Paste token.json: {"token":"...", "refresh_token":"...", "scopes":[...]}' } rows={8} style={{ ...inputStyle, width: '100%', marginTop: 12, fontFamily: 'monospace' }} />
          <div style={{ display: 'flex', gap: 10, marginTop: 12 }}><button disabled={saving} type="submit" style={{ ...buttonStyle, background: '#16a34a' }}>{saving ? 'Saving...' : 'Save & verify account'}</button><button type="button" onClick={() => setShowConnect(false)} style={{ ...buttonStyle, background: '#475569' }}>Cancel</button></div>
        </form>}
        <ColabSessionLauncher />
        {accounts.length === 0 ? <EmptyState title="No accounts in the VM vault" description="Import a Colab CLI token.json to connect Drive and Colab." /> : <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {accounts.map((account) => <div key={account.id} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 12, padding: 18 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 14, flexWrap: 'wrap' }}><div><div style={{ display: 'flex', alignItems: 'center', gap: 10 }}><strong>{account.email || account.display_name}</strong><StatusBadge status={account.status} /></div><p style={{ color: '#94a3b8', fontSize: 12 }}>Vault ID: {account.id.slice(0, 8)} · Added {new Date(account.connected_at).toLocaleString()}</p><div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>{(account.scopes || []).map((scope) => <span key={scope} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 4, padding: '2px 7px', fontSize: 11, color: '#cbd5e1' }}>{scope}</span>)}</div></div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', flexWrap: 'wrap' }}><button onClick={() => verify(account)} style={{ ...buttonStyle, background: '#0284c7' }}>Verify Drive</button><button onClick={() => testDrive(account)} style={{ ...buttonStyle, background: '#0f766e' }}>List Drive files</button><button onClick={() => disconnect(account.id)} style={{ ...buttonStyle, background: '#475569' }}>Disconnect</button><button onClick={() => disconnect(account.id, true)} style={{ ...buttonStyle, background: '#991b1b' }}>Remove</button></div>
            </div>
            {driveFiles[account.id] && <div style={{ marginTop: 14, borderTop: '1px solid #334155', paddingTop: 10, fontSize: 13 }}><strong>Drive files ({driveFiles[account.id].length})</strong>{driveFiles[account.id].slice(0, 10).map((file) => <div key={file.id} style={{ color: '#cbd5e1', paddingTop: 5 }}>{file.name} <span style={{ color: '#64748b' }}>({file.mimeType})</span></div>)}</div>}
          </div>)}
        </div>}
      </main>
    </div>
  );
}

const inputStyle = { background: '#090d16', border: '1px solid #475569', borderRadius: 6, color: '#fff', padding: 10 };
const buttonStyle = { color: '#fff', border: 0, borderRadius: 6, padding: '9px 12px', cursor: 'pointer', fontWeight: 600 };
