'use client';

import React, { useEffect, useState } from 'react';
import DashboardNavbar from '@/components/DashboardNavbar';
import { User, authClient } from '@/lib/api/authClient';
import { integrationsClient, ColabSession, ConnectedAccount, DriveFile } from '@/lib/api/integrations-client';
import { colabClient, ExternalNotebook, ExternalRun } from '@/lib/api/colab-client';
import { StatusBadge } from '@/components/shared/status-badge';
import { LoadingState } from '@/components/shared/loading-state';
import { EmptyState } from '@/components/shared/empty-state';
import { ErrorState } from '@/components/shared/error-state';

export default function IntegrationsDashboardPage() {
  const [user, setUser] = useState<User | null>(null);
  const [activeTab, setActiveTab] = useState<'drive' | 'colab'>('drive');
  const [accounts, setAccounts] = useState<ConnectedAccount[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState<string>('');
  const [driveFiles, setDriveFiles] = useState<DriveFile[]>([]);
  const [notebooks, setNotebooks] = useState<ExternalNotebook[]>([]);
  const [colabSessions, setColabSessions] = useState<ColabSession[]>([]);
  const [activeRun, setActiveRun] = useState<ExternalRun | null>(null);
  const [quotaExhaustedModal, setQuotaExhaustedModal] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshingColab, setIsRefreshingColab] = useState(false);
  const [driveMountId, setDriveMountId] = useState<string | null>(null);
  const [driveMountUrl, setDriveMountUrl] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setIsLoading(true);
      const userData = await authClient.getCurrentUser();
      setUser(userData);

      const accs = await integrationsClient.listConnectedAccounts();
      setAccounts(accs);
      if (accs.length > 0) {
        setSelectedAccountId(accs[0].id);
        if (accs[0].status === 'active') {
          const filesRes = await integrationsClient.listDriveFiles(accs[0].id);
          setDriveFiles(filesRes.files || []);
        }
      }

      const nbs = await colabClient.listNotebooks();
      setNotebooks(nbs);
      const sessionsRes = await integrationsClient.listColabSessions();
      setColabSessions(sessionsRes.sessions || []);
      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Failed to load integrations data.');
    } finally {
      setIsLoading(false);
    }
  };

  const loadColabSessions = async () => {
    try {
      setIsRefreshingColab(true);
      const response = await integrationsClient.listColabSessions();
      setColabSessions(response.sessions || []);
      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch live Colab session status from the VM.');
    } finally {
      setIsRefreshingColab(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleAccountChange = async (accId: string) => {
    setSelectedAccountId(accId);
    const targetAcc = accounts.find(a => a.id === accId);
    if (targetAcc?.status === 'quota_exhausted') {
      setQuotaExhaustedModal(true);
      return;
    }

    try {
      setIsLoading(true);
      const filesRes = await integrationsClient.listDriveFiles(accId);
      setDriveFiles(filesRes.files || []);
      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Failed to load Drive files for account.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleImportFile = async (fileId: string, fileName: string) => {
    if (!selectedAccountId) return;
    const currentAcc = accounts.find(a => a.id === selectedAccountId);
    if (currentAcc?.status === 'quota_exhausted') {
      setQuotaExhaustedModal(true);
      return;
    }

    try {
      setActionMessage(`Queueing import for Drive file '${fileName}' using selected account...`);
      const res = await integrationsClient.importDriveFile(selectedAccountId, fileId);
      setActionMessage(`Import queued successfully! Task ID: ${res.task_id}`);
    } catch (err: any) {
      if (err?.message?.includes('quota') || err?.message?.includes('429')) {
        setQuotaExhaustedModal(true);
      }
      setError(err?.message || 'Failed to queue Drive file import.');
    }
  };

  const handleRunNotebook = async (notebookId: string, name: string) => {
    const currentAcc = accounts.find(a => a.id === selectedAccountId);
    if (currentAcc?.status === 'quota_exhausted') {
      setQuotaExhaustedModal(true);
      return;
    }

    try {
      setActionMessage(`Submitting execution for notebook '${name}' with selected_google_account_id=${selectedAccountId || 'none'}...`);
      const run = await colabClient.runNotebook(notebookId, undefined, selectedAccountId || undefined);
      setActiveRun(run);
      setActionMessage(`Execution submitted! External Run ID: ${run.external_run_id}`);
    } catch (err: any) {
      if (err?.message?.includes('quota_exhausted') || err?.message?.includes('quota')) {
        setQuotaExhaustedModal(true);
      }
      setError(err?.message || 'Failed to submit notebook run.');
    }
  };

  const handleCancelRun = async (runId: string) => {
    try {
      const run = await colabClient.cancelRun(runId);
      setActiveRun(run);
      setActionMessage('External notebook execution cancelled.');
    } catch (err: any) {
      setError(err?.message || 'Failed to cancel external run.');
    }
  };

  const validColabSessionName = (name: string) => /^[A-Za-z0-9_-]{1,64}$/.test(name);

  const handleCreateColabSession = async () => {
    try {
      const sessionName = `integration-${Date.now().toString(36)}`;
      setActionMessage('Creating a CPU Colab session on the VM...');
      const result = await integrationsClient.createColabSession({
        account_id: selectedAccountId || undefined,
        session_name: sessionName,
        gpu_variant: 'CPU',
      });
      setActionMessage(`Colab session '${result.session_name}' is ready. Refreshing live state...`);
      await loadColabSessions();
    } catch (err: any) {
      setError(err?.message || 'Failed to create Colab session. Authorize the Colab CLI first if required.');
    }
  };

  const handleStartDriveMount = async (sessionName: string) => {
    if (!validColabSessionName(sessionName)) {
      setError('Select a named, active Colab session before mounting Drive.');
      return;
    }
    try {
      setActionMessage(`Preparing Google Drive consent for '${sessionName}'...`);
      const result = await integrationsClient.startColabDriveMount(sessionName);
      setDriveMountId(result.mount_id);
      setDriveMountUrl(result.authorization_url);
      window.open(result.authorization_url, '_blank', 'noopener,noreferrer');
      setActionMessage('Drive authorization opened in a new tab. Grant access, then click Complete Drive Mount below.');
    } catch (err: any) {
      setError(err?.message || 'Could not start Google Drive mount for this Colab session.');
    }
  };

  const handleCompleteDriveMount = async () => {
    if (!driveMountId) return;
    try {
      const result = await integrationsClient.completeColabDriveMount(driveMountId);
      setActionMessage(result.message || 'Drive consent sent to Colab. Refreshing the live state...');
      setDriveMountId(null);
      setDriveMountUrl(null);
      await loadColabSessions();
    } catch (err: any) {
      setError(err?.message || 'Could not complete Google Drive mount.');
    }
  };

  const handleStopColabSession = async (sessionName: string) => {
    try {
      await integrationsClient.stopColabSession(sessionName);
      setActionMessage(`Colab session '${sessionName}' was terminated.`);
      await loadColabSessions();
    } catch (err: any) {
      setError(err?.message || `Failed to terminate Colab session '${sessionName}'.`);
    }
  };

  if (isLoading) return <LoadingState message="Loading cloud integrations..." />;

  return (
    <div style={{ background: '#090d16', minHeight: '100vh', color: '#f8fafc', fontFamily: 'system-ui' }}>
      <DashboardNavbar user={user} />

      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 24px' }}>
        <div style={{ marginBottom: '24px' }}>
          <h1 style={{ fontSize: '24px', fontWeight: '700' }}>Cloud Integrations & Connectors</h1>
          <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '4px' }}>
            Authorized Google Drive file importer and Google Cloud Colab Enterprise notebook runner with explicit account selection.
          </p>
        </div>

        {actionMessage && (
          <div style={{ background: '#0369a122', border: '1px solid #0284c7', color: '#38bdf8', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', fontSize: '14px' }}>
            {actionMessage}
          </div>
        )}

        {error && <ErrorState message={error} />}

        {/* Quota Exhausted Policy Modal */}
        {quotaExhaustedModal && (
          <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
            <div style={{ background: '#1e293b', border: '1px solid #f97316', borderRadius: '14px', padding: '28px', maxWidth: '540px', width: '90%' }}>
              <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#f97316', marginBottom: '12px' }}>
                ⚠️ Quota Exhausted Policy Alert
              </h2>
              <p style={{ color: '#cbd5e1', fontSize: '14px', lineHeight: '1.5', marginBottom: '20px' }}>
                The selected Google account has reached its API resource or rate quota limit.
                <strong> Automatic account switching is disabled by policy.</strong> Please select an approved action below:
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '24px' }}>
                <div style={{ background: '#0f172a', padding: '12px 16px', borderRadius: '8px', border: '1px solid #334155', fontSize: '13px' }}>
                  🔹 <strong>Option 1:</strong> Select a different authorized Google account manually from the dropdown.
                </div>
                <div style={{ background: '#0f172a', padding: '12px 16px', borderRadius: '8px', border: '1px solid #334155', fontSize: '13px' }}>
                  🔹 <strong>Option 2:</strong> Use Colab Enterprise connector with explicit GCP credentials.
                </div>
                <div style={{ background: '#0f172a', padding: '12px 16px', borderRadius: '8px', border: '1px solid #334155', fontSize: '13px' }}>
                  🔹 <strong>Option 3:</strong> Execute work on local VM worker nodes.
                </div>
                <div style={{ background: '#0f172a', padding: '12px 16px', borderRadius: '8px', border: '1px solid #334155', fontSize: '13px' }}>
                  🔹 <strong>Option 4:</strong> Route to an approved GPU cloud worker.
                </div>
              </div>

              <button
                onClick={() => setQuotaExhaustedModal(false)}
                style={{ background: '#0284c7', color: '#fff', border: 'none', padding: '10px 20px', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', width: '100%' }}
              >
                I Understand
              </button>
            </div>
          </div>
        )}

        {/* Global Account Selector */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '16px 20px', marginBottom: '24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <span style={{ fontSize: '14px', fontWeight: '600', color: '#f8fafc' }}>Explicit Google Account Selection (`selected_google_account_id`):</span>
            <p style={{ fontSize: '12px', color: '#94a3b8', margin: '2px 0 0 0' }}>All job requests validate that the selected account belongs to your authenticated session.</p>
          </div>

          <select
            value={selectedAccountId}
            onChange={(e) => handleAccountChange(e.target.value)}
            style={{ background: '#0f172a', color: '#fff', border: '1px solid #334155', padding: '8px 16px', borderRadius: '8px', fontSize: '14px', fontWeight: '500' }}
          >
            <option value="">-- None / Default --</option>
            {accounts.map((a) => (
              <option key={a.id} value={a.id}>
                {a.email || a.display_name} [{a.status.toUpperCase()}]
              </option>
            ))}
          </select>
        </div>

        <section style={{ background: '#172033', border: '1px solid #2563eb', borderRadius: '12px', padding: '20px', marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', marginBottom: '12px' }}>
            <div>
              <h2 style={{ fontSize: '17px', fontWeight: '700' }}>⚡ Live Colab Runtime & Drive</h2>
              <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '3px' }}>Live state from the VM&apos;s Colab CLI; this is the same runtime used by Console & Runner.</p>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button onClick={handleCreateColabSession} style={{ background: '#7c3aed', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: '7px', fontWeight: '600', cursor: 'pointer', fontSize: '12px' }}>+ Create CPU Session</button>
              <button onClick={loadColabSessions} disabled={isRefreshingColab} style={{ background: '#334155', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: '7px', fontWeight: '600', cursor: isRefreshingColab ? 'wait' : 'pointer', fontSize: '12px' }}>{isRefreshingColab ? 'Refreshing...' : '↻ Refresh Live Status'}</button>
            </div>
          </div>

          {colabSessions.filter((session) => validColabSessionName(session.name)).length === 0 ? (
            <p style={{ color: '#fbbf24', fontSize: '13px', margin: 0 }}>No named active Colab session on the VM. Create one here or from Workers/Console, then refresh.</p>
          ) : (
            <div style={{ display: 'grid', gap: '10px' }}>
              {colabSessions.filter((session) => validColabSessionName(session.name)).map((session) => (
                <div key={session.name} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '12px', display: 'flex', gap: '12px', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
                  <div>
                    <div style={{ fontFamily: 'monospace', fontWeight: '700', color: '#e2e8f0' }}>{session.name} <span style={{ color: '#34d399', fontFamily: 'system-ui', fontSize: '12px' }}>● CONNECTED</span></div>
                    <div style={{ color: '#94a3b8', fontSize: '12px', marginTop: '3px' }}>{session.accelerator || 'CPU'} · {session.variant || 'DEFAULT'} · Drive: <span style={{ color: session.drive_mounted === true ? '#34d399' : session.drive_mounted === false ? '#fbbf24' : '#94a3b8', fontWeight: '600' }}>{session.drive_mounted === true ? 'MOUNTED' : session.drive_mounted === false ? 'NOT MOUNTED' : 'CHECKING'}</span></div>
                  </div>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {session.drive_mounted !== true && <button onClick={() => handleStartDriveMount(session.name)} style={{ background: '#a16207', color: '#fff', border: 'none', padding: '7px 10px', borderRadius: '6px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}>📁 Mount Drive</button>}
                    <button onClick={() => handleStopColabSession(session.name)} style={{ background: '#991b1b', color: '#fff', border: 'none', padding: '7px 10px', borderRadius: '6px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}>Terminate</button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {driveMountUrl && driveMountId && (
            <div style={{ background: '#0c4a6e55', border: '1px solid #0284c7', borderRadius: '8px', padding: '12px', marginTop: '12px', fontSize: '13px' }}>
              <div style={{ color: '#bae6fd', marginBottom: '8px' }}>Grant Drive access in the opened Google page, then complete the mount.</div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <a href={driveMountUrl} target="_blank" rel="noreferrer" style={{ color: '#38bdf8', padding: '7px 10px', border: '1px solid #0284c7', borderRadius: '6px' }}>Open Drive authorization</a>
                <button onClick={handleCompleteDriveMount} style={{ background: '#0284c7', color: '#fff', border: 'none', padding: '7px 10px', borderRadius: '6px', fontWeight: '600', cursor: 'pointer' }}>Complete Drive Mount</button>
              </div>
            </div>
          )}
        </section>

        {/* Tab Buttons */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '24px', borderBottom: '1px solid #334155', paddingBottom: '8px' }}>
          <button
            onClick={() => setActiveTab('drive')}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: '600',
              border: 'none',
              background: activeTab === 'drive' ? '#0284c7' : 'transparent',
              color: activeTab === 'drive' ? '#fff' : '#94a3b8',
              cursor: 'pointer'
            }}
          >
            📁 Google Drive Browser ({driveFiles.length})
          </button>
          <button
            onClick={() => setActiveTab('colab')}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              fontSize: '14px',
              fontWeight: '600',
              border: 'none',
              background: activeTab === 'colab' ? '#0284c7' : 'transparent',
              color: activeTab === 'colab' ? '#fff' : '#94a3b8',
              cursor: 'pointer'
            }}
          >
            🚀 Colab Enterprise Connector ({notebooks.length})
          </button>
        </div>

        {/* Drive Tab */}
        {activeTab === 'drive' && (
          <div>
            {accounts.length === 0 ? (
              <EmptyState
                title="No Google Account Connected"
                description="Please connect an account in Settings > Connections to browse Drive files."
              />
            ) : (
              <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '12px', overflow: 'hidden' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '14px' }}>
                  <thead>
                    <tr style={{ background: '#0f172a', borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                      <th style={{ padding: '12px 16px' }}>File Name</th>
                      <th style={{ padding: '12px 16px' }}>MIME Type</th>
                      <th style={{ padding: '12px 16px' }}>Size</th>
                      <th style={{ padding: '12px 16px', textAlign: 'right' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {driveFiles.map((file) => (
                      <tr key={file.id} style={{ borderBottom: '1px solid #334155' }}>
                        <td style={{ padding: '12px 16px', fontWeight: '500' }}>📄 {file.name}</td>
                        <td style={{ padding: '12px 16px', color: '#94a3b8', fontSize: '13px' }}>{file.mimeType}</td>
                        <td style={{ padding: '12px 16px', color: '#cbd5e1' }}>{file.size ? `${(parseInt(file.size) / 1024 / 1024).toFixed(2)} MB` : 'N/A'}</td>
                        <td style={{ padding: '12px 16px', textAlign: 'right' }}>
                          <button
                            onClick={() => handleImportFile(file.id, file.name)}
                            style={{ background: '#0284c7', color: '#fff', border: 'none', padding: '6px 12px', borderRadius: '6px', fontSize: '12px', cursor: 'pointer' }}
                          >
                            Import to Dataset
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Colab Enterprise Tab */}
        {activeTab === 'colab' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '16px', marginBottom: '24px' }}>
              {notebooks.map((nb) => (
                <div key={nb.id} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <h3 style={{ fontSize: '16px', fontWeight: '600' }}>📓 {nb.display_name}</h3>
                    <p style={{ color: '#94a3b8', fontSize: '13px', fontFamily: 'monospace', margin: '4px 0' }}>
                      Resource: {nb.notebook_resource_name}
                    </p>
                    <p style={{ color: '#cbd5e1', fontSize: '12px' }}>
                      Project: {nb.project_id} | Region: {nb.region}
                    </p>
                  </div>

                  <button
                    onClick={() => handleRunNotebook(nb.id, nb.display_name)}
                    style={{ background: '#0284c7', color: '#fff', border: 'none', padding: '8px 16px', borderRadius: '6px', fontWeight: '600', cursor: 'pointer' }}
                  >
                    ▶️ Run Notebook
                  </button>
                </div>
              ))}
            </div>

            {/* Active Run Status Card */}
            {activeRun && (
              <div style={{ background: '#0f172a', border: '1px solid #0284c7', borderRadius: '12px', padding: '20px', marginTop: '24px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <h3 style={{ fontSize: '16px', fontWeight: '700' }}>External Run Monitor</h3>
                  <StatusBadge status={activeRun.status} />
                </div>
                <p style={{ color: '#94a3b8', fontSize: '13px', fontFamily: 'monospace' }}>Run ID: {activeRun.external_run_id}</p>
                <p style={{ color: '#cbd5e1', fontSize: '13px', marginTop: '6px' }}>Output URI: {activeRun.output_uri || 'Pending...'}</p>

                {activeRun.status === 'running' && (
                  <button
                    onClick={() => handleCancelRun(activeRun.id)}
                    style={{ background: '#991b1b', color: '#fff', border: 'none', padding: '6px 12px', borderRadius: '6px', fontSize: '12px', marginTop: '12px', cursor: 'pointer' }}
                  >
                    Cancel Execution
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
