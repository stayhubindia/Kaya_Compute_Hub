'use client';

import React, { useState, useEffect, useRef } from 'react';
import DashboardNavbar from '@/components/DashboardNavbar';
import { User, authClient } from '@/lib/api/authClient';
import { api } from '@/lib/api/client';
import { ConnectedAccount, integrationsClient } from '@/lib/api/integrations-client';

const TEMPLATE_SCRIPTS: Record<string, { name: string; code: string }> = {
  arxiv_test: {
    name: 'ArXiv Batch Downloader Test Script',
    code: `import os
import urllib.request

output_dir = "/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv/2024/pdf"
os.makedirs(output_dir, exist_ok=True)

test_url = "https://arxiv.org/pdf/2401.00001.pdf"
target_file = os.path.join(output_dir, "2401.00001.pdf")

print(f"[TEST] Downloading test paper from {test_url}...")
try:
    req = urllib.request.Request(test_url, headers={'User-Agent': 'KayaComputeHub/1.0'})
    with urllib.request.urlopen(req) as response, open(target_file, 'wb') as f:
        f.write(response.read())
    file_size = os.path.getsize(target_file)
    print(f"[SUCCESS] Downloaded to {target_file} ({file_size} bytes)")
except Exception as e:
    print(f"[ERROR] Failed to download: {e}")
`
  },
  pytorch_check: {
    name: 'PyTorch CUDA & Hardware Diagnostics',
    code: `import torch
import sys
import os

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
else:
    print("Running on CPU mode.")

drive_path = "/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv"
print(f"Drive Path Exists ({drive_path}): {os.path.exists(drive_path)}")
`
  },
  drive_inspector: {
    name: 'Google Drive Datasets Inspector',
    code: `import os

base_path = "/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv"
print(f"=== Inspecting Colab Drive: {base_path} ===")

if not os.path.exists(base_path):
    print(f"[WARNING] Path {base_path} does not exist.")
else:
    total_files = 0
    total_size = 0
    for root, dirs, files in os.walk(base_path):
        for f in files:
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            total_files += 1
            total_size += size
            rel_path = os.path.relpath(fp, base_path)
            print(f" - {rel_path} ({size} bytes)")
            
    print(f"\n[SUMMARY] Total Files: {total_files} | Total Storage Used: {total_size / (1024*1024):.2f} MB")
`
  },
  colab_remote_exec: {
    name: 'Remote Colab Session Executor (via colab-cli)',
    code: `import subprocess

# Dispatch Python code to run inside active Google Colab GPU Container
code_to_run = """import os
import sys

print("[COLAB CONTAINER] Running in Google Colab Session!")
print(f"Python Version: {sys.version}")
print(f"Current Directory: {os.getcwd()}")
"""

try:
    res = subprocess.run(["colab", "exec"], input=code_to_run, text=True, capture_output=True, timeout=30)
    print("=== Colab Remote Output ===")
    print(res.stdout)
    if res.stderr:
        print("=== Colab Stderr ===")
        print(res.stderr)
except Exception as e:
    print(f"[ERROR] Execution failed: {e}")
`
  }
};

export default function ConsolePage() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Terminal State
  const [terminalTarget, setTerminalTarget] = useState<'colab' | 'vm'>('colab');
  const [commandInput, setCommandInput] = useState('');
  const [terminalLogs, setTerminalLogs] = useState<Array<{ timestamp: string; type: 'cmd' | 'stdout' | 'stderr' | 'system'; text: string }>>([
    { timestamp: new Date().toLocaleTimeString(), type: 'system', text: 'Kaya Compute Colab Interactive Terminal v2.0 connected.\n[TARGET: 🌐 Google Colab Cloud Container (root@colab-cloud:#)]\nType any bash command (whoami, pwd, ls -la, nvidia-smi) to execute directly inside your Colab container.' }
  ]);
  const [isExecutingCmd, setIsExecutingCmd] = useState(false);
  const [driveMountId, setDriveMountId] = useState<string | null>(null);
  const [driveMountUrl, setDriveMountUrl] = useState<string | null>(null);

  // Interactive Colab REPL State
  const [replInput, setReplInput] = useState('');
  const [replCount, setReplCount] = useState(1);
  const [replHistory, setReplHistory] = useState<Array<{ count: number; code: string; stdout: string; stderr: string; time: string }>>([
    { count: 1, code: 'import os\nprint("[COLAB KERNEL READY] Connected to active Google Colab session!")', stdout: '[COLAB KERNEL READY] Connected to active Google Colab session!\n', stderr: '', time: new Date().toLocaleTimeString() }
  ]);
  const [isExecutingRepl, setIsExecutingRepl] = useState(false);
  const replEndRef = useRef<HTMLDivElement>(null);

  // Active Console View Mode: 'repl' | 'terminal' | 'runner'
  const [activeTab, setActiveTab] = useState<'repl' | 'terminal' | 'runner'>('repl');

  useEffect(() => {
    replEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [replHistory]);

  // Code Runner State
  const [selectedTemplate, setSelectedTemplate] = useState('arxiv_test');
  const [scriptName, setScriptName] = useState('ArXiv Test Script');
  const [codeContent, setCodeContent] = useState(TEMPLATE_SCRIPTS.arxiv_test.code);
  const [executionMode, setExecutionMode] = useState<'instant' | 'job'>('instant');
  const [accounts, setAccounts] = useState<ConnectedAccount[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState('');
  const [sessionName, setSessionName] = useState('kaya-colab-worker');
  const [accelerator, setAccelerator] = useState('T4');
  const [targetDir, setTargetDir] = useState('/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv');
  const [isExecutingCode, setIsExecutingCode] = useState(false);
  const [codeOutput, setCodeOutput] = useState<{ status?: string; stdout?: string; stderr?: string; execution_time?: string; job_id?: string; message?: string } | null>(null);

  const terminalEndRef = useRef<HTMLDivElement>(null);

  const handleRunColabRepl = async (codeToRun?: string) => {
    const code = (codeToRun || replInput).trim();
    if (!code) return;

    const currentCount = replCount;
    setReplCount(prev => prev + 1);
    if (!codeToRun) setReplInput('');
    setIsExecutingRepl(true);

    try {
      const data: any = await api.post('/console/terminal/', {
        command: 'colab exec',
        stdin_input: code + '\n'
      });

      const stdout = data.stdout || '';
      const stderr = data.stderr || '';

      setReplHistory(prev => [
        ...prev,
        { count: currentCount + 1, code, stdout, stderr, time: new Date().toLocaleTimeString() }
      ]);
    } catch (err: any) {
      setReplHistory(prev => [
        ...prev,
        { count: currentCount + 1, code, stdout: '', stderr: err.message || 'Kernel execution error', time: new Date().toLocaleTimeString() }
      ]);
    } finally {
      setIsExecutingRepl(false);
    }
  };

  useEffect(() => {
    async function loadUser() {
      try {
        const u = await authClient.getCurrentUser();
        setUser(u);
        const connected = await integrationsClient.listConnectedAccounts();
        const active = connected.filter((account) => account.status === 'active');
        setAccounts(active);
        setSelectedAccountId(active[0]?.id || '');
      } catch {
        // Handled by API middleware
      } finally {
        setLoading(false);
      }
    }
    loadUser();
  }, []);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [terminalLogs]);

  const handleTemplateChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const key = e.target.value;
    setSelectedTemplate(key);
    if (TEMPLATE_SCRIPTS[key]) {
      setScriptName(TEMPLATE_SCRIPTS[key].name);
      setCodeContent(TEMPLATE_SCRIPTS[key].code);
    }
  };

  const handleRunCommand = async (cmdToRun?: string, stdinInput?: string) => {
    const rawCmd = (cmdToRun || commandInput).trim();
    if (!rawCmd) return;

    if (rawCmd === 'clear') {
      setTerminalLogs([]);
      setCommandInput('');
      return;
    }

    const isColabCli = rawCmd.startsWith('colab');
    const isCloudTarget = terminalTarget === 'colab' && !isColabCli;

    const timeStr = new Date().toLocaleTimeString();
    const promptPrefix = isCloudTarget ? 'root@colab-cloud:#' : 'durgesh@kaya-vm:$';
    setTerminalLogs(prev => [...prev, { timestamp: timeStr, type: 'cmd', text: `${promptPrefix} ${rawCmd}${stdinInput ? ' [input attached]' : ''}` }]);
    if (!cmdToRun) setCommandInput('');
    setIsExecutingCmd(true);

    try {
      let payload: any;
      if (isCloudTarget) {
        // Execute bash command inside Google Colab cloud container via colab exec
        const pyWrapper = `import subprocess\nprint(subprocess.getoutput('''${rawCmd}'''))\n`;
        payload = { command: 'colab exec', stdin_input: pyWrapper };
      } else {
        payload = { command: rawCmd };
        if (stdinInput) payload.stdin_input = stdinInput;
      }

      const data: any = await api.post('/console/terminal/', payload);
      if (data.stdout) {
        setTerminalLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), type: 'stdout', text: data.stdout }]);
      }
      if (data.stderr) {
        setTerminalLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), type: 'stderr', text: data.stderr }]);
      }
      if (!data.stdout && !data.stderr) {
        setTerminalLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), type: 'system', text: `Command completed with exit code ${data.returncode} (${data.execution_time_ms}ms)` }]);
      }
    } catch (err: any) {
      setTerminalLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), type: 'stderr', text: err.message || 'Execution error' }]);
    } finally {
      setIsExecutingCmd(false);
    }
  };

  const handleMountDrive = async () => {
    setIsExecutingCmd(true);
    try {
      const sessions = await integrationsClient.listColabSessions();
      const active = sessions.sessions[0];
      if (!active) throw new Error('Create a Colab session first.');
      const mount = await integrationsClient.startColabDriveMount(active.name);
      setDriveMountId(mount.mount_id);
      setDriveMountUrl(mount.authorization_url);
      window.open(mount.authorization_url, '_blank', 'noopener,noreferrer');
      setTerminalLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), type: 'system', text: 'Drive authorization opened in a new tab. Grant access, then click Complete Drive Mount below.' }]);
    } catch (err: any) {
      setTerminalLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), type: 'stderr', text: err?.message || 'Could not start Drive mount.' }]);
    } finally {
      setIsExecutingCmd(false);
    }
  };

  const handleCompleteDriveMount = async () => {
    if (!driveMountId) return;
    setIsExecutingCmd(true);
    try {
      const result = await integrationsClient.completeColabDriveMount(driveMountId);
      setDriveMountId(null);
      setDriveMountUrl(null);
      setTerminalLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), type: 'system', text: result.message }]);
    } catch (err: any) {
      setTerminalLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), type: 'stderr', text: err?.message || 'Could not complete Drive mount.' }]);
    } finally {
      setIsExecutingCmd(false);
    }
  };

  const handleCreateNamedConsoleSession = async () => {
    if (!selectedAccountId) {
      setTerminalLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), type: 'stderr', text: 'Authorize a Colab account first.' }]);
      return;
    }
    setIsExecutingCmd(true);
    const name = `console-${Math.random().toString(36).slice(2, 8)}`;
    try {
      const result = await integrationsClient.createColabSession({ account_id: selectedAccountId, session_name: name, gpu_variant: 'CPU' });
      setTerminalLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), type: 'stdout', text: `${result.message}\nSession name: ${name}` }]);
    } catch (err: any) {
      setTerminalLogs(prev => [...prev, { timestamp: new Date().toLocaleTimeString(), type: 'stderr', text: err?.message || 'Could not create Colab session.' }]);
    } finally {
      setIsExecutingCmd(false);
    }
  };

  const handleExecuteScript = async () => {
    if (!codeContent.trim()) return;

    setIsExecutingCode(true);
    setCodeOutput(null);

    try {
      const data: any = await api.post('/console/execute/', {
        code: codeContent,
        mode: executionMode,
        script_name: scriptName,
        target_dir: targetDir
        ,execution_target: executionMode === 'job' ? 'colab' : 'vm'
        ,selected_google_account_id: selectedAccountId || undefined
        ,session_name: sessionName
        ,accelerator
      });

      setCodeOutput(data);

      // Append instant output to terminal console as well
      if (executionMode === 'instant') {
        const timeStr = new Date().toLocaleTimeString();
        setTerminalLogs(prev => [
          ...prev,
          { timestamp: timeStr, type: 'system', text: `=== Running Test Script: ${scriptName} (${data.execution_time || ''}) ===` },
          ...(data.stdout ? [{ timestamp: timeStr, type: 'stdout' as const, text: data.stdout }] : []),
          ...(data.stderr ? [{ timestamp: timeStr, type: 'stderr' as const, text: data.stderr }] : []),
        ]);
      }
    } catch (err: any) {
      setCodeOutput({ status: 'failed', stderr: err.message || 'Script execution failed' });
    } finally {
      setIsExecutingCode(false);
    }
  };

  if (loading) {
    return (
      <div style={{ background: '#0f172a', color: '#fff', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        Loading Console...
      </div>
    );
  }

  return (
    <div style={{ background: '#090d16', color: '#f8fafc', minHeight: '100vh', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <DashboardNavbar user={user} />

      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '24px' }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <div>
            <h1 style={{ fontSize: '24px', fontWeight: '700', color: '#38bdf8', margin: '0 0 4px 0' }}>
              Colab Console & Code Execution Runner
            </h1>
            <p style={{ fontSize: '14px', color: '#94a3b8', margin: 0 }}>
              Execute interactive bash commands & dispatch test scripts or background jobs on Colab Drive.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <span style={{ background: '#064e3b', color: '#34d399', padding: '6px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: '600', border: '1px solid #059669' }}>
              ● Colab Drive Connected
            </span>
            <span style={{ background: '#1e293b', color: '#38bdf8', padding: '6px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: '600', border: '1px solid #334155' }}>
              Python 3.14 Environment
            </span>
          </div>
        </div>

        {/* Navigation Tabs Bar */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '20px', borderBottom: '1px solid #1e293b', paddingBottom: '12px' }}>
          <button
            onClick={() => setActiveTab('repl')}
            style={{
              background: activeTab === 'repl' ? 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)' : '#0f172a',
              color: '#fff',
              border: activeTab === 'repl' ? '1px solid #38bdf8' : '1px solid #1e293b',
              borderRadius: '8px',
              padding: '10px 18px',
              fontWeight: '700',
              fontSize: '14px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: activeTab === 'repl' ? '0 4px 12px rgba(56,189,248,0.25)' : 'none'
            }}
          >
            <span>🐍 Interactive Colab REPL</span>
            <span style={{ fontSize: '10px', background: '#0284c7', color: '#fff', padding: '2px 6px', borderRadius: '10px' }}>Stateful</span>
          </button>

          <button
            onClick={() => setActiveTab('terminal')}
            style={{
              background: activeTab === 'terminal' ? 'linear-gradient(135deg, #4c1d95 0%, #581c87 100%)' : '#0f172a',
              color: '#fff',
              border: activeTab === 'terminal' ? '1px solid #c084fc' : '1px solid #1e293b',
              borderRadius: '8px',
              padding: '10px 18px',
              fontWeight: '700',
              fontSize: '14px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: activeTab === 'terminal' ? '0 4px 12px rgba(192,132,252,0.25)' : 'none'
            }}
          >
            <span>🖥️ Colab Terminal Console</span>
            <span style={{ fontSize: '10px', background: '#6b21a8', color: '#fff', padding: '2px 6px', borderRadius: '10px' }}>Bash Shell</span>
          </button>

          <button
            onClick={() => setActiveTab('runner')}
            style={{
              background: activeTab === 'runner' ? 'linear-gradient(135deg, #059669 0%, #047857 100%)' : '#0f172a',
              color: '#fff',
              border: activeTab === 'runner' ? '1px solid #34d399' : '1px solid #1e293b',
              borderRadius: '8px',
              padding: '10px 18px',
              fontWeight: '700',
              fontSize: '14px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: activeTab === 'runner' ? '0 4px 12px rgba(52,211,153,0.25)' : 'none'
            }}
          >
            <span>⚡ Script Sandbox & Jobs</span>
            <span style={{ fontSize: '10px', background: '#047857', color: '#fff', padding: '2px 6px', borderRadius: '10px' }}>Async Jobs</span>
          </button>
        </div>

        {/* TAB 1: INTERACTIVE COLAB REPL (JUPYTER CELL STYLE) */}
        {activeTab === 'repl' && (
          <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px', marginBottom: '24px', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.5)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '16px', fontWeight: '700', color: '#38bdf8' }}>
                  🐍 Interactive Colab Kernel REPL
                </span>
                <span style={{ fontSize: '11px', background: '#064e3b', color: '#34d399', padding: '2px 8px', borderRadius: '12px', border: '1px solid #059669' }}>
                  ● Active Session Kernel Memory Preserved
                </span>
              </div>

              {/* REPL Quick Snippet Actions */}
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <button
                  onClick={() => handleRunColabRepl("import torch\nprint('CUDA Available:', torch.cuda.is_available())\nif torch.cuda.is_available(): print('GPU Device:', torch.cuda.get_device_name(0))")}
                  style={{ background: '#1e293b', color: '#38bdf8', border: '1px solid #334155', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}
                >
                  ⚡ Check GPU
                </button>
                <button
                  onClick={handleMountDrive}
                  style={{ background: '#1e293b', color: '#34d399', border: '1px solid #334155', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}
                >
                  📁 Mount Drive
                </button>
                <button
                  onClick={() => handleRunColabRepl("import os\nprint('Current Dir:', os.getcwd())\nprint('Content Files:', os.listdir('/content'))")}
                  style={{ background: '#1e293b', color: '#fbbf24', border: '1px solid #334155', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}
                >
                  📂 List Files
                </button>
                <button
                  onClick={() => handleRunColabRepl("import psutil\nprint(f'RAM Usage: {psutil.virtual_memory().percent}% | Total: {psutil.virtual_memory().total / (1024**3):.2f} GB')")}
                  style={{ background: '#1e293b', color: '#a78bfa', border: '1px solid #334155', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}
                >
                  📊 RAM & Disk
                </button>
                <button
                  onClick={() => handleRunCommand('colab restart-kernel')}
                  style={{ background: '#7f1d1d', color: '#fca5a5', border: '1px solid #991b1b', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}
                >
                  🔄 Restart Kernel
                </button>
                <button
                  onClick={() => setReplHistory([])}
                  style={{ background: '#334155', color: '#cbd5e1', border: 'none', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', cursor: 'pointer' }}
                >
                  🧹 Clear REPL
                </button>
              </div>
            </div>

            {/* REPL History Feed (Jupyter Notebook Cell Style) */}
            <div style={{ background: '#020617', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px', fontFamily: 'monospace', fontSize: '13px', minHeight: '320px', maxHeight: '480px', overflowY: 'auto', marginBottom: '16px' }}>
              {replHistory.map((item, idx) => (
                <div key={idx} style={{ marginBottom: '16px', borderBottom: '1px solid #0f172a', paddingBottom: '12px' }}>
                  {/* Cell Input */}
                  <div style={{ display: 'flex', gap: '12px', marginBottom: '6px', alignItems: 'flex-start' }}>
                    <span style={{ color: '#38bdf8', fontWeight: '700', minWidth: '70px', userSelect: 'none' }}>
                      In [{item.count}]:
                    </span>
                    <pre style={{ margin: 0, color: '#f8fafc', background: '#0f172a', border: '1px solid #1e293b', borderRadius: '6px', padding: '8px 12px', flex: 1, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                      {item.code}
                    </pre>
                  </div>

                  {/* Cell Output */}
                  {(item.stdout || item.stderr) && (
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                      <span style={{ color: '#34d399', fontWeight: '700', minWidth: '70px', userSelect: 'none' }}>
                        Out [{item.count}]:
                      </span>
                      <div style={{ flex: 1 }}>
                        {item.stdout && (
                          <pre style={{ margin: '0 0 4px 0', color: '#e2e8f0', background: '#020617', padding: '6px 10px', borderRadius: '4px', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                            {item.stdout}
                          </pre>
                        )}
                        {item.stderr && (
                          <pre style={{ margin: 0, color: '#fca5a5', background: '#451a1a', padding: '6px 10px', borderRadius: '4px', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                            {item.stderr}
                          </pre>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
              {isExecutingRepl && (
                <div style={{ display: 'flex', gap: '12px', color: '#38bdf8', fontStyle: 'italic' }}>
                  <span style={{ fontWeight: '700', minWidth: '70px' }}>In [*]:</span>
                  <span>Executing code in Colab Kernel...</span>
                </div>
              )}
              <div ref={replEndRef} />
            </div>

            {/* Interactive Cell Form Input */}
            <form onSubmit={(e) => { e.preventDefault(); handleRunColabRepl(); }}>
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                <span style={{ color: '#38bdf8', fontWeight: '700', fontSize: '14px', fontFamily: 'monospace' }}>
                  In [{replCount}]:
                </span>
                <textarea
                  value={replInput}
                  onChange={(e) => setReplInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (e.shiftKey || e.ctrlKey)) {
                      e.preventDefault();
                      handleRunColabRepl();
                    }
                  }}
                  placeholder="Enter Python code... (Press Shift+Enter or click Run Cell)"
                  rows={2}
                  style={{
                    flex: 1,
                    background: '#020617',
                    border: '1px solid #334155',
                    borderRadius: '6px',
                    padding: '10px 14px',
                    color: '#fff',
                    fontFamily: 'Consolas, Monaco, monospace',
                    fontSize: '14px',
                    resize: 'vertical'
                  }}
                />
                <button
                  type="submit"
                  disabled={isExecutingRepl}
                  style={{
                    background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '6px',
                    padding: '0 24px',
                    height: '52px',
                    fontWeight: '700',
                    fontSize: '14px',
                    cursor: isExecutingRepl ? 'not-allowed' : 'pointer',
                    boxShadow: '0 4px 12px rgba(2,132,199,0.3)'
                  }}
                >
                  {isExecutingRepl ? 'Running...' : '▶ Run Cell'}
                </button>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '12px', color: '#64748b' }}>
                <span>💡 Tip: Variables and imports remain alive in active Colab memory across cell executions.</span>
                <span>Shortcut: Shift + Enter</span>
              </div>
            </form>
          </div>
        )}

        {/* TAB 2: COLAB TERMINAL CONSOLE (BASH SHELL) */}
        {activeTab === 'terminal' && (
          <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px', marginBottom: '24px', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.5)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '16px', fontWeight: '700', color: '#f1f5f9' }}>
                🖥️ Colab Terminal Console
              </span>
              
              {/* Target Selector */}
              <select
                value={terminalTarget}
                onChange={(e) => setTerminalTarget(e.target.value as 'colab' | 'vm')}
                style={{
                  background: terminalTarget === 'colab' ? '#064e3b' : '#1e293b',
                  color: terminalTarget === 'colab' ? '#34d399' : '#38bdf8',
                  border: terminalTarget === 'colab' ? '1px solid #059669' : '1px solid #334155',
                  borderRadius: '6px',
                  padding: '4px 10px',
                  fontSize: '12px',
                  fontWeight: '700',
                  cursor: 'pointer'
                }}
              >
                <option value="colab">🌐 Google Colab Cloud Container (root@colab-cloud:#)</option>
                <option value="vm">💻 Host VM Server (durgesh@kaya-vm:$)</option>
              </select>
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button onClick={() => handleRunCommand('colab sessions')} style={{ background: '#1e1b4b', color: '#818cf8', border: '1px solid #4338ca', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}>
                Colab Sessions
              </button>
              <button onClick={handleCreateNamedConsoleSession} style={{ background: '#4c1d95', color: '#c084fc', border: '1px solid #6b21a8', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}>
                + Create Named Colab Session
              </button>
              <button onClick={() => handleRunCommand('colab status')} style={{ background: '#1e293b', color: '#38bdf8', border: '1px solid #334155', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', cursor: 'pointer' }}>
                Colab Status
              </button>
              <button onClick={() => handleRunCommand('nvidia-smi')} style={{ background: '#1e293b', color: '#34d399', border: '1px solid #334155', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', cursor: 'pointer' }}>
                GPU Status
              </button>
              <button onClick={handleMountDrive} style={{ background: '#1e293b', color: '#fbbf24', border: '1px solid #334155', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}>
                📁 Mount Drive
              </button>
              <button onClick={() => handleRunCommand('colab restart-kernel')} style={{ background: '#7f1d1d', color: '#fca5a5', border: '1px solid #991b1b', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}>
                🔄 Restart Kernel
              </button>
              <button onClick={() => handleRunCommand('clear')} style={{ background: '#334155', color: '#cbd5e1', border: 'none', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', cursor: 'pointer' }}>
                Clear Log
              </button>
            </div>
          </div>

          {driveMountUrl && (
            <div style={{ marginBottom: '12px', padding: '10px', borderRadius: '8px', background: '#14532d22', border: '1px solid #16a34a', color: '#dcfce7', fontSize: '12px' }}>
              <a href={driveMountUrl} target="_blank" rel="noreferrer" style={{ color: '#86efac', fontWeight: '700', wordBreak: 'break-all' }}>Open Google Drive authorization</a>
              <button onClick={handleCompleteDriveMount} disabled={isExecutingCmd} style={{ marginLeft: '12px', background: '#16a34a', color: '#fff', border: 'none', borderRadius: '6px', padding: '5px 9px', fontWeight: '600', cursor: 'pointer' }}>Complete Drive Mount</button>
            </div>
          )}

          {/* Terminal Viewport */}
          <div style={{ background: '#020617', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px', fontFamily: 'monospace', fontSize: '13px', height: '260px', overflowY: 'auto', marginBottom: '12px' }}>
            {terminalLogs.map((log, i) => (
              <div key={i} style={{ marginBottom: '6px', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                <span style={{ color: '#475569', fontSize: '11px', marginRight: '8px' }}>[{log.timestamp}]</span>
                {log.type === 'cmd' && <span style={{ color: '#38bdf8', fontWeight: 'bold' }}>{log.text}</span>}
                {log.type === 'stdout' && <span style={{ color: '#e2e8f0' }}>{log.text}</span>}
                {log.type === 'stderr' && <span style={{ color: '#fca5a5' }}>{log.text}</span>}
                {log.type === 'system' && <span style={{ color: '#34d399', fontStyle: 'italic' }}>{log.text}</span>}
              </div>
            ))}
            {isExecutingCmd && <div style={{ color: '#38bdf8' }}>Running command...</div>}
            <div ref={terminalEndRef} />
          </div>

          {/* Terminal Command Input */}
          <form onSubmit={(e) => { e.preventDefault(); handleRunCommand(); }} style={{ display: 'flex', gap: '10px' }}>
            <span style={{
              background: terminalTarget === 'colab' ? '#064e3b' : '#1e293b',
              color: terminalTarget === 'colab' ? '#34d399' : '#38bdf8',
              border: terminalTarget === 'colab' ? '1px solid #059669' : '1px solid #334155',
              padding: '10px 14px',
              borderRadius: '6px',
              fontFamily: 'monospace',
              fontSize: '13px',
              fontWeight: '700',
              display: 'flex',
              alignItems: 'center'
            }}>
              {terminalTarget === 'colab' ? 'root@colab-cloud:#' : 'durgesh@kaya-vm:$'}
            </span>
            <input
              type="text"
              value={commandInput}
              onChange={(e) => setCommandInput(e.target.value)}
              placeholder="Enter bash command e.g. ls -lh, python script.py, nvidia-smi..."
              style={{
                flex: 1,
                background: '#020617',
                border: '1px solid #334155',
                borderRadius: '6px',
                padding: '10px 14px',
                color: '#fff',
                fontFamily: 'monospace',
                fontSize: '14px'
              }}
            />
            <button
              type="submit"
              disabled={isExecutingCmd}
              style={{
                background: '#0284c7',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                padding: '0 20px',
                fontWeight: '600',
                cursor: isExecutingCmd ? 'not-allowed' : 'pointer'
              }}
            >
              {isExecutingCmd ? 'Executing...' : 'Run Command'}
            </button>
          </form>
        </div>
        )}

        {/* Section 2: Code Execution Sandbox & Script Runner */}
        {activeTab === 'runner' && (
        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '24px', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.5)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <div>
              <h2 style={{ fontSize: '18px', fontWeight: '700', color: '#f1f5f9', margin: '0 0 4px 0' }}>
                ⚡ Python Code Execution & Script Dispatcher
              </h2>
              <p style={{ fontSize: '13px', color: '#94a3b8', margin: 0 }}>
                Write Python scripts to test features instantly or dispatch as background Compute Jobs.
              </p>
            </div>

            {/* Template Selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label style={{ fontSize: '13px', color: '#cbd5e1' }}>Template:</label>
              <select
                value={selectedTemplate}
                onChange={handleTemplateChange}
                style={{
                  background: '#1e293b',
                  color: '#38bdf8',
                  border: '1px solid #334155',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  fontSize: '13px',
                  fontWeight: '500'
                }}
              >
                <option value="arxiv_test">ArXiv Downloader Test Script</option>
                <option value="pytorch_check">PyTorch & CUDA Diagnostics</option>
                <option value="drive_inspector">Google Drive Dataset Inspector</option>
                <option value="colab_remote_exec">Remote Colab Session Executor (colab exec)</option>
              </select>
            </div>
          </div>

          {/* Form Options: Mode & Target Directory */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '16px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
                Execution Mode:
              </label>
              <select
                value={executionMode}
                onChange={(e) => setExecutionMode(e.target.value as 'instant' | 'job')}
                style={{
                  width: '100%',
                  background: '#020617',
                  color: executionMode === 'instant' ? '#38bdf8' : '#a78bfa',
                  border: '1px solid #334155',
                  borderRadius: '6px',
                  padding: '10px 12px',
                  fontSize: '13px',
                  fontWeight: '600'
                }}
              >
                <option value="instant">⚡ Direct Test Script (Instant Run)</option>
                <option value="job">🚀 Background Compute Job (Orchestrated Job)</option>
              </select>
            </div>

            {executionMode === 'job' && (
              <>
                <div>
                  <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
                    Authorized Google Account:
                  </label>
                  <select
                    value={selectedAccountId}
                    onChange={(e) => setSelectedAccountId(e.target.value)}
                    style={{ width: '100%', background: '#020617', color: '#34d399', border: '1px solid #334155', borderRadius: '6px', padding: '10px 12px', fontSize: '13px' }}
                  >
                    <option value="">-- Connect an account first --</option>
                    {accounts.map((account) => (
                      <option key={account.id} value={account.id}>{account.email || account.display_name}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
                    Persistent Session:
                  </label>
                  <input
                    value={sessionName}
                    onChange={(e) => setSessionName(e.target.value)}
                    style={{ width: '100%', background: '#020617', color: '#38bdf8', border: '1px solid #334155', borderRadius: '6px', padding: '10px 12px', fontSize: '13px' }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
                    Colab Accelerator:
                  </label>
                  <select
                    value={accelerator}
                    onChange={(e) => setAccelerator(e.target.value)}
                    style={{ width: '100%', background: '#020617', color: '#fbbf24', border: '1px solid #334155', borderRadius: '6px', padding: '10px 12px', fontSize: '13px' }}
                  >
                    <option value="T4">NVIDIA T4</option>
                    <option value="L4">NVIDIA L4</option>
                    <option value="A100">NVIDIA A100</option>
                    <option value="TPU">Google TPU</option>
                    <option value="CPU">CPU</option>
                  </select>
                </div>
              </>
            )}

            <div>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
                Script Name:
              </label>
              <input
                type="text"
                value={scriptName}
                onChange={(e) => setScriptName(e.target.value)}
                style={{
                  width: '100%',
                  background: '#020617',
                  color: '#fff',
                  border: '1px solid #334155',
                  borderRadius: '6px',
                  padding: '10px 12px',
                  fontSize: '13px'
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
                Target Output Directory:
              </label>
              <input
                type="text"
                value={targetDir}
                onChange={(e) => setTargetDir(e.target.value)}
                style={{
                  width: '100%',
                  background: '#020617',
                  color: '#34d399',
                  border: '1px solid #334155',
                  borderRadius: '6px',
                  padding: '10px 12px',
                  fontSize: '13px',
                  fontFamily: 'monospace'
                }}
              />
            </div>
          </div>

          {/* Python Code Editor Textarea */}
          <div style={{ marginBottom: '16px' }}>
            <textarea
              value={codeContent}
              onChange={(e) => setCodeContent(e.target.value)}
              rows={12}
              style={{
                width: '100%',
                background: '#020617',
                border: '1px solid #334155',
                borderRadius: '8px',
                padding: '16px',
                color: '#f8fafc',
                fontFamily: 'Consolas, Monaco, "Andale Mono", monospace',
                fontSize: '14px',
                lineHeight: '1.5',
                boxSizing: 'border-box'
              }}
            />
          </div>

          {/* Execution Trigger Button */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
            <button
              onClick={handleExecuteScript}
              disabled={isExecutingCode || (executionMode === 'job' && !selectedAccountId)}
              style={{
                background: executionMode === 'instant' ? '#0284c7' : '#7c3aed',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                padding: '12px 28px',
                fontSize: '15px',
                fontWeight: '700',
                cursor: isExecutingCode ? 'not-allowed' : 'pointer',
                opacity: isExecutingCode || (executionMode === 'job' && !selectedAccountId) ? 0.7 : 1
              }}
            >
              {isExecutingCode
                ? 'Running Execution...'
                : executionMode === 'instant'
                  ? '⚡ Run Test Script Now'
                  : '🚀 Submit Persistent Colab Job'}
            </button>
          </div>

          {/* Output / Result Display */}
          {codeOutput && (
            <div style={{ marginTop: '20px', background: '#020617', border: `1px solid ${codeOutput.status === 'success' || codeOutput.status === 'queued' ? '#059669' : '#991b1b'}`, borderRadius: '8px', padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                <span style={{ fontWeight: '700', color: codeOutput.status === 'success' || codeOutput.status === 'queued' ? '#34d399' : '#fca5a5' }}>
                  Result: {codeOutput.status?.toUpperCase()} {codeOutput.execution_time ? `(${codeOutput.execution_time})` : ''}
                </span>
                {codeOutput.job_id && (
                  <a href={`/dashboard/jobs/${codeOutput.job_id}`} style={{ color: '#38bdf8', textDecoration: 'underline', fontSize: '13px', fontWeight: '600' }}>
                    View Job #{codeOutput.job_id.substring(0, 8)} in Dashboard →
                  </a>
                )}
              </div>

              {codeOutput.message && (
                <div style={{ color: '#a78bfa', fontSize: '14px', marginBottom: '10px', fontWeight: '600' }}>
                  {codeOutput.message}
                </div>
              )}

              {codeOutput.stdout && (
                <div>
                  <div style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}>Standard Output (stdout):</div>
                  <pre style={{ background: '#0f172a', padding: '12px', borderRadius: '6px', color: '#f1f5f9', fontSize: '13px', overflowX: 'auto', margin: '0 0 10px 0' }}>
                    {codeOutput.stdout}
                  </pre>
                </div>
              )}

              {codeOutput.stderr && (
                <div>
                  <div style={{ fontSize: '12px', color: '#fca5a5', marginBottom: '4px' }}>Standard Error / Log (stderr):</div>
                  <pre style={{ background: '#451a1a', padding: '12px', borderRadius: '6px', color: '#fca5a5', fontSize: '13px', overflowX: 'auto', margin: 0 }}>
                    {codeOutput.stderr}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
        )}
      </main>
    </div>
  );
}
