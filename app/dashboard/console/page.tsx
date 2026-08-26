'use client';

import React, { useState, useEffect, useRef } from 'react';
import DashboardNavbar from '@/components/DashboardNavbar';
import { User, authClient } from '@/lib/api/authClient';
import { api } from '@/lib/api/client';

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
  }
};

export default function ConsolePage() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Terminal State
  const [commandInput, setCommandInput] = useState('');
  const [terminalLogs, setTerminalLogs] = useState<Array<{ timestamp: string; type: 'cmd' | 'stdout' | 'stderr' | 'system'; text: string }>>([
    { timestamp: new Date().toLocaleTimeString(), type: 'system', text: 'Kaya Compute Colab Interactive Terminal v1.0 connected.\nType bash commands or use quick preset actions below.' }
  ]);
  const [isExecutingCmd, setIsExecutingCmd] = useState(false);

  // Code Runner State
  const [selectedTemplate, setSelectedTemplate] = useState('arxiv_test');
  const [scriptName, setScriptName] = useState('ArXiv Test Script');
  const [codeContent, setCodeContent] = useState(TEMPLATE_SCRIPTS.arxiv_test.code);
  const [executionMode, setExecutionMode] = useState<'instant' | 'job'>('instant');
  const [targetDir, setTargetDir] = useState('/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv');
  const [isExecutingCode, setIsExecutingCode] = useState(false);
  const [codeOutput, setCodeOutput] = useState<{ status?: string; stdout?: string; stderr?: string; execution_time?: string; job_id?: string; message?: string } | null>(null);

  const terminalEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    async function loadUser() {
      try {
        const u = await authClient.getCurrentUser();
        setUser(u);
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

  const handleRunCommand = async (cmdToRun?: string) => {
    const cmd = (cmdToRun || commandInput).trim();
    if (!cmd) return;

    if (cmd === 'clear') {
      setTerminalLogs([]);
      setCommandInput('');
      return;
    }

    const timeStr = new Date().toLocaleTimeString();
    setTerminalLogs(prev => [...prev, { timestamp: timeStr, type: 'cmd', text: `$ ${cmd}` }]);
    if (!cmdToRun) setCommandInput('');
    setIsExecutingCmd(true);

    try {
      const data: any = await api.post('/console/terminal/', { command: cmd });
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

        {/* Section 1: Colab Interactive Web Terminal */}
        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '20px', marginBottom: '24px', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.5)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '16px', fontWeight: '700', color: '#f1f5f9' }}>
                🖥️ Colab Terminal Console
              </span>
              <span style={{ fontSize: '12px', color: '#64748b' }}>(CWD: /content/drive/MyDrive/.../Arxiv)</span>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button onClick={() => handleRunCommand('nvidia-smi')} style={{ background: '#1e293b', color: '#38bdf8', border: '1px solid #334155', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', cursor: 'pointer' }}>
                GPU Status
              </button>
              <button onClick={() => handleRunCommand('ls -la "/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv"')} style={{ background: '#1e293b', color: '#34d399', border: '1px solid #334155', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', cursor: 'pointer' }}>
                Check Drive
              </button>
              <button onClick={() => handleRunCommand('python3 --version && df -h /content/drive')} style={{ background: '#1e293b', color: '#fbbf24', border: '1px solid #334155', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', cursor: 'pointer' }}>
                System Info
              </button>
              <button onClick={() => handleRunCommand('clear')} style={{ background: '#334155', color: '#cbd5e1', border: 'none', borderRadius: '6px', padding: '4px 10px', fontSize: '12px', cursor: 'pointer' }}>
                Clear Log
              </button>
            </div>
          </div>

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
            <span style={{ background: '#1e293b', color: '#38bdf8', padding: '10px 14px', borderRadius: '6px', fontFamily: 'monospace', fontSize: '14px', display: 'flex', alignItems: 'center' }}>
              durgesh@kaya:~$
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

        {/* Section 2: Code Execution Sandbox & Script Runner */}
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
              </select>
            </div>
          </div>

          {/* Form Options: Mode & Target Directory */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px', marginBottom: '16px' }}>
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
              disabled={isExecutingCode}
              style={{
                background: executionMode === 'instant' ? '#0284c7' : '#7c3aed',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                padding: '12px 28px',
                fontSize: '15px',
                fontWeight: '700',
                cursor: isExecutingCode ? 'not-allowed' : 'pointer',
                opacity: isExecutingCode ? 0.7 : 1
              }}
            >
              {isExecutingCode
                ? 'Running Execution...'
                : executionMode === 'instant'
                  ? '⚡ Run Test Script Now'
                  : '🚀 Submit Background Job'}
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
      </main>
    </div>
  );
}
