// Qwen AI Studio Mission Control Frontend Logic

let currentSSE = null;
let lossPoints = [
  { step: 0, val: 3.3912 },
  { step: 25, val: 2.3174 },
  { step: 50, val: 2.1197 },
  { step: 75, val: 2.0334 },
  { step: 100, val: 1.9842 },
  { step: 125, val: 1.9507 },
  { step: 150, val: 1.9373 },
  { step: 175, val: 1.9356 },
  { step: 200, val: 1.9206 },
  { step: 225, val: 1.9168 },
  { step: 250, val: 1.9046 },
  { step: 275, val: 1.8926 },
  { step: 300, val: 1.9170 },
  { step: 325, val: 1.9164 },
  { step: 350, val: 1.9099 },
  { step: 375, val: 1.9064 },
  { step: 400, val: 1.9039 },
  { step: 425, val: 1.9086 },
  { step: 450, val: 1.9085 },
  { step: 475, val: 1.9053 },
  { step: 500, val: 1.8925 },
  { step: 525, val: 1.8967 }
];

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initChart();
  refreshStatus();
  setInterval(refreshStatus, 4000);
});

// Tab Navigation
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      switchTab(targetId);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.toggle('active', t.getAttribute('data-tab') === tabId);
  });
  document.querySelectorAll('.tab-pane').forEach(p => {
    p.classList.toggle('active', p.id === tabId);
  });
}

// Telemetry & Hardware Status
async function refreshStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.status === 'ok') {
      const sys = data.system;
      document.getElementById('val-cpu').textContent = `${sys.cpu_percent}%`;
      document.getElementById('val-ram').textContent = `${sys.ram_used_gb} / ${sys.ram_total_gb} GB`;

      if (sys.vram.has_gpu) {
        document.getElementById('val-gpu-name').textContent = sys.vram.gpu_name;
        const vramGb = (sys.vram.allocated_mb / 1024).toFixed(2);
        const totalGb = (sys.vram.total_mb / 1024).toFixed(2);
        document.getElementById('val-vram').textContent = `${vramGb} / ${totalGb} GB`;
        const pct = Math.round((sys.vram.allocated_mb / sys.vram.total_mb) * 100);
        document.getElementById('vram-fill').style.width = `${pct}%`;
      }

      // Heartbeat
      const hb = data.heartbeat;
      if (hb && hb.step) {
        document.getElementById('stat-step').textContent = `${hb.step} / 828`;
        document.getElementById('stat-epoch').textContent = `${hb.epoch} / 3.0`;
        if (hb.best_validation_loss) {
          document.getElementById('stat-best-loss').textContent = `${hb.best_validation_loss}`;
        }
        document.getElementById('badge-heartbeat-state').textContent = `${hb.state || 'RUNNING'}`;
      }

      // Manifest stats
      const manifest = data.dataset_manifest;
      if (manifest && manifest.splits) {
        document.getElementById('stat-train-count').textContent = `${manifest.splits.train?.record_count || 2206} records`;
        document.getElementById('stat-val-count').textContent = `${manifest.splits.validation?.record_count || 123} records`;
        document.getElementById('stat-test-count').textContent = `${manifest.splits.test?.record_count || 123} records`;
      }
    }
  } catch (err) {
    console.error('Status fetch error:', err);
  }
}

// Stage Dispatchers
async function dispatchJob(stage, payload, title) {
  appendLog(`\n[Job Dispatch] Starting stage: ${title}...`, 'system');
  document.getElementById('active-job-title').textContent = title;
  document.getElementById('val-job-status').textContent = 'Running';
  document.getElementById('job-dot').className = 'pill-dot amber';

  // Expand console
  document.getElementById('console-drawer').classList.remove('minimized');

  try {
    const res = await fetch(`/api/run/${stage}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (result.status === 'started') {
      appendLog(`[Job ID: ${result.job_id}] Command: ${result.command.join(' ')}`, 'system');
      streamJobLogs(result.job_id);
    } else {
      appendLog(`[Error] ${result.message || 'Failed to start stage'}`, 'error');
    }
  } catch (err) {
    appendLog(`[Network Error] ${err.message}`, 'error');
  }
}

// Server-Sent Events (SSE) Stream Reader
function streamJobLogs(jobId) {
  if (currentSSE) {
    currentSSE.close();
  }

  currentSSE = new EventSource(`/api/stream/${jobId}`);

  currentSSE.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'log') {
        appendLog(data.line);
        parseTrainingLogLine(data.line);
      } else if (data.type === 'end') {
        currentSSE.close();
        currentSSE = null;
        const isSuccess = data.status === 'completed';
        appendLog(`[Job Finished] Status: ${data.status} (Exit Code: ${data.exit_code})`, isSuccess ? 'success' : 'error');
        document.getElementById('val-job-status').textContent = isSuccess ? 'Completed' : 'Failed';
        document.getElementById('job-dot').className = isSuccess ? 'pill-dot green' : 'pill-dot red';
        refreshStatus();
      }
    } catch (e) {
      appendLog(event.data);
    }
  };

  currentSSE.onerror = () => {
    if (currentSSE) {
      currentSSE.close();
      currentSSE = null;
    }
  };
}

function parseTrainingLogLine(line) {
  // Check if line contains Validation Loss: e.g. "Step 525 Validation Loss: 1.8967"
  const valMatch = line.match(/Step\s+(\d+)\s+Validation Loss:\s+([0-9.]+)/i);
  if (valMatch) {
    const step = parseInt(valMatch[1]);
    const loss = parseFloat(valMatch[2]);
    lossPoints.push({ step, val: loss });
    initChart();
  }
}

// Console Helpers
function appendLog(line, className = '') {
  const body = document.getElementById('console-body');
  const div = document.createElement('div');
  div.className = `console-line ${className}`;
  div.textContent = line;
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
}

function clearConsole() {
  document.getElementById('console-body').innerHTML = '';
}

function toggleConsole() {
  const drawer = document.getElementById('console-drawer');
  const btn = document.getElementById('btn-toggle-console');
  drawer.classList.toggle('minimized');
  btn.textContent = drawer.classList.contains('minimized') ? '▲' : '▼';
}

// Form Handlers
function handleIngest(e) {
  e.preventDefault();
  const payload = {
    input: document.getElementById('ingest-input').value,
    output_dir: document.getElementById('ingest-output-dir').value,
    source: document.getElementById('ingest-source').value,
    format: document.getElementById('ingest-format').value,
    workers: parseInt(document.getElementById('ingest-workers').value),
    dry_run: document.getElementById('ingest-dry-run').checked
  };
  dispatchJob('ingest', payload, 'Document Ingestion');
}

function handleGenerate(e) {
  e.preventDefault();
  const payload = {
    count: parseInt(document.getElementById('gen-count').value),
    seed: parseInt(document.getElementById('gen-seed').value),
    difficulty: document.getElementById('gen-difficulty').value,
    output: document.getElementById('gen-output').value
  };
  dispatchJob('generate', payload, 'Synthetic Generation');
}

function handleQA(e) {
  e.preventDefault();
  const payload = {
    input: document.getElementById('qa-input').value,
    output: document.getElementById('qa-output').value,
    min_score: parseFloat(document.getElementById('qa-score').value)
  };
  dispatchJob('qa', payload, 'QA & Cleaning');
}

function handleSplit(e) {
  e.preventDefault();
  const payload = {
    train_ratio: parseFloat(document.getElementById('split-train').value),
    val_ratio: parseFloat(document.getElementById('split-val').value),
    test_ratio: parseFloat(document.getElementById('split-test').value),
    seed: parseInt(document.getElementById('split-seed').value)
  };
  dispatchJob('split', payload, 'Split Generation');
}

function handleFreeze() {
  dispatchJob('freeze', {}, 'Zero-Leakage Audit & Dataset Freezing');
}

function handleSmokeTest() {
  const session = document.getElementById('smoke-session').value;
  dispatchJob('smoke_test', { session }, `Colab T4 Smoke Test [${session}]`);
}

function handleTrain() {
  const session = document.getElementById('smoke-session')?.value || 't4-prod';
  dispatchJob('train', { session }, `Production QLoRA Training [${session}]`);
}

function handleColabManage(action = 'allocate') {
  const session = document.getElementById('smoke-session')?.value || 't4-prod';
  const actionLabels = {
    allocate: 'Smart Multi-Account GPU Allocation',
    switch: 'Switch Colab Account',
    clear: 'Clear Colab Auth Credentials'
  };
  const title = actionLabels[action] || 'Colab Multi-Account Manager';
  dispatchJob('colab_manage', { session, action, gpu: 'T4' }, `${title} [${session}]`);
}

// Chart Rendering (Canvas)
function initChart() {
  const canvas = document.getElementById('lossChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid lines
  ctx.strokeStyle = 'rgba(75, 85, 99, 0.25)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = 20 + i * ((h - 40) / 3);
    ctx.beginPath();
    ctx.moveTo(30, y);
    ctx.lineTo(w - 20, y);
    ctx.stroke();
  }

  if (lossPoints.length < 2) return;

  const minLoss = 1.5;
  const maxLoss = 3.5;

  const getX = (step) => 40 + (step / 828) * (w - 70);
  const getY = (val) => (h - 30) - ((val - minLoss) / (maxLoss - minLoss)) * (h - 60);

  // Gradient area under curve
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(6, 182, 212, 0.35)');
  grad.addColorStop(1, 'rgba(6, 182, 212, 0.0)');

  ctx.beginPath();
  ctx.moveTo(getX(lossPoints[0].step), getY(lossPoints[0].val));
  for (let i = 1; i < lossPoints.length; i++) {
    ctx.lineTo(getX(lossPoints[i].step), getY(lossPoints[i].val));
  }
  ctx.lineTo(getX(lossPoints[lossPoints.length - 1].step), h - 30);
  ctx.lineTo(getX(lossPoints[0].step), h - 30);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Draw line
  ctx.beginPath();
  ctx.moveTo(getX(lossPoints[0].step), getY(lossPoints[0].val));
  for (let i = 1; i < lossPoints.length; i++) {
    ctx.lineTo(getX(lossPoints[i].step), getY(lossPoints[i].val));
  }
  ctx.strokeStyle = '#06b6d4';
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Draw points
  lossPoints.forEach(p => {
    ctx.beginPath();
    ctx.arc(getX(p.step), getY(p.val), 3.5, 0, Math.PI * 2);
    ctx.fillStyle = p.val <= 1.895 ? '#10b981' : '#06b6d4';
    ctx.fill();
  });
}

// Sharable Report Generator
async function generateAndExportReport() {
  const container = document.getElementById('report-view-container');
  container.innerHTML = '<div class="loading-spinner">Synthesizing comprehensive publication report...</div>';

  try {
    const res = await fetch('/api/reports/generate', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'ok') {
      container.innerHTML = `
        <div style="margin-bottom:16px; display:flex; justify-content:space-between; align-items:center;">
          <strong style="color:var(--accent-green)">✓ Sharable HTML Report Saved: reports/training_v2_sharable_report.html</strong>
          <a href="/reports/training_v2_sharable_report.html" target="_blank" class="btn btn-secondary" style="font-size:12px;">🔗 Open Fullscreen Report</a>
        </div>
        <iframe srcdoc="${escapeHtml(data.html_content)}" style="width:100%; height:500px; border:1px solid var(--border-subtle); border-radius:8px; background:#fff;"></iframe>
      `;
    }
  } catch (err) {
    container.innerHTML = `<div class="console-line error">Failed to generate report: ${err.message}</div>`;
  }
}

function printReport() {
  window.print();
}

function escapeHtml(text) {
  return text.replace(/"/g, '&quot;');
}
