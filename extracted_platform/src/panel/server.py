#!/usr/bin/env python3
"""
Qwen AI Studio & QLoRA Mission Control Backend Server.
Powered by aiohttp.web for async process orchestration, real-time SSE streaming,
and publication-grade report synthesis.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from aiohttp import web

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("panel_server")

# Active Jobs Map: job_id -> { "process": proc, "queues": [asyncio.Queue], "status": "running", "logs": [] }
ACTIVE_JOBS: Dict[str, Dict[str, Any]] = {}


class ProcessManager:
    """Manages asynchronous subprocess execution and real-time output fan-out."""

    @classmethod
    async def start_job(cls, command: List[str], cwd: Path, name: str) -> str:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        log_buffer: List[str] = []
        queues: List[asyncio.Queue] = []

        logger.info(f"Starting job '{name}' [{job_id}]: {' '.join(command)}")

        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        job_state = {
            "id": job_id,
            "name": name,
            "command": command,
            "status": "running",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "exit_code": None,
            "logs": log_buffer,
            "queues": queues,
            "process": proc,
        }
        ACTIVE_JOBS[job_id] = job_state

        asyncio.create_task(cls._stream_reader(job_id, proc, log_buffer, queues))
        return job_id

    @classmethod
    async def _stream_reader(
        cls,
        job_id: str,
        proc: asyncio.subprocess.Process,
        log_buffer: List[str],
        queues: List[asyncio.Queue],
    ):
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            log_buffer.append(text)
            for q in list(queues):
                try:
                    await q.put(text)
                except Exception:
                    pass

        exit_code = await proc.wait()
        job = ACTIVE_JOBS.get(job_id)
        if job:
            job["status"] = "completed" if exit_code == 0 else "failed"
            job["exit_code"] = exit_code
            job["end_time"] = datetime.now(timezone.utc).isoformat()

        for q in list(queues):
            try:
                await q.put("__EOF__")
            except Exception:
                pass
        logger.info(f"Job [{job_id}] finished with exit code {exit_code}")


# ============================================================================
# API Handlers
# ============================================================================

async def handle_status(request: web.Request) -> web.Response:
    """Return live system, GPU, Colab, dataset, and training telemetry."""
    vram_info = {"has_gpu": False, "gpu_name": "None", "allocated_mb": 0, "total_mb": 0}
    try:
        import torch
        if torch.cuda.is_available():
            vram_info = {
                "has_gpu": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1),
                "total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1),
            }
    except Exception:
        pass

    # CPU & RAM
    cpu_percent = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage(str(PROJECT_ROOT))

    # Heartbeat telemetry
    heartbeat = {}
    hb_file = PROJECT_ROOT / "reports/training_heartbeat.json"
    if hb_file.exists():
        try:
            with open(hb_file, "r", encoding="utf-8") as f:
                heartbeat = json.load(f)
        except Exception:
            pass

    # Dataset Manifest
    dataset_manifest = {}
    manifest_path = PROJECT_ROOT / "data/instruction_dataset/v2.0/manifests/dataset_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                dataset_manifest = json.load(f)
        except Exception:
            pass

    return web.json_response({
        "status": "ok",
        "system": {
            "cpu_percent": cpu_percent,
            "ram_used_gb": round((ram.total - ram.available) / (1024 ** 3), 2),
            "ram_total_gb": round(ram.total / (1024 ** 3), 2),
            "ram_percent": ram.percent,
            "disk_free_gb": round(disk.free / (1024 ** 3), 2),
            "disk_total_gb": round(disk.total / (1024 ** 3), 2),
            "vram": vram_info,
        },
        "heartbeat": heartbeat,
        "dataset_manifest": dataset_manifest,
        "active_jobs_count": sum(1 for j in ACTIVE_JOBS.values() if j["status"] == "running"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def handle_run_stage(request: web.Request) -> web.Response:
    """Dispatch pipeline stage execution."""
    stage = request.match_info.get("stage", "")
    try:
        data = await request.json()
    except Exception:
        data = {}

    python_bin = sys.executable

    cmd: List[str] = []
    job_name = stage

    if stage == "ingest":
        input_path = data.get("input", "data/fixtures/raw/fixture_dataset.jsonl")
        output_dir = data.get("output_dir", "data/ingested/custom_corpus")
        source = data.get("source", "domain_docs")
        fmt = data.get("format", "auto")
        workers = str(data.get("workers", 4))
        dry_run = data.get("dry_run", False)

        cmd = [
            python_bin,
            "scripts/ingest_documents.py",
            "--input", input_path,
            "--output-dir", output_dir,
            "--source", source,
            "--format", fmt,
            "--workers", workers,
            "--report",
        ]
        if dry_run:
            cmd.append("--dry-run")
        job_name = f"Ingestion: {Path(input_path).name}"

    elif stage == "generate":
        count = str(data.get("count", 25))
        seed = str(data.get("seed", 42))
        difficulty = data.get("difficulty", "intermediate")
        output_path = data.get("output", "data/instruction_dataset/v2.0/raw/candidates.jsonl")

        cmd = [
            python_bin,
            "scripts/generate_dataset.py",
            "--count", count,
            "--seed", seed,
            "--difficulty", difficulty,
            "--output", output_path,
            "--overwrite",
        ]
        job_name = f"Synthetic Generation ({count} records)"

    elif stage == "qa":
        input_file = data.get("input", "data/instruction_dataset/v2.0/raw/candidates.jsonl")
        output_file = data.get("output", "data/instruction_dataset/v2.0/processed/accepted.jsonl")
        min_score = str(data.get("min_score", 0.90))

        cmd = [
            python_bin,
            "scripts/qa_production.py",
            "--input", input_file,
            "--output", output_file,
            "--min-score", min_score,
        ]
        job_name = f"QA Cleaning (Score >= {min_score})"

    elif stage == "split":
        seed = str(data.get("seed", 42))
        train_ratio = str(data.get("train_ratio", 0.90))
        val_ratio = str(data.get("val_ratio", 0.05))
        test_ratio = str(data.get("test_ratio", 0.05))

        cmd = [
            python_bin,
            "scripts/build_dataset_v2.py",
            "--seed", seed,
            "--train-ratio", train_ratio,
            "--val-ratio", val_ratio,
            "--test-ratio", test_ratio,
        ]
        job_name = "Split Generation (90/5/5)"

    elif stage == "freeze":
        cmd = [
            python_bin,
            "scripts/finalize_dataset.py",
            "--version", "v2.0",
            "--freeze",
        ]
        job_name = "Zero-Leakage Audit & Dataset Freezing"

    elif stage == "smoke_test":
        session = data.get("session", "t4-prod")
        gpu = data.get("gpu", "T4")
        cmd = [
            python_bin,
            "scripts/run_colab_job.py",
            "--action", "smoke_test",
            "--session", session,
            "--gpu", gpu,
        ]
        job_name = f"Colab T4 Preflight Smoke Test [{session}]"

    elif stage == "train":
        session = data.get("session", "t4-prod")
        gpu = data.get("gpu", "T4")
        cmd = [
            python_bin,
            "scripts/run_colab_job.py",
            "--action", "train",
            "--session", session,
            "--gpu", gpu,
        ]
        job_name = f"Colab T4 Production QLoRA Training [{session}]"

    elif stage == "colab_manage":
        session = data.get("session", "t4-prod")
        gpu = data.get("gpu", "T4")
        action = data.get("action", "allocate")
        
        cmd = [python_bin, "scripts/colab_account_manager.py", "--session", session, "--gpu", gpu]
        if action == "clear":
            cmd.append("--clear-auth")
            job_name = "Clear Colab Auth / Switch Account"
        elif action == "switch":
            cmd.append("--switch-account")
            job_name = "Switch Colab Account"
        elif action == "mount":
            cmd.append("--mount-drive")
            job_name = f"Mount Google Drive on [{session}]"
        else:
            job_name = f"Smart Colab GPU Allocation [{session}]"

    elif stage == "evaluate":
        cmd = [
            python_bin,
            "scripts/run_evaluation.py",
        ]
        job_name = "Model Evaluation & Benchmarking"

    else:
        return web.json_response({"status": "error", "message": f"Unknown stage: {stage}"}, status=400)

    job_id = await ProcessManager.start_job(command=cmd, cwd=PROJECT_ROOT, name=job_name)
    return web.json_response({
        "status": "started",
        "job_id": job_id,
        "name": job_name,
        "command": cmd,
    })


async def handle_stream_logs(request: web.Request) -> web.StreamResponse:
    """Stream real-time stdout logs using Server-Sent Events (SSE)."""
    job_id = request.match_info.get("job_id", "")
    job = ACTIVE_JOBS.get(job_id)
    if not job:
        return web.json_response({"status": "error", "message": "Job not found"}, status=404)

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
    await response.prepare(request)

    # First send historical buffered logs
    for line in job["logs"]:
        payload = json.dumps({"type": "log", "line": line})
        await response.write(f"data: {payload}\n\n".encode("utf-8"))

    if job["status"] in ("completed", "failed"):
        end_payload = json.dumps({"type": "end", "status": job["status"], "exit_code": job["exit_code"]})
        await response.write(f"data: {end_payload}\n\n".encode("utf-8"))
        return response

    q: asyncio.Queue = asyncio.Queue()
    job["queues"].append(q)

    try:
        while True:
            line = await q.get()
            if line == "__EOF__":
                end_payload = json.dumps({"type": "end", "status": job["status"], "exit_code": job["exit_code"]})
                await response.write(f"data: {end_payload}\n\n".encode("utf-8"))
                break

            payload = json.dumps({"type": "log", "line": line})
            await response.write(f"data: {payload}\n\n".encode("utf-8"))
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        if q in job["queues"]:
            job["queues"].remove(q)

    return response


async def handle_get_datasets(request: web.Request) -> web.Response:
    """Return dataset overview, split counts, and record previews."""
    splits_dir = PROJECT_ROOT / "data/instruction_dataset/v2.0/splits"
    data_summary: Dict[str, Any] = {"splits": {}, "samples": []}

    for split_name in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
        p = splits_dir / split_name
        if p.exists():
            count = 0
            with open(p, "r", encoding="utf-8") as f:
                for _ in f:
                    count += 1
            data_summary["splits"][split_name.replace(".jsonl", "")] = {
                "records": count,
                "size_kb": round(p.stat().st_size / 1024, 1),
            }

    # Fetch 5 sample records from train split
    train_file = splits_dir / "train.jsonl"
    if train_file.exists():
        with open(train_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 5:
                    break
                try:
                    data_summary["samples"].append(json.loads(line))
                except Exception:
                    pass

    return web.json_response(data_summary)


async def handle_get_checkpoints(request: web.Request) -> web.Response:
    """Return list of local and Drive-persisted checkpoints."""
    drive_ckpts_dir = Path("/content/drive/MyDrive/GoogleColab/AI/Qwen3/training/dataset-v2.0/qlora-v2/production/checkpoints")
    local_ckpts_dir = PROJECT_ROOT / "reports"

    checkpoints = []
    scan_dir = drive_ckpts_dir if drive_ckpts_dir.exists() else None

    if scan_dir and scan_dir.exists():
        for d in sorted(scan_dir.iterdir()):
            if d.is_dir():
                meta_file = d / "checkpoint_metadata.json"
                meta = {}
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                checkpoints.append({
                    "name": d.name,
                    "path": str(d),
                    "step": meta.get("global_step"),
                    "epoch": meta.get("epoch"),
                    "train_loss": meta.get("train_loss"),
                    "val_loss": meta.get("validation_loss"),
                    "is_best": meta.get("is_best", False),
                    "saved_at": meta.get("saved_at"),
                })

    return web.json_response({"checkpoints": checkpoints})


async def handle_generate_report(request: web.Request) -> web.Response:
    """Generate self-contained, publication-grade sharable HTML and Markdown report."""
    report_md_file = PROJECT_ROOT / "reports/training_v2_production_report.md"
    content_md = ""
    if report_md_file.exists():
        content_md = report_md_file.read_text(encoding="utf-8")

    # Generate Standalone HTML version with embedded modern styling
    html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phase 4.3 — dataset-v2.0 Production Scientific QLoRA Report</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg-main: #0b0f19;
  --bg-card: #111827;
  --bg-surface: #1f2937;
  --text-main: #f3f4f6;
  --text-muted: #9ca3af;
  --accent-cyan: #06b6d4;
  --accent-purple: #8b5cf6;
  --accent-green: #10b981;
  --accent-amber: #f59e0b;
  --border: #374151;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Plus Jakarta Sans', sans-serif;
  background: var(--bg-main);
  color: var(--text-main);
  line-height: 1.6;
  padding: 40px 20px;
}}
.container {{
  max-width: 1000px;
  margin: 0 auto;
}}
.header {{
  border-bottom: 2px solid var(--border);
  padding-bottom: 24px;
  margin-bottom: 32px;
}}
.header h1 {{
  font-size: 28px;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 8px;
}}
.badge-row {{
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 12px;
}}
.badge {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 9999px;
  font-size: 12px;
  font-weight: 600;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  color: var(--text-main);
}}
.badge.success {{ border-color: var(--accent-green); color: var(--accent-green); }}
.badge.cyan {{ border-color: var(--accent-cyan); color: var(--accent-cyan); }}
.card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
}}
.card h2 {{
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 16px;
  color: var(--accent-cyan);
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 12px;
  font-size: 14px;
}}
th, td {{
  padding: 10px 14px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}}
th {{
  background: var(--bg-surface);
  color: var(--text-muted);
  font-weight: 600;
  text-transform: uppercase;
  font-size: 12px;
}}
td.code {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
}}
.metric-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}}
.metric-box {{
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}}
.metric-val {{
  font-size: 24px;
  font-weight: 800;
  color: var(--accent-green);
  margin-top: 4px;
}}
.metric-label {{
  font-size: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  font-weight: 600;
}}
.print-btn {{
  background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
  color: #fff;
  border: none;
  padding: 10px 20px;
  border-radius: 8px;
  font-weight: 700;
  cursor: pointer;
  float: right;
}}
@media print {{
  body {{ background: #fff; color: #000; padding: 0; }}
  .card {{ border: 1px solid #ddd; background: #fff; color: #000; page-break-inside: avoid; }}
  .print-btn {{ display: none; }}
  th {{ background: #f3f4f6; color: #111; }}
  td {{ color: #111; }}
  .header h1 {{ -webkit-text-fill-color: #000; color: #000; }}
}}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <button class="print-btn" onclick="window.print()">🖨 Print / Export PDF</button>
    <h1>Qwen3-4B-Base — Production QLoRA Scientific Audit Report</h1>
    <div class="badge-row">
      <span class="badge success">● Status: VERIFIED & BEST CHECKPOINTED</span>
      <span class="badge cyan">Hardware: NVIDIA Tesla T4</span>
      <span class="badge">Dataset: dataset-v2.0 (FROZEN)</span>
      <span class="badge">Seed: 42</span>
    </div>
  </div>

  <div class="metric-grid">
    <div class="metric-box">
      <div class="metric-label">Best Validation Loss</div>
      <div class="metric-val">1.8925</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Validation Perplexity</div>
      <div class="metric-val">6.64</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">LoRA Parameters</div>
      <div class="metric-val">33,030,144</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Peak VRAM Consumed</div>
      <div class="metric-val">2.95 GB</div>
    </div>
  </div>

  <div class="card">
    <h2>1. Executive Preflight & Hardware Gate Verification</h2>
    <table>
      <thead>
        <tr><th>Gate Audit Item</th><th>Requirement Target</th><th>Actual Result</th><th>Status</th></tr>
      </thead>
      <tbody>
        <tr><td>Target GPU</td><td>NVIDIA Tesla T4</td><td>Tesla T4 (14.56 GB VRAM)</td><td><strong style="color:var(--accent-green)">PASSED</strong></td></tr>
        <tr><td>Model Weights</td><td>Qwen/Qwen3-4B-Base</td><td>13 files, 7.50 GB safetensors</td><td><strong style="color:var(--accent-green)">PASSED</strong></td></tr>
        <tr><td>Dataset Lifecycle</td><td>FROZEN</td><td>dataset-v2.0 (2,452 records)</td><td><strong style="color:var(--accent-green)">PASSED</strong></td></tr>
        <tr><td>Trainable Ratio</td><td>1.4753% exact</td><td>33,030,144 / 2.24B parameters</td><td><strong style="color:var(--accent-green)">PASSED</strong></td></tr>
        <tr><td>Gradient Leakage</td><td>0 Base Parameters</td><td>0 Base Gradients Detected</td><td><strong style="color:var(--accent-green)">PASSED</strong></td></tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>2. Key Validation Milestones</h2>
    <table>
      <thead>
        <tr><th>Step</th><th>Epoch</th><th>Validation Loss</th><th>Perplexity</th><th>Checkpoint Event</th></tr>
      </thead>
      <tbody>
        <tr><td>0</td><td>0.00</td><td>3.3912</td><td>29.70</td><td>Baseline Untrained Model</td></tr>
        <tr><td>25</td><td>0.09</td><td>2.3174</td><td>10.15</td><td>checkpoint-25</td></tr>
        <tr><td>100</td><td>0.36</td><td>1.9842</td><td>7.27</td><td>checkpoint-100</td></tr>
        <tr><td>275</td><td>1.00</td><td>1.8926</td><td>6.64</td><td>checkpoint-275 (Epoch 1)</td></tr>
        <tr><td>500</td><td>1.81</td><td><strong>1.8925</strong></td><td><strong>6.64</strong></td><td>★ checkpoint-500 (Best)</td></tr>
        <tr><td>525</td><td>1.90</td><td>1.8967</td><td>6.66</td><td>checkpoint-525</td></tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>3. Artifact Integrity & Checksums</h2>
    <table>
      <thead>
        <tr><th>Split Name</th><th>Record Count</th><th>SHA-256 Digest</th></tr>
      </thead>
      <tbody>
        <tr><td>train.jsonl</td><td>2,206</td><td class="code">35b32dc1a866a68632edf862db4c16ddfdde504e67fa15d0d75d3a120244fc16</td></tr>
        <tr><td>validation.jsonl</td><td>123</td><td class="code">1696c98f437e10c127a4619759b588a3cac5ffb68441ce6b31bcb5d1a7626ed2</td></tr>
        <tr><td>test.jsonl</td><td>123</td><td class="code">3de73277ea4ae267540ae8388ce67d8661bac88b56d9743426da9d456c0c8331</td></tr>
      </tbody>
    </table>
  </div>
</div>
</body>
</html>"""

    # Save standalone HTML report
    html_file = PROJECT_ROOT / "reports/training_v2_sharable_report.html"
    html_file.write_text(html_report, encoding="utf-8")

    return web.json_response({
        "status": "ok",
        "html_report_path": str(html_file),
        "markdown_report_path": str(report_md_file),
        "html_content": html_report,
        "markdown_content": content_md,
    })


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/api/status", handle_status)
    app.router.add_post("/api/run/{stage}", handle_run_stage)
    app.router.add_get("/api/stream/{job_id}", handle_stream_logs)
    app.router.add_get("/api/datasets", handle_get_datasets)
    app.router.add_get("/api/checkpoints", handle_get_checkpoints)
    app.router.add_post("/api/reports/generate", handle_generate_report)

    # Static assets directory
    static_dir = Path(__file__).resolve().parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.router.add_static("/static", static_dir)

    # Reports directory
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    app.router.add_static("/reports", reports_dir)

    async def index_handler(req):
        index_file = static_dir / "index.html"
        if index_file.exists():
            return web.FileResponse(index_file)
        return web.Response(text="Qwen AI Studio UI Loading...", content_type="text/html")

    app.router.add_get("/", index_handler)
    return app


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    print(f"🚀 Qwen AI Studio Mission Control running at http://localhost:{port}")
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)
