"""
Unit and integration tests for Ingestion CLI tool (Phase 3.3).
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest


def test_cli_help():
    cmd = [sys.executable, "scripts/ingest_documents.py", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--dry-run" in result.stdout


def test_cli_dry_run_and_execution(tmp_path):
    input_dir = tmp_path / "cli_input"
    input_dir.mkdir()
    output_dir = tmp_path / "cli_output"

    (input_dir / "sample.html").write_text(
        "<html><body><h1>Title</h1><p>Content paragraph here for testing.</p></body></html>",
        encoding="utf-8"
    )

    # 1. Dry run
    dry_cmd = [
        sys.executable,
        "scripts/ingest_documents.py",
        "--input", str(input_dir),
        "--source", "nptel",
        "--output-dir", str(output_dir),
        "--dry-run",
    ]
    res_dry = subprocess.run(dry_cmd, capture_output=True, text=True)
    assert res_dry.returncode == 0
    assert "[DRY-RUN]" in res_dry.stdout or "[DRY-RUN]" in res_dry.stderr
    assert not (output_dir / "manifest.json").exists()

    # 2. Real run
    run_cmd = [
        sys.executable,
        "scripts/ingest_documents.py",
        "--input", str(input_dir),
        "--source", "nptel",
        "--output-dir", str(output_dir),
        "--report",
    ]
    res_run = subprocess.run(run_cmd, capture_output=True, text=True)
    assert res_run.returncode == 0
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "reports" / "ingestion_report.md").is_file()
