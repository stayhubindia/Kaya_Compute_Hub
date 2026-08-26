"""
Test Suite: Gate Decision Matrix and CLI Utility (Phase 3.5).
"""

import subprocess
import sys
from pathlib import Path
import pytest

from src.dataset.final_qa_auditor import FinalQAAuditor, GateStatus, LifecycleState


@pytest.fixture
def dataset_root():
    return Path("data/instruction_dataset/v2.0").resolve()


def test_15_gate_matrix_decisions(dataset_root):
    """Verifies that all 15 gates are present, evaluated, and critical gates pass."""
    auditor = FinalQAAuditor(dataset_dir=dataset_root)
    report = auditor.run_full_audit()

    assert len(report.gate_matrix) == 15
    critical_gates = [g for g in report.gate_matrix if g.is_critical]
    assert len(critical_gates) == 9

    for g in critical_gates:
        assert g.status == GateStatus.PASS, f"Critical gate {g.gate_id} ({g.gate_name}) did not pass!"

    assert report.all_critical_gates_passed is True
    assert report.lifecycle_state == LifecycleState.READY


def test_cli_audit_command(dataset_root):
    """Verifies that finalize_dataset.py CLI runs successfully with --audit and --report."""
    cmd = [
        sys.executable,
        "scripts/finalize_dataset.py",
        "--audit",
        "--report",
        "--dataset-dir", str(dataset_root),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "15-DIMENSION QUALITY GATE SCORECARD" in res.stdout
    assert "Dataset Lifecycle State: READY" in res.stdout


def test_cli_verify_command(dataset_root):
    """Verifies that finalize_dataset.py CLI runs successfully with --verify."""
    cmd = [
        sys.executable,
        "scripts/finalize_dataset.py",
        "--verify",
        "--dataset-dir", str(dataset_root),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "All dataset checksums cryptographically verified!" in res.stdout
