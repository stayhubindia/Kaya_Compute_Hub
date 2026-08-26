"""
Unit tests for Phase 5.4 ProductionReleaseGate and Release Validation Subsystem.
"""

import json
from pathlib import Path
import pytest

from src.release.production_gate import (
    ProductionReleaseGate,
    ProductionGateStatus,
    ProductionReleaseAuditReport,
)


def test_production_gate_full_success():
    """Verify that current release package passes all 9 release gates."""
    gate = ProductionReleaseGate(
        release_dir="releases/qwen3-4b-qlora-v1.0",
        dataset_manifest="datasets/production/manifests/production_manifest.json",
        benchmark_manifest="benchmarks/benchmark-v1.0/manifest.json",
        reports_dir="reports",
    )
    report = gate.execute_full_release_gate()

    assert report.is_approved() is True
    assert report.gate_status == ProductionGateStatus.APPROVED
    assert report.decision == "RELEASE APPROVED"
    assert len(report.errors) == 0
    assert len(report.gates) == 9

    # Check each individual gate
    for gate_name, res in report.gates.items():
        assert res.passed is True, f"Gate {gate_name} failed: {res.errors}"


def test_production_gate_tampered_file_detection(tmp_path: Path):
    """Verify that file tampering triggers RELEASE BLOCKED."""
    # Copy release files to tmp_path
    rel_dir = tmp_path / "qwen3-4b-qlora-v1.0"
    rel_dir.mkdir(parents=True)
    adapter_dir = rel_dir / "adapter"
    adapter_dir.mkdir()

    # Create dummy files
    (rel_dir / "manifest.json").write_text(json.dumps({"status": "RELEASED"}))
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }))
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"dummy_weights")
    (rel_dir / "checksums.sha256").write_text("0000000000000000000000000000000000000000000000000000000000000000  manifest.json\n")

    gate = ProductionReleaseGate(release_dir=rel_dir)
    integrity_res = gate.audit_adapter_integrity()
    assert integrity_res.passed is False
    assert len(integrity_res.errors) > 0


def test_production_gate_regression_threshold():
    """Verify regression safety boundary checking."""
    gate = ProductionReleaseGate()
    eval_gate, reg_gate = gate.audit_evaluation_and_regression_safety()

    assert eval_gate.passed is True
    assert reg_gate.passed is True
    assert reg_gate.details["regression_rate"] == 8 / 500
    assert reg_gate.details["regression_rate"] < 0.05


def test_production_gate_performance_safety():
    """Verify VRAM headroom and performance audit."""
    gate = ProductionReleaseGate()
    perf_gate = gate.audit_performance_safety()

    assert perf_gate.passed is True
    assert perf_gate.details["peak_adapter_vram_gb"] == 6.48
    assert perf_gate.details["headroom_gb"] == 8.08
    assert perf_gate.details["oom_events"] == 0
