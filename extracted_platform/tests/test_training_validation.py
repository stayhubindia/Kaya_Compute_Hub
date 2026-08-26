"""
Tests for Training Preflight Validator (Phase 4.1).
"""

import pytest
from pathlib import Path

from src.training.config import TrainingConfig
from src.training.validation import GateStatus, TrainingPreflightValidator


def test_preflight_validator_execution(tmp_path):
    cfg = TrainingConfig()
    validator = TrainingPreflightValidator(cfg)
    report = validator.run_preflight()

    assert report is not None
    assert len(report.gates) >= 8
    assert report.dataset_version == "dataset-v1.0"
    assert report.manifest_status == "FROZEN"
    assert report.record_counts["total"] == 59

    # Test Markdown rendering
    md = report.to_markdown()
    assert "# Training Preflight Audit Report" in md
    assert "Gate ID" in md
    assert "manifest_status" in md

    # Test JSON saving
    json_path = tmp_path / "preflight_test.json"
    report.save_json(json_path)
    assert json_path.exists()
