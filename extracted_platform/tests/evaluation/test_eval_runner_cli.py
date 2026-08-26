"""
Unit tests for Evaluation Runner & CLI Utilities (Phase 4.4).
"""

from pathlib import Path
from src.evaluation.config import EvaluationConfig
from src.evaluation.runner import EvaluationRunner


def test_runner_preflight_audit():
    runner = EvaluationRunner()
    result = runner.run_preflight_audit()
    assert result.is_passed is True
    assert result.checks["evaluation_config_valid"] is True
    assert result.checks["dataset_frozen_lifecycle"] is True
    assert result.checks["split_files_present"] is True
    assert result.checks["sha256_checksums_match"] is True
    assert result.checks["split_isolation_verified"] is True
    assert result.checks["test_examples_loaded"] is True


def test_runner_dry_run_base_and_adapter(tmp_path: Path):
    cfg = EvaluationConfig(reports_dir=str(tmp_path), output_dir=str(tmp_path))
    runner = EvaluationRunner(cfg)

    # Base evaluation (dry-run)
    base_rep = runner.evaluate(model_type="base", dry_run=True)
    assert base_rep.sample_count == 7
    assert base_rep.is_mock is True
    assert base_rep.model_type == "base"

    # Adapter evaluation (dry-run)
    adapt_rep = runner.evaluate(model_type="adapter", dry_run=True)
    assert adapt_rep.sample_count == 7
    assert adapt_rep.is_mock is True
    assert adapt_rep.model_type == "adapter"

    # Compare
    regression = runner.compare(base_rep, adapt_rep)
    assert regression.baseline_model.startswith("Qwen/Qwen3-4B-Base")
    assert (tmp_path / "regression_report.json").exists()
    assert (tmp_path / "regression_report.md").exists()


def test_runner_blocked_gpu_behavior(tmp_path: Path):
    cfg = EvaluationConfig(reports_dir=str(tmp_path), output_dir=str(tmp_path))
    runner = EvaluationRunner(cfg)

    # When CUDA is unavailable and dry_run=False, inference is blocked
    if not runner.hardware.cuda_available:
        rep = runner.evaluate(model_type="base", dry_run=False)
        assert rep.sample_count == 0

        manifest_path = tmp_path / "evaluation_manifest.json"
        assert manifest_path.exists()
        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            assert manifest_data["status"] == "BLOCKED"
            assert "MODEL INFERENCE BLOCKED" in manifest_data["details"]["reason"]
