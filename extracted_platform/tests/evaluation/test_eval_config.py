"""
Unit tests for Evaluation Configuration (Phase 4.4).
"""

import pytest
from pathlib import Path
from src.evaluation.config import (
    EvaluationConfig,
    EvaluationDatasetConfig,
    EvaluationModelConfig,
    GenerationConfig,
    MetricsConfig,
    RegressionConfig,
)


def test_default_config_validity():
    cfg = EvaluationConfig()
    assert cfg.model.name == "Qwen/Qwen3-4B-Base"
    assert cfg.model.model_type == "base"
    assert cfg.dataset.version == "dataset-v1.0"
    assert cfg.dataset.lifecycle == "FROZEN"
    assert cfg.generation.max_new_tokens == 512
    assert cfg.metrics.compute_deterministic is True
    assert cfg.regression.tolerance_pct == 5.0
    assert cfg.seed == 42


def test_invalid_model_type():
    with pytest.raises(Exception):
        EvaluationModelConfig(model_type="unsupported")  # type: ignore


def test_invalid_lifecycle():
    with pytest.raises(ValueError, match="Evaluation requires FROZEN lifecycle"):
        EvaluationDatasetConfig(lifecycle="DRAFT")


def test_invalid_split():
    with pytest.raises(ValueError, match="Invalid evaluation split"):
        EvaluationDatasetConfig(split="invalid_split")


def test_yaml_load_and_save(tmp_path: Path):
    cfg = EvaluationConfig()
    yaml_path = tmp_path / "test_eval.yaml"
    cfg.to_yaml(yaml_path)
    assert yaml_path.exists()

    loaded = EvaluationConfig.from_yaml(yaml_path)
    assert loaded.model.name == cfg.model.name
    assert loaded.dataset.version == cfg.dataset.version
    assert loaded.compute_hash() == cfg.compute_hash()


def test_config_hash_determinism():
    cfg1 = EvaluationConfig(seed=42)
    cfg2 = EvaluationConfig(seed=42)
    assert cfg1.compute_hash() == cfg2.compute_hash()

    cfg3 = EvaluationConfig(seed=99)
    assert cfg1.compute_hash() != cfg3.compute_hash()
