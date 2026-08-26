"""
Unit tests for AdapterValidator and config schema checks (Phase 5.1).
"""

import json
from pathlib import Path
import pytest

from src.release.adapter import AdapterValidator


def test_adapter_absence_reports_artifact_not_available(tmp_path: Path):
    validator = AdapterValidator()
    result = validator.validate_directory(tmp_path / "non_existent")
    assert result.is_valid is False
    assert result.status == "ARTIFACT_NOT_AVAILABLE"


def test_adapter_valid_structure(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "bias": "none",
        "base_model_name_or_path": "Qwen/Qwen3-4B-Base",
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(cfg, f)

    with open(adapter_dir / "adapter_model.safetensors", "wb") as f:
        f.write(b"dummy_weights_data")

    validator = AdapterValidator()
    result = validator.validate_directory(adapter_dir)
    assert result.is_valid is True
    assert result.status == "VALID_ARTIFACT"
    assert result.weights_format == "safetensors"


def test_adapter_invalid_rank_and_alpha(tmp_path: Path):
    adapter_dir = tmp_path / "adapter_bad"
    adapter_dir.mkdir()

    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,  # Expected 16
        "lora_alpha": 16,  # Expected 32
        "lora_dropout": 0.05,
        "target_modules": ["q_proj"],
        "bias": "none",
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(cfg, f)

    with open(adapter_dir / "adapter_model.safetensors", "wb") as f:
        f.write(b"dummy_weights")

    validator = AdapterValidator()
    result = validator.validate_directory(adapter_dir)
    assert result.is_valid is False
    assert result.status == "INVALID_FORMAT"
    assert any("LoRA rank 16, got 8" in e for e in result.errors)
    assert any("LoRA alpha 32, got 16" in e for e in result.errors)
