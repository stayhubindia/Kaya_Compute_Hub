"""
Unit tests for BaseModelCompatibilityValidator (Phase 5.1).
"""

from pathlib import Path
import pytest

from src.release.compatibility import BaseModelCompatibilityValidator


def test_base_model_compatibility_valid():
    validator = BaseModelCompatibilityValidator()
    cfg = {
        "_name_or_path": "Qwen/Qwen3-4B-Base",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "vocab_size": 151936,
    }
    res = validator.validate_base_model_metadata(cfg)
    assert res.is_compatible is True
    assert res.architecture_matched is True


def test_base_model_compatibility_rejects_incompatible():
    validator = BaseModelCompatibilityValidator()
    cfg = {
        "_name_or_path": "Llama-3-8B",
        "architectures": ["LlamaForCausalLM"],
    }
    res = validator.validate_base_model_metadata(cfg, base_model_id="Llama-3-8B")
    assert res.is_compatible is False
    assert len(res.errors) > 0
