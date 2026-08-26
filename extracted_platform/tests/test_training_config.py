"""
Tests for Training Configuration Schema & Validation (Phase 4.1).
"""

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from src.training.config import (
    CheckpointConfig,
    DatasetConfig,
    EvaluationConfig,
    LoraConfig,
    ModelConfig,
    QuantizationConfig,
    TokenizerConfig,
    TrainingConfig,
    TrainingHyperparameters,
)


def test_default_training_config():
    config = TrainingConfig()
    assert config.model.name == "Qwen/Qwen3-4B-Base"
    assert config.dataset.version == "dataset-v1.0"
    assert config.quantization.load_in_4bit is True
    assert config.quantization.quant_type == "nf4"
    assert config.lora.r == 16
    assert config.lora.lora_alpha == 32
    assert config.training.num_train_epochs == 3
    assert config.training.per_device_train_batch_size == 1
    assert config.training.gradient_accumulation_steps == 8
    assert config.training.effective_batch_size == 8
    assert config.training.assistant_only_loss is True


def test_load_save_training_yaml(tmp_path):
    config = TrainingConfig()
    yaml_path = tmp_path / "test_training.yaml"
    config.save_to_yaml(yaml_path)
    assert yaml_path.exists()

    loaded = TrainingConfig.load_from_yaml(yaml_path)
    assert loaded.model.name == config.model.name
    assert loaded.lora.r == config.lora.r
    assert loaded.training.learning_rate == config.training.learning_rate


def test_config_validation_rules():
    config = TrainingConfig()
    errors = config.validate_rules()
    assert len(errors) == 0

    # Invalid learning rate caught by pydantic
    with pytest.raises(ValidationError):
        TrainingHyperparameters(learning_rate=-0.01)


def test_split_ratios_validation():
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        DatasetConfig(split_ratios={"train": 0.8, "validation": 0.1, "test": 0.05})


def test_quant_type_validation():
    with pytest.raises(ValueError, match="quant_type must be 'nf4' or 'fp4'"):
        QuantizationConfig(quant_type="invalid_type")
