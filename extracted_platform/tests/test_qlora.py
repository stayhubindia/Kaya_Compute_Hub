"""
Tests for QLoRA 4-bit Quantization & LoRA Adapter Setup (Phase 4.1).
"""

import pytest
import torch

from src.training.config import TrainingConfig
from src.training.qlora import QLoRAConfigurator


def test_qlora_bnb_config():
    cfg = TrainingConfig()
    configurator = QLoRAConfigurator(cfg)
    bnb_cfg = configurator.get_bnb_config()

    assert bnb_cfg is not None
    assert bnb_cfg.load_in_4bit is True
    assert bnb_cfg.bnb_4bit_quant_type == "nf4"
    assert bnb_cfg.bnb_4bit_use_double_quant is True


def test_qlora_peft_config():
    cfg = TrainingConfig()
    configurator = QLoRAConfigurator(cfg)
    peft_cfg = configurator.get_peft_config()

    assert peft_cfg.r == 16
    assert peft_cfg.lora_alpha == 32
    assert peft_cfg.lora_dropout == 0.05
    assert "q_proj" in peft_cfg.target_modules
    assert "down_proj" in peft_cfg.target_modules


def test_estimate_qwen_parameters():
    report = QLoRAConfigurator.estimate_qwen_qlora_parameters(
        num_layers=36,
        hidden_size=2048,
        intermediate_size=11008,
        r=16,
    )
    assert report.total_parameters > 2_000_000_000
    assert report.trainable_parameters == 29_933_568
    assert 0.01 <= report.trainable_percentage <= 5.0
    assert report.lora_rank == 16
