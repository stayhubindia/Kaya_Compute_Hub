"""
Comprehensive Test Suite for Phase 4.1 — dataset-v2.0 Training Specification & Preflight.

Validates:
1. Frozen dataset state verification
2. Split SHA-256 checksum verification against manifest
3. Cross-split record isolation and zero hash collisions
4. Tokenization and 0.00% truncation rate with max_seq_length=2048
5. Assistant-only loss masking with -100 on system/user prompt tokens
6. QLoRA 4-bit NF4 configuration & target module resolution
7. Exact LoRA parameter accounting (29,933,568 trainable params, 0.9700%)
8. Hardware envelope & peak VRAM budget (< 16 GB Tesla T4 limit)
9. Configuration SHA-256 hash generation and immutability
10. Version-locking and state integrity
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch

from src.training.config import TrainingConfig, DatasetConfig
from src.training.dataset import TrainingDatasetLoader, DatasetIntegrityError
from src.training.qlora import QLoRAConfigurator
from src.training.tokenizer import TrainingTokenizerWrapper
from src.training.collator import DataCollatorForAssistantOnlyLoss
from src.training.validation import TrainingPreflightValidator, GateStatus


@pytest.fixture
def config_v2():
    config_path = Path("configs/training_v2.yaml")
    assert config_path.exists(), "configs/training_v2.yaml must exist"
    return TrainingConfig.load_from_yaml(config_path)


def test_frozen_dataset_verification(config_v2):
    """Test 1: Enforce that dataset-v2.0 is strictly verified in FROZEN state."""
    loader = TrainingDatasetLoader(config_v2.dataset)
    manifest = loader.load_manifest()

    status = getattr(manifest, "lifecycle_state", None) or getattr(manifest, "status", None)
    if hasattr(status, "value"):
        status = status.value
    assert str(status).upper() == "FROZEN"
    assert manifest.dataset_version == "dataset-v2.0"


def test_split_sha256_checksums(config_v2):
    """Test 2: Validate split file checksums match cryptographic manifest signatures."""
    loader = TrainingDatasetLoader(config_v2.dataset)
    loader.load_manifest()

    expected_hashes = {
        "train.jsonl": "35b32dc1a866a68632edf862db4c16ddfdde504e67fa15d0d75d3a120244fc16",
        "validation.jsonl": "1696c98f437e10c127a4619759b588a3cac5ffb68441ce6b31bcb5d1a7626ed2",
        "test.jsonl": "3de73277ea4ae267540ae8388ce67d8661bac88b56d9743426da9d456c0c8331",
    }

    for fname, exp_hash in expected_hashes.items():
        file_path = Path(f"data/instruction_dataset/v2.0/splits/{fname}")
        actual_hash = loader.verify_file_checksum(file_path, expected_sha256=exp_hash)
        assert actual_hash.lower() == exp_hash.lower()


def test_cross_split_isolation(config_v2):
    """Test 3: Enforce zero cross-split record leakage and zero hash collision."""
    loader = TrainingDatasetLoader(config_v2.dataset)
    train_ds, val_ds, test_ds = loader.load_splits()

    assert len(train_ds) == 2206
    assert len(val_ds) == 123
    assert len(test_ds) == 123
    assert len(train_ds) + len(val_ds) + len(test_ds) == 2452

    # Audit isolation
    loader.audit_split_isolation(train_ds.records, val_ds.records, test_ds.records)


def test_tokenization_and_zero_truncation(config_v2):
    """Test 4: Validate Native ChatML tokenization and 0.00% truncation rate at max_seq_length=2048."""
    tok_wrapper = TrainingTokenizerWrapper(config_v2.tokenizer)
    tok = tok_wrapper.load()

    assert tok is not None
    assert tok.vocab_size >= 151643

    loader = TrainingDatasetLoader(config_v2.dataset)
    train_ds, val_ds, test_ds = loader.load_splits()
    all_records = list(train_ds.records) + list(val_ds.records) + list(test_ds.records)

    report = tok_wrapper.analyze_token_lengths(all_records, max_seq_length=2048)

    assert report.record_count == 2452
    assert report.total_tokens == 831694
    assert report.truncated_count == 0
    assert report.truncation_rate == 0.0
    assert report.max <= 2048
    assert report.counts_le_2048 == 2452


def test_assistant_only_loss_masking(config_v2):
    """Test 5: Enforce deterministic assistant-only loss masking with -100 on prompt tokens."""
    tok_wrapper = TrainingTokenizerWrapper(config_v2.tokenizer)
    tok = tok_wrapper.load()

    loader = TrainingDatasetLoader(config_v2.dataset)
    train_ds, _, _ = loader.load_splits()

    collator = DataCollatorForAssistantOnlyLoss(
        tokenizer=tok,
        max_seq_length=config_v2.tokenizer.max_seq_length,
        assistant_only_loss=True,
    )

    sample_records = train_ds.records[:10]
    collator.assert_assistant_only_masking(sample_records)

    # Validate batch collation tensor structures
    batch = collator(sample_records)
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    assert batch["input_ids"].shape == batch["labels"].shape

    # Validate that prompt tokens contain -100 and assistant tokens contain non -100
    for i in range(len(sample_records)):
        labels = batch["labels"][i]
        assert -100 in labels, "Prompt tokens must be masked with -100"
        active_tokens = labels[labels != -100]
        assert len(active_tokens) > 0, "Assistant response tokens must have active targets"


def test_qlora_configuration_parameters(config_v2):
    """Test 6: Verify 4-bit NF4 double quantization and PEFT LoRA target modules."""
    configurator = QLoRAConfigurator(config_v2)
    bnb_cfg = configurator.get_bnb_config()
    peft_cfg = configurator.get_peft_config()

    assert bnb_cfg.load_in_4bit is True
    assert bnb_cfg.bnb_4bit_quant_type == "nf4"
    assert bnb_cfg.bnb_4bit_use_double_quant is True

    assert peft_cfg.r == 16
    assert peft_cfg.lora_alpha == 32
    assert peft_cfg.lora_dropout == 0.05
    assert set(peft_cfg.target_modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }


def test_exact_lora_parameter_accounting():
    """Test 7: Validate analytical and exact parameter accounting for Qwen3-4B-Base."""
    param_report = QLoRAConfigurator.estimate_qwen_qlora_parameters(
        num_layers=36,
        hidden_size=2048,
        intermediate_size=11008,
        num_heads=16,
        num_kv_heads=2,
        r=16,
        lora_alpha=32,
    )

    # 36 layers * 831,488 params/layer = 29,933,568 params
    assert param_report.trainable_parameters == 29_933_568
    assert param_report.frozen_parameters == 3_085_846_528
    assert round(param_report.trainable_percentage, 4) == 0.9700
    assert len(param_report.target_modules) == 7


def test_vram_budget_feasibility(config_v2):
    """Test 8: Validate that the peak memory envelope safely fits within 16GB Tesla T4."""
    validator = TrainingPreflightValidator(config_v2)
    report = validator.run_preflight()

    vram_gate = next(g for g in report.gates if g.gate_id == "estimated_vram")
    assert vram_gate.status == GateStatus.PASS
    assert report.estimated_vram_gb <= 8.0  # ~5.34 GB estimated peak
    assert report.estimated_vram_gb < 16.0


def test_config_sha256_immutability():
    """Test 9: Verify configuration file hashing and reproducibility."""
    cfg_file = Path("configs/training_v2.yaml")
    assert cfg_file.exists()
    content = cfg_file.read_bytes()
    h = hashlib.sha256(content).hexdigest()
    assert len(h) == 64

    # Verify manifest hash matches
    manifest_p = Path("reports/training_v2_config_manifest.json")
    if manifest_p.exists():
        m_data = json.loads(manifest_p.read_text())
        assert m_data["run_identity"]["config_sha256"] == h


def test_version_locking_rejection(config_v2):
    """Test 10: Verify that dataset loader rejects mismatched versions or non-frozen states."""
    bad_cfg = config_v2.dataset.model_copy(update={"version": "dataset-v99.0"})
    loader = TrainingDatasetLoader(bad_cfg)
    with pytest.raises(DatasetIntegrityError):
        loader.load_manifest()
