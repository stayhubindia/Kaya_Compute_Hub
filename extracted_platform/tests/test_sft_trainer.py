"""
Tests for Production QLoRA Supervised Fine-Tuning Engine (Phase 4.2).
Validates dataset locking, ChatML assistant-only masking assertions,
QLoRA model preparation, optimizer/scheduler setup, smoke tests,
checkpoint saving/reloading, dataset version mismatch rejection, and telemetry reporting.
"""

import json
from pathlib import Path
import pytest
import torch

from src.dataset.schema import DatasetRecord, Message, RecordMetadata, Role
from src.training.checkpoint import TrainingCheckpointManager
from src.training.collator import DataCollatorForAssistantOnlyLoss
from src.training.config import TrainingConfig
from src.training.sft_trainer import ProductionSFTTrainer, SmokeTestResult, TrainingTelemetry
from src.training.tokenizer import MockQwenTokenizer, TrainingTokenizerWrapper


@pytest.fixture
def test_config():
    cfg = TrainingConfig()
    cfg.model.path = "models/non_existent_mock_path"
    cfg.model.fallback_pretrained_id = None
    cfg.training.num_train_epochs = 1
    cfg.training.per_device_train_batch_size = 1
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.logging_steps = 1
    cfg.training.save_steps = 2
    cfg.training.eval_steps = 2
    return cfg


def test_trainer_initialization_and_token_report(test_config, tmp_path):
    test_config.training.output_dir = str(tmp_path / "output")
    test_config.training.local_fallback_output_dir = str(tmp_path / "fallback")
    trainer = ProductionSFTTrainer(test_config)
    trainer.reports_dir = tmp_path / "reports"

    token_report = trainer.initialize_and_audit()
    assert token_report.record_count > 0
    assert token_report.total_tokens > 0
    assert (tmp_path / "reports" / "tokenization_report.json").exists()
    assert (tmp_path / "reports" / "tokenization_report.md").exists()


def test_assistant_only_loss_batch_assertion(test_config):
    tokenizer = MockQwenTokenizer()
    collator = DataCollatorForAssistantOnlyLoss(tokenizer=tokenizer, assistant_only_loss=True)

    records = [
        DatasetRecord(
            messages=[
                Message(role=Role.SYSTEM, content="System prompt instructions."),
                Message(role=Role.USER, content="User request question."),
                Message(role=Role.ASSISTANT, content="Assistant valid answer here."),
            ],
            metadata=RecordMetadata(
                domain="coding",
                topic="python",
                task_type="code_generation",
                difficulty="intermediate",
                quality_score=0.95,
            ),
        )
    ]

    # Valid assertion should pass without error
    collator.assert_assistant_only_masking(records)


def test_assistant_only_loss_assertion_failure_on_unmasked():
    tokenizer = MockQwenTokenizer()
    # Collator with assistant_only_loss=False leaves all tokens unmasked
    collator_unmasked = DataCollatorForAssistantOnlyLoss(tokenizer=tokenizer, assistant_only_loss=False)

    records = [
        DatasetRecord(
            messages=[
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hi there"),
            ],
            metadata=RecordMetadata(
                domain="general_knowledge",
                topic="greeting",
                task_type="explanation",
                difficulty="beginner",
                quality_score=0.9,
            ),
        )
    ]

    with pytest.raises(ValueError, match="Assistant-only masking assertion failed"):
        collator_unmasked.assert_assistant_only_masking(records)


def test_prepare_model_and_optimizer(test_config, tmp_path):
    test_config.training.output_dir = str(tmp_path / "output")
    trainer = ProductionSFTTrainer(test_config)
    trainer.reports_dir = tmp_path / "reports"
    trainer.initialize_and_audit()

    model, optimizer, scheduler = trainer.prepare_model_and_optimizer()
    assert model is not None
    assert optimizer is not None
    assert scheduler is not None


def test_smoke_test_pipeline(test_config, tmp_path):
    test_config.training.output_dir = str(tmp_path / "smoke_out")
    test_config.training.local_fallback_output_dir = str(tmp_path / "smoke_out")
    trainer = ProductionSFTTrainer(test_config)
    trainer.reports_dir = tmp_path / "reports"

    smoke_result = trainer.run_smoke_test()
    assert smoke_result.success is True
    assert smoke_result.loss_finite is True
    assert smoke_result.gradients_finite is True
    assert smoke_result.optimizer_step_successful is True
    assert smoke_result.checkpoint_written is True
    assert smoke_result.checkpoint_reloaded is True


def test_checkpoint_save_and_resume_locking(test_config, tmp_path):
    out_dir = tmp_path / "checkpoints_test"
    ckpt_mgr = TrainingCheckpointManager(
        output_dir=out_dir,
        config=test_config.checkpointing,
        dataset_version="dataset-v1.0",
        dataset_sha256="abc123sha",
    )

    ckpt_path = ckpt_mgr.save_checkpoint_metadata(
        step=10,
        epoch=1.0,
        loss=2.45,
        learning_rate=2e-4,
    )
    assert ckpt_path.exists()

    # Valid resume
    meta = ckpt_mgr.validate_resume_checkpoint(ckpt_path)
    assert meta.global_step == 10
    assert meta.dataset_version == "dataset-v1.0"

    # Mismatched dataset version rejection
    mismatched_mgr = TrainingCheckpointManager(
        output_dir=out_dir,
        config=test_config.checkpointing,
        dataset_version="dataset-v2.0",
        dataset_sha256="different_sha",
    )
    with pytest.raises(ValueError, match="Checkpoint version mismatch"):
        mismatched_mgr.validate_resume_checkpoint(ckpt_path)


def test_training_execution_and_telemetry(test_config, tmp_path):
    test_config.training.output_dir = str(tmp_path / "train_out")
    test_config.training.local_fallback_output_dir = str(tmp_path / "train_out")
    trainer = ProductionSFTTrainer(test_config)
    trainer.reports_dir = tmp_path / "reports"

    telemetry = trainer.train(max_steps=2, override_epochs=1)
    assert telemetry.training_status == "COMPLETED"
    assert telemetry.total_steps == 2
    assert telemetry.total_tokens_processed > 0
    assert (tmp_path / "reports" / "training_report.json").exists()
    assert (tmp_path / "reports" / "training_report.md").exists()

    # Verify markdown report format
    md_content = telemetry.to_markdown()
    assert "Production QLoRA SFT Training Telemetry Report" in md_content
    assert "COMPLETED" in md_content
