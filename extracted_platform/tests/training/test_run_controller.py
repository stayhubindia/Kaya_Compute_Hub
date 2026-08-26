"""
Unit tests for Production Training Run Controller, Recovery & Checkpoint Orchestrator (Phase 5.2).
"""

import json
import shutil
from pathlib import Path
import pytest

from src.training.config import TrainingConfig
from src.training.run_controller import (
    CheckpointManifest,
    CheckpointValidator,
    FailureType,
    TrainingFailureRecord,
    TrainingHeartbeat,
    TrainingRunController,
    TrainingRunIdentity,
    TrainingRunManifest,
    TrainingRunState,
)
from src.training.sft_trainer import TrainingTelemetry


def test_run_identity_determinism():
    id1 = TrainingRunIdentity.create(
        qlora_version="qlora-v1",
        dataset_version="dataset-v1.0",
        seed=42,
        dataset_sha256="abc123sha",
        training_config_hash="def456hash",
    )
    id2 = TrainingRunIdentity.create(
        qlora_version="QLORA-V1",
        dataset_version="DATASET-V1.0",
        seed=42,
        dataset_sha256="abc123sha",
        training_config_hash="def456hash",
    )
    assert id1.run_id == "qlora-v1-dataset-v1.0-seed42"
    assert id2.run_id == "qlora-v1-dataset-v1.0-seed42"
    assert id1.dataset_version == "dataset-v1.0"
    assert id1.seed == 42


def test_state_transitions():
    manifest = TrainingRunManifest(
        run_identity=TrainingRunIdentity.create(),
        status=TrainingRunState.PLANNED,
    )
    assert manifest.status == TrainingRunState.PLANNED

    # Valid transitions: PLANNED -> PREFLIGHT -> SMOKE_TEST -> TRAINING -> FINALIZING -> FINALIZED
    manifest.transition_to(TrainingRunState.PREFLIGHT)
    assert manifest.status == TrainingRunState.PREFLIGHT

    manifest.transition_to(TrainingRunState.SMOKE_TEST)
    assert manifest.status == TrainingRunState.SMOKE_TEST

    manifest.transition_to(TrainingRunState.TRAINING)
    assert manifest.status == TrainingRunState.TRAINING

    manifest.transition_to(TrainingRunState.FINALIZING)
    assert manifest.status == TrainingRunState.FINALIZING

    manifest.transition_to(TrainingRunState.FINALIZED)
    assert manifest.status == TrainingRunState.FINALIZED

    # Illegal transition from terminal state
    with pytest.raises(ValueError, match="Illegal state transition"):
        manifest.transition_to(TrainingRunState.TRAINING)


def test_checkpoint_manifest_creation_and_validation(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoint-50"
    ckpt_dir.mkdir()
    (ckpt_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"dummy_adapter_weights")
    (ckpt_dir / "optimizer.pt").write_bytes(b"dummy_optimizer_state")

    manifest = CheckpointValidator.generate_checkpoint_manifest(
        checkpoint_dir=ckpt_dir,
        step=50,
        epoch=1.5,
        loss=1.2345,
        validation_loss=1.1234,
        dataset_version="dataset-v1.0",
        dataset_sha256="test_dataset_sha",
        training_config_hash="test_config_hash",
        run_id="qlora-v1-dataset-v1.0-seed42",
    )
    assert (ckpt_dir / "checkpoint_manifest.json").exists()
    assert manifest.global_step == 50
    assert len(manifest.artifact_paths) == 3

    # Audit validation
    val_res = CheckpointValidator.validate_checkpoint(
        checkpoint_dir=ckpt_dir,
        expected_dataset_version="dataset-v1.0",
        expected_dataset_sha="test_dataset_sha",
        expected_config_hash="test_config_hash",
        expected_run_id="qlora-v1-dataset-v1.0-seed42",
    )
    assert val_res.is_valid is True
    assert len(val_res.errors) == 0


def test_checkpoint_corruption_detection(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoint-100"
    ckpt_dir.mkdir()
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"original_weights")

    CheckpointValidator.generate_checkpoint_manifest(
        checkpoint_dir=ckpt_dir,
        step=100,
        epoch=2.0,
        loss=0.98,
        validation_loss=0.88,
        dataset_version="dataset-v1.0",
        dataset_sha256="correct_sha",
        training_config_hash="correct_hash",
        run_id="qlora-v1-dataset-v1.0-seed42",
    )

    # Tamper with weights
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"corrupted_tampered_weights")

    val_res = CheckpointValidator.validate_checkpoint(
        checkpoint_dir=ckpt_dir,
        expected_dataset_version="dataset-v1.0",
        expected_dataset_sha="correct_sha",
        expected_config_hash="correct_hash",
        expected_run_id="qlora-v1-dataset-v1.0-seed42",
    )
    assert val_res.is_valid is False
    assert any("Corrupted artifact 'adapter_model.safetensors'" in e for e in val_res.errors)


def test_find_latest_valid_checkpoint_with_corrupted_latest(tmp_path: Path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    # Create older valid checkpoint (step 50)
    ckpt50 = checkpoints_dir / "checkpoint-50"
    ckpt50.mkdir()
    (ckpt50 / "weights.bin").write_bytes(b"weights_50")
    CheckpointValidator.generate_checkpoint_manifest(
        checkpoint_dir=ckpt50,
        step=50,
        epoch=1.0,
        loss=1.5,
        validation_loss=1.4,
        dataset_version="dataset-v1.0",
        dataset_sha256="test_sha",
        training_config_hash="test_hash",
        run_id="qlora-v1-dataset-v1.0-seed42",
    )

    # Create newer corrupted checkpoint (step 100)
    ckpt100 = checkpoints_dir / "checkpoint-100"
    ckpt100.mkdir()
    (ckpt100 / "weights.bin").write_bytes(b"weights_100")
    CheckpointValidator.generate_checkpoint_manifest(
        checkpoint_dir=ckpt100,
        step=100,
        epoch=2.0,
        loss=1.1,
        validation_loss=1.0,
        dataset_version="dataset-v1.0",
        dataset_sha256="test_sha",
        training_config_hash="test_hash",
        run_id="qlora-v1-dataset-v1.0-seed42",
    )
    # Tamper with step 100
    (ckpt100 / "weights.bin").write_bytes(b"corrupted")

    controller = TrainingRunController()
    controller.dataset_sha = "test_sha"
    controller.config_hash = "test_hash"
    controller.run_identity = TrainingRunIdentity.create(
        dataset_version="dataset-v1.0",
        dataset_sha256="test_sha",
        training_config_hash="test_hash",
    )

    latest_valid, corrupted = controller.find_latest_valid_checkpoint(checkpoints_dir=checkpoints_dir)
    assert latest_valid == ckpt50
    assert ckpt100 in corrupted


def test_hardware_gate_enforcement():
    # Test on CPU: hardware gate blocks execution
    controller = TrainingRunController(
        expected_gpu_target="Tesla T4",
        allow_gpu_target_mismatch=False,
    )
    hw_ok, hw_msg = controller.check_hardware_gate()
    assert hw_ok is False
    assert "HARDWARE_GATE_BLOCKED" in hw_msg or "GPU_TARGET_MISMATCH" in hw_msg


def test_durable_storage_check(tmp_path: Path):
    controller = TrainingRunController()
    writable_dir = tmp_path / "valid_storage"
    ok, msg = controller.check_durable_storage(writable_dir)
    assert ok is True
    assert "writable" in msg


def test_disk_space_monitoring(tmp_path: Path):
    controller = TrainingRunController(minimum_free_space_gb=1.0)
    ok, free_gb, msg = controller.check_disk_space(tmp_path)
    assert ok is True
    assert free_gb > 0


def test_training_heartbeat(tmp_path: Path):
    heartbeat_path = tmp_path / "reports" / "training_heartbeat.json"
    heartbeat = TrainingHeartbeat(
        run_id="qlora-v1-dataset-v1.0-seed42",
        state=TrainingRunState.TRAINING,
        epoch=1.5,
        step=50,
        last_checkpoint="checkpoint-50",
        vram_allocated_mb=4500.0,
        vram_reserved_mb=5000.0,
        elapsed_seconds=120.5,
    )
    heartbeat.save(heartbeat_path)
    assert heartbeat_path.exists()

    with open(heartbeat_path, "r") as f:
        data = json.load(f)
    assert data["step"] == 50
    assert data["state"] == "TRAINING"


def test_preflight_execution():
    controller = TrainingRunController()
    report = controller.execute_preflight()
    assert report.dataset_version == "dataset-v1.0"
    assert report.manifest_status == "FROZEN"


def test_dry_run_execution():
    controller = TrainingRunController()
    dry_res = controller.execute_dry_run()
    assert dry_res["dry_run"] is True
    assert dry_res["status"] == "VALIDATED"
    assert dry_res["run_identity"]["run_id"] == "qlora-v1-dataset-v1.0-seed42"


def test_training_finalization(tmp_path: Path):
    controller = TrainingRunController()
    controller.output_dir = tmp_path / "output"
    controller.output_dir.mkdir(parents=True, exist_ok=True)
    controller.reports_dir = tmp_path / "reports"
    controller.reports_dir.mkdir(parents=True, exist_ok=True)

    controller.manifest.transition_to(TrainingRunState.PREFLIGHT)
    controller.manifest.transition_to(TrainingRunState.SMOKE_TEST)
    controller.manifest.transition_to(TrainingRunState.TRAINING)

    telemetry = TrainingTelemetry(
        training_status="COMPLETED",
        total_steps=100,
        total_epochs=3.0,
        best_validation_loss=0.85,
        best_checkpoint_path=str(tmp_path / "checkpoints" / "checkpoint-100"),
        eval_history=[{"step": 100, "epoch": 3.0, "val_loss": 0.85}],
    )

    manifest_path = controller.finalize_training_run(
        best_checkpoint_path=tmp_path / "checkpoints" / "checkpoint-100",
        telemetry=telemetry,
    )
    assert manifest_path.exists()
    assert controller.manifest.status == TrainingRunState.FINALIZED
    with open(manifest_path, "r") as f:
        comp_data = json.load(f)
    assert comp_data["status"] == "COMPLETED"
    assert comp_data["best_validation_loss"] == 0.85
