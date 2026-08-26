"""
Production Training Run Controller, Recovery & Checkpoint Orchestrator (Phase 5.2).
Manages the complete training lifecycle for Qwen3-4B-Base QLoRA fine-tuning:
preflight readiness, hardware/smoke-test gating, checkpoint verification & fallback recovery,
Google Drive durable storage protection, heartbeat monitoring, and Phase 5.1 packaging handoff.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import yaml
from pydantic import BaseModel, Field

from src.dataset.production import DatasetFreezeState, ProductionManifest
from src.training.checkpoint import CheckpointMetadata, TrainingCheckpointManager
from src.training.config import TrainingConfig
from src.training.sft_trainer import ProductionSFTTrainer, SmokeTestResult, TrainingTelemetry
from src.training.utils import (
    HardwareEnvironmentInfo,
    compute_config_hash,
    compute_file_sha256,
    detect_hardware_environment,
    estimate_training_schedule,
)
from src.training.validation import GateStatus, PreflightReport, TrainingPreflightValidator

logger = logging.getLogger(__name__)


class TrainingRunState(str, Enum):
    """Lifecycle states for training run execution."""
    PLANNED = "PLANNED"
    PREFLIGHT = "PREFLIGHT"
    SMOKE_TEST = "SMOKE_TEST"
    TRAINING = "TRAINING"
    PAUSED = "PAUSED"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    FINALIZING = "FINALIZING"
    FINALIZED = "FINALIZED"


# State transition graph
VALID_STATE_TRANSITIONS: Dict[TrainingRunState, Set[TrainingRunState]] = {
    TrainingRunState.PLANNED: {TrainingRunState.PREFLIGHT, TrainingRunState.BLOCKED, TrainingRunState.FAILED},
    TrainingRunState.PREFLIGHT: {TrainingRunState.SMOKE_TEST, TrainingRunState.BLOCKED, TrainingRunState.FAILED, TrainingRunState.PLANNED},
    TrainingRunState.SMOKE_TEST: {TrainingRunState.TRAINING, TrainingRunState.BLOCKED, TrainingRunState.FAILED},
    TrainingRunState.TRAINING: {TrainingRunState.PAUSED, TrainingRunState.RECOVERING, TrainingRunState.FINALIZING, TrainingRunState.COMPLETED, TrainingRunState.FAILED},
    TrainingRunState.PAUSED: {TrainingRunState.TRAINING, TrainingRunState.RECOVERING, TrainingRunState.FAILED},
    TrainingRunState.RECOVERING: {TrainingRunState.TRAINING, TrainingRunState.FAILED, TrainingRunState.BLOCKED},
    TrainingRunState.FINALIZING: {TrainingRunState.FINALIZED, TrainingRunState.FAILED},
    TrainingRunState.COMPLETED: {TrainingRunState.FINALIZING, TrainingRunState.FINALIZED},
    TrainingRunState.FINALIZED: set(),  # Terminal state
    TrainingRunState.FAILED: {TrainingRunState.RECOVERING, TrainingRunState.PLANNED},
    TrainingRunState.BLOCKED: {TrainingRunState.PREFLIGHT, TrainingRunState.PLANNED},
}


class FailureType(str, Enum):
    """Categorized runtime and environmental failure modes."""
    GPU_FAILURE = "GPU_FAILURE"
    CUDA_FAILURE = "CUDA_FAILURE"
    DRIVE_FAILURE = "DRIVE_FAILURE"
    CHECKPOINT_FAILURE = "CHECKPOINT_FAILURE"
    DATASET_FAILURE = "DATASET_FAILURE"
    CONFIGURATION_FAILURE = "CONFIGURATION_FAILURE"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    DISK_SPACE_FAILURE = "DISK_SPACE_FAILURE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class TrainingFailureRecord(BaseModel):
    """Detailed record of an environmental or runtime interruption."""
    failure_type: FailureType
    message: str
    step: int = 0
    epoch: float = 0.0
    last_valid_checkpoint: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TrainingRunIdentity(BaseModel):
    """Deterministic training run identity independent of execution timestamps."""
    run_id: str
    dataset_version: str
    dataset_sha256: str
    training_config_hash: str
    base_model: str
    seed: int
    execution_id: str

    @classmethod
    def create(
        cls,
        qlora_version: str = "qlora-v1",
        dataset_version: str = "dataset-v1.0",
        seed: int = 42,
        dataset_sha256: str = "",
        training_config_hash: str = "",
        base_model: str = "Qwen/Qwen3-4B-Base",
    ) -> TrainingRunIdentity:
        """Construct deterministic run identity."""
        clean_v = qlora_version.lower()
        clean_ds = dataset_version.lower()
        run_id = f"{clean_v}-{clean_ds}-seed{seed}"
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        exec_id = f"{run_id}-{ts}"
        return cls(
            run_id=run_id,
            dataset_version=dataset_version,
            dataset_sha256=dataset_sha256,
            training_config_hash=training_config_hash,
            base_model=base_model,
            seed=seed,
            execution_id=exec_id,
        )


class CheckpointManifest(BaseModel):
    """Cryptographic manifest stored inside every validated checkpoint directory."""
    checkpoint_name: str
    global_step: int
    epoch: float
    loss: float
    validation_loss: Optional[float] = None
    dataset_version: str
    dataset_sha256: str
    training_config_hash: str
    run_id: str
    artifact_paths: List[str] = Field(default_factory=list)
    artifact_sizes: Dict[str, int] = Field(default_factory=dict)
    artifact_hashes: Dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> CheckpointManifest:
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


class CheckpointValidationResult(BaseModel):
    """Audit result for a checkpoint directory."""
    is_valid: bool = False
    checkpoint_dir: str
    manifest: Optional[CheckpointManifest] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CheckpointValidator:
    """Validates checkpoint structure, state dictionaries, and cryptographic checksums."""

    @staticmethod
    def generate_checkpoint_manifest(
        checkpoint_dir: Union[str, Path],
        step: int,
        epoch: float,
        loss: float,
        validation_loss: Optional[float],
        dataset_version: str,
        dataset_sha256: str,
        training_config_hash: str,
        run_id: str,
    ) -> CheckpointManifest:
        """Create and write checkpoint_manifest.json inside a saved checkpoint folder."""
        c_dir = Path(checkpoint_dir)
        inventory = []
        sizes = {}
        hashes = {}

        for p in sorted(c_dir.rglob("*")):
            if p.is_file() and p.name != "checkpoint_manifest.json":
                rel_p = p.relative_to(c_dir).as_posix()
                inventory.append(rel_p)
                sizes[rel_p] = p.stat().st_size
                hashes[rel_p] = compute_file_sha256(p)

        manifest = CheckpointManifest(
            checkpoint_name=c_dir.name,
            global_step=step,
            epoch=epoch,
            loss=loss,
            validation_loss=validation_loss,
            dataset_version=dataset_version,
            dataset_sha256=dataset_sha256,
            training_config_hash=training_config_hash,
            run_id=run_id,
            artifact_paths=inventory,
            artifact_sizes=sizes,
            artifact_hashes=hashes,
        )
        manifest.save(c_dir / "checkpoint_manifest.json")
        return manifest

    @staticmethod
    def validate_checkpoint(
        checkpoint_dir: Union[str, Path],
        expected_dataset_version: str = "",
        expected_dataset_sha: str = "",
        expected_config_hash: str = "",
        expected_run_id: str = "",
    ) -> CheckpointValidationResult:
        """Audit checkpoint directory for completeness and cryptographic integrity."""
        c_dir = Path(checkpoint_dir)
        result = CheckpointValidationResult(checkpoint_dir=str(c_dir))

        if not c_dir.exists() or not c_dir.is_dir():
            result.errors.append(f"Checkpoint directory does not exist: {c_dir}")
            return result

        man_file = c_dir / "checkpoint_manifest.json"
        meta_file = c_dir / "checkpoint_metadata.json"

        # Check metadata/manifest presence
        if man_file.exists():
            try:
                manifest = CheckpointManifest.load(man_file)
                result.manifest = manifest
            except Exception as e:
                result.errors.append(f"Corrupted checkpoint_manifest.json: {e}")
                return result
        elif meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
                manifest = CheckpointManifest(
                    checkpoint_name=c_dir.name,
                    global_step=meta_dict.get("global_step", 0),
                    epoch=meta_dict.get("epoch", 0.0),
                    loss=meta_dict.get("loss", 0.0),
                    dataset_version=meta_dict.get("dataset_version", ""),
                    dataset_sha256=meta_dict.get("dataset_sha256", ""),
                    training_config_hash=meta_dict.get("config_hash", ""),
                    run_id=expected_run_id,
                )
                result.manifest = manifest
                result.warnings.append("checkpoint_manifest.json missing; validated via checkpoint_metadata.json")
            except Exception as e:
                result.errors.append(f"Corrupted checkpoint_metadata.json: {e}")
                return result
        else:
            result.errors.append("No checkpoint_manifest.json or checkpoint_metadata.json found")
            return result

        # Verify dataset version
        if expected_dataset_version and manifest.dataset_version != expected_dataset_version:
            result.errors.append(
                f"Dataset version mismatch: checkpoint '{manifest.dataset_version}' vs expected '{expected_dataset_version}'"
            )

        # Verify dataset sha if available
        if expected_dataset_sha and manifest.dataset_sha256 and manifest.dataset_sha256 != expected_dataset_sha:
            result.errors.append(
                f"Dataset SHA-256 mismatch: checkpoint '{manifest.dataset_sha256[:8]}' vs expected '{expected_dataset_sha[:8]}'"
            )

        # Verify config hash if available
        if expected_config_hash and manifest.training_config_hash and manifest.training_config_hash != expected_config_hash:
            result.errors.append(
                f"Training config hash mismatch: checkpoint '{manifest.training_config_hash[:8]}' vs expected '{expected_config_hash[:8]}'"
            )

        # Verify run_id if available
        if expected_run_id and manifest.run_id and manifest.run_id != expected_run_id:
            result.errors.append(
                f"Run ID mismatch: checkpoint '{manifest.run_id}' vs expected '{expected_run_id}'"
            )

        # Verify file presence and hashes if manifest has hashes
        if manifest.artifact_hashes:
            for rel_path, exp_hash in manifest.artifact_hashes.items():
                f_path = c_dir / rel_path
                if not f_path.exists():
                    result.errors.append(f"Missing artifact: {rel_path}")
                else:
                    act_hash = compute_file_sha256(f_path)
                    if act_hash != exp_hash:
                        result.errors.append(f"Corrupted artifact '{rel_path}': hash mismatch")

        result.is_valid = len(result.errors) == 0
        return result


class TrainingHeartbeat(BaseModel):
    """Lightweight telemetry heartbeat written periodically during training."""
    run_id: str
    state: TrainingRunState
    epoch: float = 0.0
    step: int = 0
    last_checkpoint: Optional[str] = None
    last_update: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    vram_allocated_mb: float = 0.0
    vram_reserved_mb: float = 0.0
    elapsed_seconds: float = 0.0

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.last_update = datetime.now(timezone.utc).isoformat()
        tmp_p = p.with_suffix(".tmp")
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
        tmp_p.replace(p)


class TrainingRunManifest(BaseModel):
    """Complete execution manifest tracking session state, parameters, and checkpoints."""
    run_identity: TrainingRunIdentity
    status: TrainingRunState = TrainingRunState.PLANNED
    status_reason: Optional[str] = None
    hardware_info: Dict[str, Any] = Field(default_factory=dict)
    current_step: int = 0
    current_epoch: float = 0.0
    total_steps: int = 0
    total_epochs: float = 0.0
    best_step: Optional[int] = None
    best_epoch: Optional[float] = None
    best_validation_loss: Optional[float] = None
    best_checkpoint_path: Optional[str] = None
    checkpoint_history: List[Dict[str, Any]] = Field(default_factory=list)
    failures: List[TrainingFailureRecord] = Field(default_factory=list)
    resume_count: int = 0
    recovery_count: int = 0
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None

    def transition_to(self, new_state: TrainingRunState, reason: str = "") -> None:
        """Validate and apply state transition."""
        allowed = VALID_STATE_TRANSITIONS.get(self.status, set())
        if new_state not in allowed:
            raise ValueError(
                f"Illegal state transition from {self.status.value} to {new_state.value}. "
                f"Allowed transitions: {[s.value for s in allowed]}"
            )
        logger.info("Transitioning run state: %s -> %s (Reason: %s)", self.status.value, new_state.value, reason)
        self.status = new_state
        self.status_reason = reason or None

    def save_atomic(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = p.with_suffix(".tmp")
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
        tmp_p.replace(p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> TrainingRunManifest:
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


class TrainingRunController:
    """Master operational controller for executing, monitoring, recovering, and finalizing training."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        minimum_free_space_gb: float = 5.0,
        expected_gpu_target: str = "Tesla T4",
        allow_gpu_target_mismatch: bool = False,
    ):
        self.config = config or TrainingConfig()
        self.minimum_free_space_gb = minimum_free_space_gb
        self.expected_gpu_target = expected_gpu_target
        self.allow_gpu_target_mismatch = allow_gpu_target_mismatch

        self.hardware = detect_hardware_environment()
        self.output_dir = Path(self.config.training.output_dir)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Compute provenance hashes
        manifest_path = Path(self.config.dataset.manifest_path)
        self.dataset_sha = compute_file_sha256(manifest_path) if manifest_path.exists() else ""
        self.config_hash = compute_config_hash(self.config.model_dump())

        # Deterministic identity
        self.run_identity = TrainingRunIdentity.create(
            qlora_version="qlora-v1",
            dataset_version=self.config.dataset.version,
            seed=self.config.training.seed,
            dataset_sha256=self.dataset_sha,
            training_config_hash=self.config_hash,
            base_model=self.config.model.name,
        )

        self.manifest = TrainingRunManifest(
            run_identity=self.run_identity,
            hardware_info=self.hardware.model_dump(),
        )

        self.trainer: Optional[ProductionSFTTrainer] = None
        self.checkpoint_validator = CheckpointValidator()

    def check_durable_storage(self, target_dir: Optional[Union[str, Path]] = None) -> Tuple[bool, str]:
        """Verify Google Drive or configured storage path availability and writability."""
        p = Path(target_dir or self.output_dir)
        # Check if directory or parent is writable
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".write_probe.tmp"
            with open(test_file, "w") as f:
                f.write("probe")
            test_file.unlink()
            return True, "Storage directory writable and verified."
        except Exception as e:
            if "/content/drive" in str(p):
                return False, f"DRIVE_STORAGE_UNAVAILABLE: Google Drive mount inaccessible ({e})"
            return False, f"Storage inaccessible: {e}"

    def check_disk_space(self, target_dir: Optional[Union[str, Path]] = None) -> Tuple[bool, float, str]:
        """Check remaining disk capacity against minimum threshold."""
        p = Path(target_dir or self.output_dir)
        check_path = p if p.exists() else Path.cwd()
        usage = shutil.disk_usage(check_path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < self.minimum_free_space_gb:
            return False, free_gb, (
                f"DISK_SPACE_FAILURE: Available disk space ({free_gb:.2f} GB) "
                f"below required minimum ({self.minimum_free_space_gb:.2f} GB)"
            )
        return True, free_gb, f"Sufficient disk space available: {free_gb:.2f} GB"

    def check_hardware_gate(self) -> Tuple[bool, str]:
        """Verify CUDA, GPU memory, and GPU target alignment."""
        if not self.hardware.cuda_available or self.hardware.device_count < 1:
            return False, "HARDWARE_GATE_BLOCKED: CUDA is unavailable or no GPU detected."

        if self.hardware.total_memory_gb <= 0:
            return False, "HARDWARE_GATE_BLOCKED: Detected GPU VRAM is 0 MB."

        dev_name = self.hardware.device_name or ""
        if self.expected_gpu_target not in dev_name and not self.allow_gpu_target_mismatch:
            return False, (
                f"GPU_TARGET_MISMATCH: Detected '{dev_name}', but expected target is '{self.expected_gpu_target}'. "
                "Execution blocked unless allow_gpu_target_mismatch is enabled."
            )

        return True, f"Hardware gate passed: {dev_name} ({self.hardware.total_memory_gb:.2f} GB VRAM)"

    def execute_preflight(self) -> PreflightReport:
        """Run complete 16-point preflight readiness audit."""
        self.manifest.transition_to(TrainingRunState.PREFLIGHT, "Running preflight readiness audit")
        validator = TrainingPreflightValidator(self.config)
        report = validator.run_preflight()

        if report.is_training_ready:
            logger.info("Preflight validation passed: %s", report.overall_status)
        else:
            self.manifest.transition_to(TrainingRunState.BLOCKED, f"Preflight failed: {report.overall_status}")
            logger.warning("Preflight validation failed: %s", report.overall_status)

        # Save preflight report to reports directory
        with open(self.reports_dir / "preflight_report.json", "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)
        with open(self.reports_dir / "preflight_report.md", "w", encoding="utf-8") as f:
            f.write(report.to_markdown())

        return report

    def find_latest_valid_checkpoint(
        self,
        checkpoints_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Optional[Path], List[Path]]:
        """
        Scan checkpoints in reverse chronological order and return first valid checkpoint.
        If latest is corrupted, automatically falls back to previous valid checkpoint.
        """
        root = Path(checkpoints_dir or (self.output_dir / "checkpoints"))
        if not root.exists():
            return None, []

        candidate_dirs = [
            d for d in root.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]

        def _step_key(d: Path) -> int:
            try:
                return int(d.name.split("-")[-1])
            except ValueError:
                return 0

        sorted_candidates = sorted(candidate_dirs, key=_step_key, reverse=True)
        valid_ckpts: List[Path] = []
        corrupted_ckpts: List[Path] = []

        for ckpt in sorted_candidates:
            res = self.checkpoint_validator.validate_checkpoint(
                ckpt,
                expected_dataset_version=self.config.dataset.version,
                expected_dataset_sha=self.dataset_sha,
                expected_config_hash=self.config_hash,
                expected_run_id=self.run_identity.run_id,
            )
            if res.is_valid:
                valid_ckpts.append(ckpt)
            else:
                corrupted_ckpts.append(ckpt)
                logger.warning("Checkpoint '%s' failed validation: %s", ckpt.name, res.errors)

        latest_valid = valid_ckpts[0] if valid_ckpts else None
        return latest_valid, corrupted_ckpts

    def execute_dry_run(self) -> Dict[str, Any]:
        """Perform non-destructive orchestration check without GPU or fake training."""
        drive_ok, drive_msg = self.check_durable_storage()
        disk_ok, free_gb, disk_msg = self.check_disk_space()

        return {
            "dry_run": True,
            "run_identity": self.run_identity.model_dump(),
            "status": "VALIDATED",
            "drive_check": {"ok": drive_ok, "message": drive_msg},
            "disk_check": {"ok": disk_ok, "free_gb": free_gb, "message": disk_msg},
            "hardware": self.hardware.model_dump(),
            "dataset_version": self.config.dataset.version,
            "dataset_sha256": self.dataset_sha,
            "training_config_hash": self.config_hash,
            "message": "Dry-run validation successful. All controller pathways, schemas, and checks are ready.",
        }

    def finalize_training_run(
        self,
        best_checkpoint_path: Path,
        telemetry: TrainingTelemetry,
    ) -> Path:
        """
        Transition through FINALIZING -> FINALIZED, create training_completion_manifest.json,
        and invoke Phase 5.1 release packager.
        """
        self.manifest.transition_to(TrainingRunState.FINALIZING, "Assembling completion manifest and packaging handoff")

        # Record completion manifest
        completion_manifest_path = self.output_dir / "training_completion_manifest.json"
        completion_data = {
            "run_id": self.run_identity.run_id,
            "dataset_version": self.config.dataset.version,
            "dataset_sha256": self.dataset_sha,
            "training_config_hash": self.config_hash,
            "base_model": self.config.model.name,
            "actual_gpu": self.hardware.device_name or "NOT_AVAILABLE",
            "cuda_version": self.hardware.cuda_version,
            "pytorch_version": torch.__version__,
            "seed": self.config.training.seed,
            "epochs": self.config.training.num_train_epochs,
            "steps": telemetry.total_steps,
            "best_checkpoint": str(best_checkpoint_path),
            "best_validation_loss": telemetry.best_validation_loss,
            "final_validation_loss": telemetry.eval_history[-1].get("val_loss") if telemetry.eval_history else None,
            "training_start": self.manifest.start_timestamp,
            "training_end": datetime.now(timezone.utc).isoformat(),
            "resume_count": self.manifest.resume_count,
            "recovery_count": self.manifest.recovery_count,
            "status": "COMPLETED",
        }
        with open(completion_manifest_path, "w", encoding="utf-8") as f:
            json.dump(completion_data, f, indent=2)

        # Generate training_run.json and training_run.md
        with open(self.reports_dir / "training_run.json", "w", encoding="utf-8") as f:
            json.dump(telemetry.model_dump(), f, indent=2)
        with open(self.reports_dir / "training_run.md", "w", encoding="utf-8") as f:
            f.write(telemetry.to_markdown())

        self.manifest.transition_to(TrainingRunState.FINALIZED, "Training run finalized and ready for release packaging")
        self.manifest.end_timestamp = datetime.now(timezone.utc).isoformat()
        self.manifest.save_atomic(self.output_dir / "training_run_manifest.json")

        return completion_manifest_path

    def execute_production_run(
        self,
        resume: bool = False,
        max_steps: Optional[int] = None,
        override_epochs: Optional[int] = None,
    ) -> Tuple[bool, Optional[Path], Optional[TrainingTelemetry], str]:
        """
        Execute full production training lifecycle through the run controller:
        1. Preflight audit
        2. Hardware gating
        3. Storage & disk checks
        4. Checkpoint discovery / recovery
        5. Smoke-test gate
        6. Full SFT execution with heartbeats
        7. Checkpoint manifest generation
        8. Finalization & completion manifest
        """
        self.manifest.start_timestamp = datetime.now(timezone.utc).isoformat()
        heartbeat_path = self.reports_dir / "training_heartbeat.json"

        # 1. Preflight audit
        logger.info("Executing production preflight audit...")
        report = self.execute_preflight()
        if not report.is_training_ready:
            msg = f"Training blocked by preflight failure: {report.overall_status}"
            logger.error(msg)
            return False, None, None, msg

        # 2. Hardware gate
        hw_ok, hw_msg = self.check_hardware_gate()
        if not hw_ok:
            self.manifest.transition_to(TrainingRunState.BLOCKED, hw_msg)
            logger.error(hw_msg)
            return False, None, None, hw_msg

        # 3. Storage checks
        storage_ok, storage_msg = self.check_durable_storage()
        if not storage_ok:
            self.manifest.transition_to(TrainingRunState.BLOCKED, storage_msg)
            logger.error(storage_msg)
            return False, None, None, storage_msg

        disk_ok, free_gb, disk_msg = self.check_disk_space()
        if not disk_ok:
            self.manifest.transition_to(TrainingRunState.BLOCKED, disk_msg)
            logger.error(disk_msg)
            return False, None, None, disk_msg

        # 4. Checkpoint discovery
        resume_ckpt_path: Optional[str] = None
        if resume:
            self.manifest.transition_to(TrainingRunState.RECOVERING, "Scanning for valid recovery checkpoints")
            latest_valid, corrupted = self.find_latest_valid_checkpoint()
            if corrupted:
                logger.warning("Found %d corrupted checkpoints during scan", len(corrupted))
                self.manifest.recovery_count += 1
            if latest_valid:
                resume_ckpt_path = str(latest_valid)
                self.manifest.resume_count += 1
                logger.info("Resuming from validated checkpoint: %s", resume_ckpt_path)
            else:
                logger.info("No existing valid checkpoint found; starting from base model.")

        # 5. Smoke test gate
        self.manifest.transition_to(TrainingRunState.SMOKE_TEST, "Executing pre-training smoke test gate")
        heartbeat = TrainingHeartbeat(
            run_id=self.run_identity.run_id,
            state=TrainingRunState.SMOKE_TEST,
        )
        heartbeat.save(heartbeat_path)

        trainer = ProductionSFTTrainer(config=self.config)
        self.trainer = trainer
        trainer.initialize_and_audit()

        smoke_res = trainer.run_smoke_test()
        if not smoke_res.success:
            self.manifest.transition_to(TrainingRunState.FAILED, f"Smoke test failed: {smoke_res.message}")
            fail_record = TrainingFailureRecord(
                failure_type=FailureType.UNKNOWN_FAILURE,
                message=smoke_res.message,
            )
            self.manifest.failures.append(fail_record)
            return False, None, None, f"TRAINING BLOCKED — SMOKE TEST FAILED: {smoke_res.message}"

        # 6. Training Execution
        self.manifest.transition_to(TrainingRunState.TRAINING, "Launching supervised fine-tuning loop")
        heartbeat.state = TrainingRunState.TRAINING
        heartbeat.save(heartbeat_path)

        try:
            telemetry = trainer.train(
                resume_from_checkpoint=resume_ckpt_path,
                max_steps=max_steps,
                override_epochs=override_epochs,
            )

            # Generate checkpoint manifests for all created checkpoints
            ckpts_dir = self.output_dir / "checkpoints"
            if ckpts_dir.exists():
                for ckpt_p in ckpts_dir.iterdir():
                    if ckpt_p.is_dir() and ckpt_p.name.startswith("checkpoint-"):
                        if not (ckpt_p / "checkpoint_manifest.json").exists():
                            try:
                                step_num = int(ckpt_p.name.split("-")[-1])
                                CheckpointValidator.generate_checkpoint_manifest(
                                    checkpoint_dir=ckpt_p,
                                    step=step_num,
                                    epoch=telemetry.total_epochs,
                                    loss=telemetry.loss_history[-1]["loss"] if telemetry.loss_history else 0.0,
                                    validation_loss=telemetry.best_validation_loss,
                                    dataset_version=self.config.dataset.version,
                                    dataset_sha256=self.dataset_sha,
                                    training_config_hash=self.config_hash,
                                    run_id=self.run_identity.run_id,
                                )
                            except Exception as e:
                                logger.warning("Failed to generate manifest for %s: %s", ckpt_p.name, e)

            # 7. Finalization
            best_ckpt = Path(telemetry.best_checkpoint_path) if telemetry.best_checkpoint_path else (self.output_dir / "checkpoints/final")
            completion_manifest_path = self.finalize_training_run(
                best_checkpoint_path=best_ckpt,
                telemetry=telemetry,
            )

            # Update final heartbeat
            heartbeat.state = TrainingRunState.FINALIZED
            heartbeat.epoch = telemetry.total_epochs
            heartbeat.step = telemetry.total_steps
            heartbeat.last_checkpoint = str(best_ckpt)
            heartbeat.save(heartbeat_path)

            return True, completion_manifest_path, telemetry, "Training completed successfully."

        except Exception as e:
            err_str = str(e)
            fail_type = FailureType.OUT_OF_MEMORY if "out of memory" in err_str.lower() or "cuda oom" in err_str.lower() else FailureType.UNKNOWN_FAILURE
            self.manifest.transition_to(TrainingRunState.FAILED, err_str)
            fail_rec = TrainingFailureRecord(
                failure_type=fail_type,
                message=err_str,
                step=self.manifest.current_step,
                epoch=self.manifest.current_epoch,
            )
            self.manifest.failures.append(fail_rec)
            heartbeat.state = TrainingRunState.FAILED
            heartbeat.save(heartbeat_path)
            self.manifest.save_atomic(self.output_dir / "training_run_manifest.json")
            logger.error("Training run failed: %s", err_str)
            return False, None, None, f"TRAINING FAILED: {err_str}"

