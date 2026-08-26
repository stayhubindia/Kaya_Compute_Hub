"""
Training Checkpoint Manager & Lifecycle Governance (Phase 4.1 & 4.2).
Provides full checkpoint persistence, rotation with save_total_limit,
and dataset-version isolation protection to prevent resuming across differing datasets.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
from pydantic import BaseModel, Field

from src.training.config import CheckpointConfig


class CheckpointMetadata(BaseModel):
    """Metadata describing a saved training checkpoint."""
    checkpoint_name: str
    global_step: int
    epoch: float
    loss: float
    learning_rate: float
    dataset_version: str
    dataset_sha256: str
    config_hash: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def save_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)


class TrainingCheckpointManager:
    """Manages version-locked checkpoint creation, metadata tracking, rotation, and resumption."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        config: CheckpointConfig,
        dataset_version: str = "dataset-v1.0",
        dataset_sha256: str = "",
        config_hash: str = "",
    ):
        self.output_dir = Path(output_dir)
        self.config = config
        self.dataset_version = dataset_version
        self.dataset_sha256 = dataset_sha256
        self.config_hash = config_hash

    def get_checkpoint_dir(self, step: int) -> Path:
        """Construct path for a specific step checkpoint."""
        return self.output_dir / "checkpoints" / f"checkpoint-{step}"

    def save_checkpoint_metadata(
        self,
        step: int,
        epoch: float,
        loss: float,
        learning_rate: float,
    ) -> Path:
        """Record checkpoint metadata and enforce rotation limits."""
        ckpt_dir = self.get_checkpoint_dir(step)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        meta = CheckpointMetadata(
            checkpoint_name=ckpt_dir.name,
            global_step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            dataset_version=self.dataset_version,
            dataset_sha256=self.dataset_sha256,
            config_hash=self.config_hash,
        )
        meta_path = ckpt_dir / "checkpoint_metadata.json"
        meta.save_json(meta_path)

        self._rotate_checkpoints()
        return ckpt_dir

    def save_full_checkpoint(
        self,
        step: int,
        epoch: float,
        loss: float,
        learning_rate: float,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        trainer_state: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
    ) -> Path:
        """
        Save complete training state:
        - LoRA adapter weights / config
        - Optimizer state dict
        - Scheduler state dict
        - Trainer state JSON
        - Tokenizer files
        - Checkpoint metadata JSON
        """
        ckpt_dir = self.get_checkpoint_dir(step)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save LoRA adapter weights
        if model is not None:
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(str(ckpt_dir))
            else:
                torch.save(model.state_dict(), ckpt_dir / "model_weights.pt")

        # 2. Save Optimizer state
        if optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        # 3. Save Scheduler state
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

        # 4. Save Trainer state
        state_payload = trainer_state or {}
        state_payload.update({
            "global_step": step,
            "epoch": epoch,
            "loss": loss,
            "learning_rate": learning_rate,
            "dataset_version": self.dataset_version,
        })
        with open(ckpt_dir / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump(state_payload, f, indent=2)

        # 5. Save Tokenizer
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            try:
                tokenizer.save_pretrained(str(ckpt_dir))
            except Exception:
                pass

        # 6. Save Checkpoint metadata
        meta = CheckpointMetadata(
            checkpoint_name=ckpt_dir.name,
            global_step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            dataset_version=self.dataset_version,
            dataset_sha256=self.dataset_sha256,
            config_hash=self.config_hash,
        )
        meta.save_json(ckpt_dir / "checkpoint_metadata.json")

        self._rotate_checkpoints()
        return ckpt_dir

    def list_checkpoints(self) -> List[Path]:
        """List all valid step checkpoints sorted chronologically."""
        ckpts_root = self.output_dir / "checkpoints"
        if not ckpts_root.exists():
            return []

        dirs = [d for d in ckpts_root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]

        def _get_step(d: Path) -> int:
            try:
                return int(d.name.split("-")[-1])
            except ValueError:
                return 0

        return sorted(dirs, key=_get_step)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Return path to the most recent checkpoint if available."""
        ckpts = self.list_checkpoints()
        return ckpts[-1] if ckpts else None

    def validate_resume_checkpoint(self, checkpoint_path: Union[str, Path]) -> CheckpointMetadata:
        """
        Validate checkpoint before resuming: ensures metadata exists and dataset_version matches.
        Raises ValueError if checkpoint originates from a different dataset release.
        """
        p = Path(checkpoint_path)
        meta_file = p / "checkpoint_metadata.json"
        if not meta_file.exists():
            raise ValueError(f"Invalid checkpoint at '{p}': missing 'checkpoint_metadata.json'")

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = CheckpointMetadata(**data)

        if self.config.enforce_dataset_version_lock:
            if meta.dataset_version != self.dataset_version:
                raise ValueError(
                    f"Checkpoint version mismatch: checkpoint '{p.name}' was trained on "
                    f"dataset '{meta.dataset_version}', but current target is '{self.dataset_version}'."
                )

        return meta

    def _rotate_checkpoints(self) -> None:
        """Prune oldest checkpoints when exceeding save_total_limit."""
        if not self.config.save_total_limit or self.config.save_total_limit <= 0:
            return

        ckpts = self.list_checkpoints()
        while len(ckpts) > self.config.save_total_limit:
            oldest = ckpts.pop(0)
            try:
                shutil.rmtree(oldest)
            except Exception:
                pass
