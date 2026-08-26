"""
Tests for Training Checkpoint Manager & Version Protection (Phase 4.1).
"""

import pytest
from pathlib import Path

from src.training.checkpoint import TrainingCheckpointManager
from src.training.config import CheckpointConfig


def test_checkpoint_saving_and_rotation(tmp_path):
    cfg = CheckpointConfig(save_total_limit=2)
    mgr = TrainingCheckpointManager(
        output_dir=tmp_path,
        config=cfg,
        dataset_version="dataset-v1.0",
        dataset_sha256="test_sha",
    )

    # Save 3 checkpoints
    p1 = mgr.save_checkpoint_metadata(step=25, epoch=1.0, loss=2.5, learning_rate=2e-4)
    p2 = mgr.save_checkpoint_metadata(step=50, epoch=2.0, loss=2.1, learning_rate=1e-4)
    p3 = mgr.save_checkpoint_metadata(step=75, epoch=3.0, loss=1.8, learning_rate=5e-5)

    ckpts = mgr.list_checkpoints()
    assert len(ckpts) == 2  # p1 should be pruned
    assert ckpts[0].name == "checkpoint-50"
    assert ckpts[1].name == "checkpoint-75"


def test_checkpoint_version_protection(tmp_path):
    cfg = CheckpointConfig(enforce_dataset_version_lock=True)
    mgr = TrainingCheckpointManager(
        output_dir=tmp_path,
        config=cfg,
        dataset_version="dataset-v1.0",
        dataset_sha256="test_sha",
    )

    ckpt_path = mgr.save_checkpoint_metadata(step=25, epoch=1.0, loss=2.5, learning_rate=2e-4)

    # Validate with matching version
    meta = mgr.validate_resume_checkpoint(ckpt_path)
    assert meta.dataset_version == "dataset-v1.0"

    # Validate with different version target
    mgr_diff = TrainingCheckpointManager(
        output_dir=tmp_path,
        config=cfg,
        dataset_version="dataset-v2.0",  # Different dataset target
    )
    with pytest.raises(ValueError, match="Checkpoint version mismatch"):
        mgr_diff.validate_resume_checkpoint(ckpt_path)
