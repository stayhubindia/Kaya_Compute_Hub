"""
Unit tests for GenerationCheckpointManager (src/generation/checkpoint.py).
"""

import pytest
from pathlib import Path
from src.generation.checkpoint import GenerationCheckpointManager


def test_checkpoint_lifecycle(tmp_path: Path):
    ckpt_path = tmp_path / "checkpoint.json"
    mgr = GenerationCheckpointManager(ckpt_path)

    assert not mgr.is_processed("unit_1")

    mgr.record_completed("unit_1", generated_count=2, accepted_count=2, rejected_count=0)
    mgr.record_failed("unit_2", error_msg="Validation failed")
    mgr.save()

    assert mgr.is_processed("unit_1")
    assert mgr.is_processed("unit_2")
    assert not mgr.is_processed("unit_3")

    # Reload in a new instance
    mgr_loaded = GenerationCheckpointManager(ckpt_path)
    assert mgr_loaded.is_processed("unit_1")
    assert mgr_loaded.is_processed("unit_2")
    assert mgr_loaded.total_generated == 2
    assert mgr_loaded.total_accepted == 2
