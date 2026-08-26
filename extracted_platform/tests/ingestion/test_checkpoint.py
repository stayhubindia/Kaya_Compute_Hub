"""
Unit tests for Ingestion Checkpoint Manager (Phase 3.3).
"""

import pytest
from src.ingestion.checkpoint import IngestionCheckpointManager


def test_checkpoint_atomic_save_and_reload(tmp_path):
    mgr = IngestionCheckpointManager(checkpoint_dir=tmp_path, execution_id="test-exec-1")
    mgr.set_document_state("doc_001", "COMPLETED")
    mgr.set_document_state("doc_002", "PARTIAL")
    mgr.set_document_state("doc_003", "FAILED")
    mgr.save_atomic()

    assert (tmp_path / "ingestion_checkpoint.json").is_file()

    # Load in new manager instance
    mgr2 = IngestionCheckpointManager(checkpoint_dir=tmp_path)
    loaded = mgr2.load()

    assert loaded is True
    assert mgr2.execution_id == "test-exec-1"
    assert mgr2.is_completed("doc_001") is True
    assert mgr2.is_completed("doc_002") is False
    assert mgr2.is_completed("doc_003") is False
    assert mgr2.document_states["doc_003"] == "FAILED"
