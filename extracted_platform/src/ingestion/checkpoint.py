"""
Ingestion Checkpoint Manager (Phase 3.3).
Maintains atomic, crash-resilient execution states for large-scale document ingestion runs.
Enables instant resumption without reprocessing completed documents.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class IngestionCheckpointManager:
    """Manages atomic checkpoint saving and resumption tracking."""

    def __init__(self, checkpoint_dir: Union[str, Path], execution_id: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "ingestion_checkpoint.json"
        self.execution_id = execution_id or f"ingest-{uuid.uuid4().hex[:8]}"

        self.document_states: Dict[str, str] = {}  # doc_id -> state (PENDING, PROCESSING, COMPLETED, PARTIAL, FAILED)
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.last_updated = self.created_at

    def load(self) -> bool:
        """Loads existing checkpoint if present. Returns True if successfully resumed."""
        if not self.checkpoint_path.is_file():
            return False

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.execution_id = data.get("execution_id", self.execution_id)
            self.created_at = data.get("created_at", self.created_at)
            self.last_updated = data.get("last_updated", self.last_updated)
            self.document_states = data.get("document_states", {})
            logger.info(f"Loaded checkpoint with {len(self.document_states)} tracked documents.")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint file: {e}")
            return False

    def is_completed(self, doc_id: str) -> bool:
        """Checks if a document ID was already successfully completed."""
        return self.document_states.get(doc_id) == "COMPLETED"

    def set_document_state(self, doc_id: str, state: str) -> None:
        """Updates the state for a single document."""
        self.document_states[doc_id] = state
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def save_atomic(self) -> None:
        """Writes checkpoint state atomically using a temp file."""
        data = {
            "execution_id": self.execution_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "total_documents": len(self.document_states),
            "completed_count": sum(1 for s in self.document_states.values() if s == "COMPLETED"),
            "failed_count": sum(1 for s in self.document_states.values() if s == "FAILED"),
            "partial_count": sum(1 for s in self.document_states.values() if s == "PARTIAL"),
            "document_states": self.document_states,
        }

        tmp_file = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Atomic replacement
        tmp_file.replace(self.checkpoint_path)
