"""
Checkpoint Manager for Scientific Instruction Generation (Phase 3.4).
Provides atomic persistence and resumability tracking for chunk processing states.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from src.generation.models import (
    ChunkCheckpointStatus,
    GenerationCheckpoint,
)


class ChunkCheckpointManager:
    """Manages atomic persistence and resumability of scientific dataset generation."""

    def __init__(self, checkpoint_path: Union[str, Path]):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint: GenerationCheckpoint = self._load_or_initialize()

    def _load_or_initialize(self) -> GenerationCheckpoint:
        if self.checkpoint_path.is_file():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return GenerationCheckpoint.model_validate(data)
            except Exception:
                pass
        return GenerationCheckpoint()

    def save(self) -> None:
        """Atomically writes checkpoint to disk."""
        self.checkpoint.updated_at = datetime.now(timezone.utc).isoformat()
        tmp_path = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(self.checkpoint.model_dump_json(indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.checkpoint_path)

    def is_chunk_completed(self, chunk_id: str) -> bool:
        state = self.checkpoint.chunk_states.get(chunk_id, {})
        return state.get("status") == ChunkCheckpointStatus.COMPLETED.value

    def mark_chunk_processing(self, chunk_id: str) -> None:
        self.checkpoint.chunk_states[chunk_id] = {
            "status": ChunkCheckpointStatus.PROCESSING.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def mark_chunk_completed(
        self,
        chunk_id: str,
        generated: int,
        accepted: int,
        rejected: int,
    ) -> None:
        self.checkpoint.chunk_states[chunk_id] = {
            "status": ChunkCheckpointStatus.COMPLETED.value,
            "generated": generated,
            "accepted": accepted,
            "rejected": rejected,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.checkpoint.completed_chunks = sum(
            1 for s in self.checkpoint.chunk_states.values() if s.get("status") == ChunkCheckpointStatus.COMPLETED.value
        )
        self.checkpoint.total_candidates_generated += generated
        self.checkpoint.total_candidates_accepted += accepted
        self.checkpoint.total_candidates_rejected += rejected
        if self.checkpoint.completed_chunks % 100 == 0:
            self.save()

    def mark_chunk_failed(self, chunk_id: str, error: str) -> None:
        self.checkpoint.chunk_states[chunk_id] = {
            "status": ChunkCheckpointStatus.FAILED.value,
            "error": error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.checkpoint.failed_chunks = sum(
            1 for s in self.checkpoint.chunk_states.values() if s.get("status") == ChunkCheckpointStatus.FAILED.value
        )
        self.save()

    def reset_failed(self) -> None:
        for chunk_id, st in list(self.checkpoint.chunk_states.items()):
            if st.get("status") == ChunkCheckpointStatus.FAILED.value:
                del self.checkpoint.chunk_states[chunk_id]
        self.checkpoint.failed_chunks = 0
        self.save()
