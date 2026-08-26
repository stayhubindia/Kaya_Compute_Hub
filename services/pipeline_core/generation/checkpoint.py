"""
Instruction Generation Checkpoint Manager (Phase 3.4).
Provides atomic state persistence and resumption for long-running dataset generation jobs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


class GenerationCheckpointManager:
    """Manages atomic progress checkpointing for instruction generation."""

    def __init__(self, checkpoint_path: Union[str, Path]):
        self.checkpoint_path = Path(checkpoint_path)
        self.completed_unit_ids: Set[str] = set()
        self.failed_unit_ids: Dict[str, str] = {}
        self.total_generated: int = 0
        self.total_accepted: int = 0
        self.total_rejected: int = 0
        self.start_time: str = datetime.now(timezone.utc).isoformat()
        self.last_updated: str = self.start_time

        self._load()

    def _load(self) -> None:
        """Loads state from existing checkpoint file if available."""
        if not self.checkpoint_path.is_file():
            return
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.completed_unit_ids = set(data.get("completed_unit_ids", []))
            self.failed_unit_ids = data.get("failed_unit_ids", {})
            self.total_generated = data.get("total_generated", 0)
            self.total_accepted = data.get("total_accepted", 0)
            self.total_rejected = data.get("total_rejected", 0)
            self.start_time = data.get("start_time", self.start_time)
            self.last_updated = data.get("last_updated", self.last_updated)
        except Exception:
            # If checkpoint is corrupted, start fresh
            pass

    def is_processed(self, unit_id: str) -> bool:
        """Checks if a knowledge unit has already been processed."""
        return unit_id in self.completed_unit_ids or unit_id in self.failed_unit_ids

    def record_completed(self, unit_id: str, generated_count: int, accepted_count: int, rejected_count: int) -> None:
        """Records a successfully processed knowledge unit."""
        self.completed_unit_ids.add(unit_id)
        self.total_generated += generated_count
        self.total_accepted += accepted_count
        self.total_rejected += rejected_count
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def record_failed(self, unit_id: str, error_msg: str) -> None:
        """Records a failed knowledge unit."""
        self.failed_unit_ids[unit_id] = error_msg
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def save(self) -> None:
        """Atomically saves checkpoint state to disk."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "start_time": self.start_time,
            "last_updated": self.last_updated,
            "completed_unit_ids": sorted(list(self.completed_unit_ids)),
            "failed_unit_ids": self.failed_unit_ids,
            "total_completed_units": len(self.completed_unit_ids),
            "total_failed_units": len(self.failed_unit_ids),
            "total_generated": self.total_generated,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
        }
        tmp_file = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_file.replace(self.checkpoint_path)
