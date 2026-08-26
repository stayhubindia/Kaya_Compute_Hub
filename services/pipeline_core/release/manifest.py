"""
Release Manifest Specification & Lifecycle Governance (Phase 5.1).
Defines strongly-typed ReleaseManifest schema, lifecycle states,
deterministic release ID construction, and atomic persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReleaseStatus(str, Enum):
    """Lifecycle states for adapter packaging and release."""
    PLANNED = "PLANNED"          # No adapter validated or packaging planned
    BUILDING = "BUILDING"        # Packaging in progress
    VALIDATING = "VALIDATING"    # Artifacts exist and are undergoing validation
    READY = "READY"              # All mandatory validations pass
    RELEASED = "RELEASED"        # Explicit final release state
    INVALID = "INVALID"          # Integrity, compatibility, or format failure


def construct_release_id(
    base_model_id: str,
    adapter_version: str,
    dataset_version: str,
    training_config_hash: str,
) -> str:
    """
    Construct a deterministic release identity from immutable inputs.
    Example: qwen3-4b-qlora-v1.0
    """
    # Clean base model name (e.g. Qwen/Qwen3-4B-Base -> qwen3-4b)
    base_clean = base_model_id.split("/")[-1].lower().replace("-base", "")
    v_clean = adapter_version.lower()
    if not v_clean.startswith("v"):
        v_clean = f"v{v_clean}"
    return f"{base_clean}-qlora-{v_clean}"


class ReleaseManifest(BaseModel):
    """Immutable release manifest for QLoRA adapter distribution and verification."""
    release_id: str
    release_version: str = "v1.0"
    status: ReleaseStatus = ReleaseStatus.PLANNED
    status_reason: Optional[str] = None
    base_model: Dict[str, Any] = Field(default_factory=dict)
    adapter_type: str = "QLoRA"
    dataset_version: str = "dataset-v1.0"
    dataset_sha256: str = ""
    training_config_hash: str = ""
    generation_config_hash: str = ""
    benchmark_version: str = "benchmark-v1.0"
    benchmark_sha256: str = ""
    baseline_experiment_id: str = "NOT_AVAILABLE"
    adapter_experiment_id: str = "NOT_AVAILABLE"
    artifact_inventory: List[str] = Field(default_factory=list)
    artifact_hashes: Dict[str, str] = Field(default_factory=dict)
    compatibility: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    reproducibility: Dict[str, Any] = Field(default_factory=dict)
    creation_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool_versions: Dict[str, str] = Field(default_factory=dict)

    def save_atomic(self, path: Union[str, Path]) -> None:
        """Persist manifest atomically via temp file replacement."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.updated_timestamp = datetime.now(timezone.utc).isoformat()
        tmp_p = p.with_suffix(".tmp")
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
        tmp_p.replace(p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> ReleaseManifest:
        """Load manifest from JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found at: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
