"""
Source Registry Abstraction.
Provides strongly typed registration, discovery, validation, and manifest loading
for diverse dataset origins (human-authored, synthetic, documentation, external datasets).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from src.dataset.schema import ProvenanceInfo, SourceType


class SourceDefinition(BaseModel):
    """Declarative definition of an authoritative data source."""

    source_id: str
    source_type: str = SourceType.UNKNOWN.value
    name: str
    description: Optional[str] = None
    version: Optional[str] = "1.0"
    license: Optional[str] = None
    generator: Optional[str] = None
    generator_version: Optional[str] = None
    created_at: Optional[str] = None
    url: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source_id must be a non-empty string.")
        return v.strip()

    @field_validator("source_type", mode="before")
    @classmethod
    def normalize_source_type(cls, v: Any) -> str:
        if isinstance(v, SourceType):
            return v.value
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in [s.value for s in SourceType]:
                return clean
        return str(v).strip().lower() if v else SourceType.UNKNOWN.value

    def create_provenance(
        self,
        item_id: Optional[str] = None,
        created_at: Optional[str] = None,
        **overrides: Any,
    ) -> ProvenanceInfo:
        """Constructs a ProvenanceInfo record bound to this registered source."""
        return ProvenanceInfo(
            source_type=overrides.get("override_source_type") or self.source_type,
            source=overrides.get("override_source") or self.name,
            source_id=item_id or overrides.get("source_id") or self.source_id,
            license=overrides.get("license") or self.license,
            created_at=created_at or overrides.get("created_at") or self.created_at or datetime.now(timezone.utc).isoformat(),
            generator=overrides.get("generator") or self.generator,
            generator_version=overrides.get("generator_version") or self.generator_version,
            source_url=overrides.get("source_url") or self.url,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "name": self.name,
            "version": self.version,
            "license": self.license,
        }
        if self.description:
            data["description"] = self.description
        if self.generator:
            data["generator"] = self.generator
        if self.generator_version:
            data["generator_version"] = self.generator_version
        if self.created_at:
            data["created_at"] = self.created_at
        if self.url:
            data["url"] = self.url
        if self.extra:
            data["extra"] = self.extra
        return data


class SourceRegistry:
    """In-memory registry managing registered training data sources."""

    def __init__(self):
        self._sources: Dict[str, SourceDefinition] = {}

    def register_source(self, definition: SourceDefinition, overwrite: bool = False) -> None:
        """Registers a source definition. Raises ValueError if source_id already exists and overwrite is False."""
        self.validate_source(definition)
        if definition.source_id in self._sources and not overwrite:
            raise ValueError(f"Duplicate source_id '{definition.source_id}' already registered.")
        self._sources[definition.source_id] = definition

    def lookup_source(self, source_id: str) -> Optional[SourceDefinition]:
        """Returns the registered source definition or None if not found."""
        return self._sources.get(source_id)

    def get_source(self, source_id: str) -> SourceDefinition:
        """Returns the registered source definition or raises KeyError."""
        if source_id not in self._sources:
            raise KeyError(f"Source '{source_id}' is not registered in source registry.")
        return self._sources[source_id]

    def list_sources(self) -> List[SourceDefinition]:
        """Lists all registered source definitions."""
        return list(self._sources.values())

    def validate_source(self, definition: SourceDefinition) -> bool:
        """Validates that a source definition is well-formed."""
        if not definition.source_id:
            raise ValueError("source_id cannot be blank.")
        if not definition.name:
            raise ValueError("name cannot be blank.")
        return True

    def load_manifest(self, manifest_path: Union[str, Path]) -> int:
        """
        Loads sources from a YAML manifest file (e.g. configs/sources.yaml).
        Returns the count of newly registered sources.
        """
        path = Path(manifest_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source manifest file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        sources_list = data.get("sources", [])
        loaded_count = 0
        for entry in sources_list:
            defn = SourceDefinition.model_validate(entry)
            self.register_source(defn, overwrite=True)
            loaded_count += 1

        return loaded_count

    @classmethod
    def from_manifest(cls, manifest_path: Union[str, Path]) -> SourceRegistry:
        """Factory method to construct a registry directly from a manifest file."""
        registry = cls()
        registry.load_manifest(manifest_path)
        return registry
