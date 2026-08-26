"""
Provenance Tracking and Registry Integration (Phase 3.3).
Integrates ingestion pipelines with the authoritative SourceRegistry and ProvenanceInfo schemas.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.dataset.schema import ProvenanceInfo, SourceType
from src.dataset.source_registry import SourceDefinition, SourceRegistry


class ProvenanceTracker:
    """Manages provenance generation and registration for ingested documents."""

    def __init__(self, registry: Optional[SourceRegistry] = None):
        self.registry = registry or SourceRegistry()

    def register_default_sources(self) -> None:
        """Registers canonical sources for NPTEL, arXiv, and MIT OpenCourseWare."""
        sources = [
            SourceDefinition(
                source_id="nptel-physics-v1",
                source_type=SourceType.LICENSED_MATERIAL.value,
                name="NPTEL Physics Courseware",
                description="National Programme on Technology Enhanced Learning (NPTEL) course notes and slides.",
                version="1.0",
                license="CC-BY-NC-SA-4.0",
                created_at="2026-08-13T00:00:00Z",
                url="https://nptel.ac.in",
            ),
            SourceDefinition(
                source_id="arxiv-astrophysics-v1",
                source_type=SourceType.DOCUMENTATION.value,
                name="arXiv Astrophysics Preprints",
                description="Open scientific preprints from arXiv Astrophysics (astro-ph) repository.",
                version="1.0",
                license="arXiv Non-exclusive License",
                created_at="2026-08-13T00:00:00Z",
                url="https://arxiv.org",
            ),
            SourceDefinition(
                source_id="mit-ocw-physics-v1",
                source_type=SourceType.LICENSED_MATERIAL.value,
                name="MIT OpenCourseWare Physics",
                description="MIT OpenCourseWare physics lecture notes and curriculum materials.",
                version="1.0",
                license="CC-BY-NC-SA-4.0",
                created_at="2026-08-13T00:00:00Z",
                url="https://ocw.mit.edu",
            ),
        ]
        for s in sources:
            self.registry.register_source(s, overwrite=True)

    def create_provenance(
        self,
        source: str,
        document_id: str,
        source_type: Optional[str] = None,
        license: Optional[str] = None,
        source_url: Optional[str] = None,
        generator_version: str = "1.0.0",
    ) -> ProvenanceInfo:
        """Constructs an immutable ProvenanceInfo instance for a document or chunk."""
        # Check if source exists in registry
        reg_source = self.registry.lookup_source(source)
        resolved_type = source_type or (reg_source.source_type if reg_source else SourceType.DOCUMENTATION.value)
        resolved_license = license or (reg_source.license if reg_source else None)
        resolved_url = source_url or (reg_source.url if reg_source else None)

        return ProvenanceInfo(
            source_type=resolved_type,
            source=source,
            source_id=document_id,
            license=resolved_license,
            created_at=datetime.now(timezone.utc).isoformat(),
            generator="knowledge_ingestion_engine",
            generator_version=generator_version,
            source_url=resolved_url,
        )
