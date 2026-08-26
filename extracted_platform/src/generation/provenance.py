"""
Scientific Instruction Provenance and Lineage Manager (Phase 3.4).
Ensures full traceable cryptographic provenance linking generated candidate records
back to their originating Phase 3.3 knowledge documents, chunks, and sections.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.dataset.schema import DatasetRecord, ProvenanceInfo, RecordMetadata
from src.generation.models import ExtendedProvenance, KnowledgeUnit


class InstructionProvenanceManager:
    """Manages lineage and rights compliance for instruction generation."""

    def __init__(self, generator_name: str = "scientific_instruction_engine", version: str = "1.0.0"):
        self.generator_name = generator_name
        self.version = version

    def create_provenance(
        self,
        unit: KnowledgeUnit,
        seed: int = 42,
        generation_method: str = "scientific_rule_based",
        template_id: Optional[str] = None,
    ) -> ExtendedProvenance:
        """Constructs an ExtendedProvenance instance for a candidate record."""
        needs_verification = (
            unit.license_status != "KNOWN"
            or unit.license is None
            or unit.internal_only
        )

        return ExtendedProvenance(
            source_type=unit.source_type,
            source=unit.source,
            source_id=unit.chunk_id or unit.document_id,
            source_url=unit.source_url,
            license=unit.license,
            license_status=unit.license_status,
            internal_only=unit.internal_only,
            rights_verification_required=needs_verification,
            created_at=datetime.now(timezone.utc).isoformat(),
            generator=self.generator_name,
            generator_version=self.version,
            knowledge_document_id=unit.document_id,
            knowledge_chunk_id=unit.chunk_id,
            knowledge_section_id=unit.section_id,
            generation_template_id=template_id,
            generation_seed=seed,
            generation_method=generation_method,
        )

    def attach_provenance(
        self,
        record: DatasetRecord,
        provenance: ExtendedProvenance,
    ) -> DatasetRecord:
        """Attaches provenance to record metadata while preserving canonical schema compatibility."""
        meta = record.metadata
        prov_info = provenance.to_provenance_info()

        updated_meta = RecordMetadata(
            domain=meta.domain,
            topic=meta.topic,
            task_type=meta.task_type,
            difficulty=meta.difficulty,
            quality_score=meta.quality_score,
            source=provenance.source,
            source_type=provenance.source_type,
            created_at=provenance.created_at,
            source_id=provenance.source_id,
            license=provenance.license,
            generator=provenance.generator,
            generator_version=provenance.generator_version,
            dimensions=meta.dimensions,
            provenance=prov_info,
        )

        return DatasetRecord(messages=record.messages, metadata=updated_meta)
