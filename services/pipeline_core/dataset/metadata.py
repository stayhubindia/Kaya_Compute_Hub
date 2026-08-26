"""
Metadata & Provenance Management.
Ensures rigorous tracking of data origin, generator versions, timestamps, and taxonomy assignments.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.dataset.schema import DatasetRecord, ProvenanceInfo, RecordMetadata, SourceType


class MetadataEnricher:
    """Enriches and ensures completeness of record provenance and taxonomy fields."""

    def __init__(
        self,
        pipeline_version: str = "1.0.0",
        default_source_type: str = SourceType.UNKNOWN.value,
        generator_name: Optional[str] = None,
    ):
        self.pipeline_version = pipeline_version
        self.default_source_type = default_source_type
        self.generator_name = generator_name

    def enrich_record(
        self,
        record: DatasetRecord,
        source_id: Optional[str] = None,
        force_timestamp: bool = False,
    ) -> DatasetRecord:
        """Enriches a record's metadata with standard provenance fields."""
        meta = record.metadata

        new_source_id = meta.source_id or (meta.provenance.source_id if meta.provenance else None) or source_id or f"rec_{uuid.uuid4().hex[:12]}"
        new_timestamp = (
            datetime.now(timezone.utc).isoformat()
            if (force_timestamp or not meta.created_at)
            else meta.created_at
        )
        new_source_type = meta.source_type or (meta.provenance.source_type if meta.provenance else None) or self.default_source_type
        new_source = meta.source or (meta.provenance.source if meta.provenance else None) or "unknown"
        new_license = meta.license or (meta.provenance.license if meta.provenance else None)
        new_generator = meta.generator or (meta.provenance.generator if meta.provenance else None) or self.generator_name
        new_generator_version = meta.generator_version or (meta.provenance.generator_version if meta.provenance else None) or self.pipeline_version

        provenance = ProvenanceInfo(
            source_type=new_source_type,
            source=new_source,
            source_id=new_source_id,
            license=new_license,
            created_at=new_timestamp,
            generator=new_generator,
            generator_version=new_generator_version,
            source_url=meta.provenance.source_url if meta.provenance else None,
        )

        updated_meta = RecordMetadata(
            domain=meta.domain,
            topic=meta.topic,
            task_type=meta.task_type,
            difficulty=meta.difficulty,
            quality_score=meta.quality_score,
            source=new_source,
            source_type=new_source_type,
            created_at=new_timestamp,
            source_id=new_source_id,
            license=new_license,
            generator=new_generator,
            generator_version=new_generator_version,
            dimensions=meta.dimensions,
            provenance=provenance,
        )

        return DatasetRecord(messages=record.messages, metadata=updated_meta)

    def enrich_records(self, records: List[DatasetRecord]) -> List[DatasetRecord]:
        """Batch enrichment of records."""
        return [self.enrich_record(r) for r in records]
