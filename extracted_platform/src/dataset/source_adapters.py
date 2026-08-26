"""
Source Adapter Architecture.
Provides modular adapters mapping diverse data source formats (existing datasets,
documentation, human-authored examples, and synthetic data) to canonical DatasetRecord objects
while immutably binding provenance metadata.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from src.dataset.loader import RawRecord
from src.dataset.normalizer import DatasetNormalizer
from src.dataset.schema import (
    DatasetRecord,
    Message,
    ProvenanceInfo,
    RecordMetadata,
    Role,
    SourceType,
)
from src.dataset.source_registry import SourceDefinition


class SourceAdapter(ABC):
    """Abstract base adapter for training data sources."""

    def __init__(
        self,
        source_definition: Optional[SourceDefinition] = None,
        normalizer: Optional[DatasetNormalizer] = None,
    ):
        self.source_definition = source_definition
        self.normalizer = normalizer or DatasetNormalizer()

    @abstractmethod
    def adapt(self, raw_item: Union[RawRecord, Dict[str, Any]]) -> DatasetRecord:
        """Transforms a raw item into a canonical DatasetRecord with intact provenance."""
        pass

    def adapt_batch(
        self, raw_items: List[Union[RawRecord, Dict[str, Any]]]
    ) -> List[DatasetRecord]:
        """Transforms a batch of raw records."""
        return [self.adapt(item) for item in raw_items]

    def _build_provenance(
        self,
        item_id: Optional[str] = None,
        explicit_provenance: Optional[Union[ProvenanceInfo, Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> ProvenanceInfo:
        """Constructs a ProvenanceInfo object respecting registered source definition or item overrides."""
        if isinstance(explicit_provenance, ProvenanceInfo):
            return explicit_provenance
        if isinstance(explicit_provenance, dict):
            return ProvenanceInfo.model_validate(explicit_provenance)

        clean_overrides = {k: v for k, v in overrides.items() if v is not None}

        if self.source_definition:
            return self.source_definition.create_provenance(item_id=item_id, **clean_overrides)

        return ProvenanceInfo(
            source_type=clean_overrides.get("source_type", SourceType.UNKNOWN.value),
            source=clean_overrides.get("source", "unknown"),
            source_id=item_id or clean_overrides.get("source_id"),
            license=clean_overrides.get("license"),
            created_at=clean_overrides.get("created_at") or datetime.now(timezone.utc).isoformat(),
            generator=clean_overrides.get("generator"),
            generator_version=clean_overrides.get("generator_version"),
            source_url=clean_overrides.get("source_url"),
        )


class ExistingDatasetAdapter(SourceAdapter):
    """Adapts external or existing formatted conversational datasets (ShareGPT, Alpaca, ChatML, Prompt-Response)."""

    def adapt(self, raw_item: Union[RawRecord, Dict[str, Any]]) -> DatasetRecord:
        if isinstance(raw_item, RawRecord):
            normalized_dict = self.normalizer.normalize_record(raw_item)
        else:
            raw_rec = RawRecord(data=raw_item, source_file=raw_item.get("_source_file"))
            normalized_dict = self.normalizer.normalize_record(raw_rec)

        messages = [
            Message(role=Role(m["role"]), content=m["content"])
            for m in normalized_dict["messages"]
        ]
        meta_dict = normalized_dict["metadata"]
        item_id = meta_dict.get("source_id") or (
            f"ext_{uuid.uuid4().hex[:10]}" if not self.source_definition else None
        )

        prov = self._build_provenance(
            item_id=item_id,
            explicit_provenance=meta_dict.get("provenance"),
            source_type=self.source_definition.source_type if self.source_definition else meta_dict.get("source_type", SourceType.EXISTING_DATASET.value),
            source=self.source_definition.name if self.source_definition else meta_dict.get("source", "existing_dataset"),
            license=meta_dict.get("license"),
            created_at=meta_dict.get("created_at"),
        )

        metadata = RecordMetadata(
            domain=meta_dict.get("domain", "general_knowledge"),
            topic=meta_dict.get("topic", "general"),
            task_type=meta_dict.get("task_type", "explanation"),
            difficulty=meta_dict.get("difficulty", "intermediate"),
            quality_score=meta_dict.get("quality_score"),
            source=prov.source,
            source_type=prov.source_type,
            created_at=prov.created_at,
            source_id=prov.source_id,
            license=prov.license,
            generator=prov.generator,
            generator_version=prov.generator_version,
            provenance=prov,
        )

        return DatasetRecord(messages=messages, metadata=metadata)


class DocumentationAdapter(SourceAdapter):
    """Adapts technical documentation, reference manuals, or markdown guides into conversational QA pairs."""

    def adapt(self, raw_item: Union[RawRecord, Dict[str, Any]]) -> DatasetRecord:
        data = raw_item.data if isinstance(raw_item, RawRecord) else raw_item

        question = data.get("question") or f"Explain the technical concepts behind {data.get('title', 'this topic')}."
        answer = data.get("answer") or data.get("content") or ""

        messages = [
            Message(role=Role.USER, content=self.normalizer.normalize_text(question)),
            Message(role=Role.ASSISTANT, content=self.normalizer.normalize_text(answer)),
        ]

        item_id = data.get("source_id") or data.get("doc_id")
        prov = self._build_provenance(
            item_id=item_id,
            explicit_provenance=data.get("provenance"),
            source_type=self.source_definition.source_type if self.source_definition else SourceType.DOCUMENTATION.value,
            source=self.source_definition.name if self.source_definition else data.get("source", "technical_documentation"),
            license=data.get("license"),
            created_at=data.get("created_at"),
            source_url=data.get("url"),
        )

        metadata = RecordMetadata(
            domain=data.get("domain", "linux_systems"),
            topic=data.get("topic", "documentation"),
            task_type=data.get("task_type", "explanation"),
            difficulty=data.get("difficulty", "intermediate"),
            quality_score=data.get("quality_score", 0.90),
            source=prov.source,
            source_type=prov.source_type,
            created_at=prov.created_at,
            source_id=prov.source_id,
            license=prov.license,
            provenance=prov,
        )

        return DatasetRecord(messages=messages, metadata=metadata)


class HumanAuthoredAdapter(SourceAdapter):
    """Adapts internally curated or human-written technical problems and solutions."""

    def adapt(self, raw_item: Union[RawRecord, Dict[str, Any]]) -> DatasetRecord:
        data = raw_item.data if isinstance(raw_item, RawRecord) else raw_item

        if "messages" in data:
            messages = [
                Message(
                    role=Role(self.normalizer.normalize_role(m["role"])),
                    content=self.normalizer.normalize_text(m["content"]),
                )
                for m in data["messages"]
            ]
        else:
            prompt = data.get("prompt") or data.get("instruction") or ""
            response = data.get("response") or data.get("output") or ""
            messages = [
                Message(role=Role.USER, content=self.normalizer.normalize_text(prompt)),
                Message(role=Role.ASSISTANT, content=self.normalizer.normalize_text(response)),
            ]

        meta_dict = data.get("metadata", data)
        item_id = meta_dict.get("source_id")
        prov = self._build_provenance(
            item_id=item_id,
            explicit_provenance=meta_dict.get("provenance"),
            source_type=self.source_definition.source_type if self.source_definition else meta_dict.get("source_type", SourceType.HUMAN_AUTHORED.value),
            source=self.source_definition.name if self.source_definition else meta_dict.get("source", "human_authored_internal"),
            license=meta_dict.get("license"),
            created_at=meta_dict.get("created_at"),
        )

        metadata = RecordMetadata(
            domain=meta_dict.get("domain", "programming"),
            topic=meta_dict.get("topic", "python"),
            task_type=meta_dict.get("task_type", "coding"),
            difficulty=meta_dict.get("difficulty", "intermediate"),
            quality_score=meta_dict.get("quality_score", 0.95),
            source=prov.source,
            source_type=prov.source_type,
            created_at=prov.created_at,
            source_id=prov.source_id,
            license=prov.license,
            provenance=prov,
        )

        return DatasetRecord(messages=messages, metadata=metadata)


class SyntheticAdapter(SourceAdapter):
    """Adapts synthetic generator output records, embedding generator model and version provenance."""

    def adapt(self, raw_item: Union[RawRecord, Dict[str, Any]]) -> DatasetRecord:
        data = raw_item.data if isinstance(raw_item, RawRecord) else raw_item

        if "messages" in data:
            messages = [
                Message(
                    role=Role(self.normalizer.normalize_role(m["role"])),
                    content=self.normalizer.normalize_text(m["content"]),
                )
                for m in data["messages"]
            ]
        else:
            prompt = data.get("prompt", "")
            response = data.get("response", "")
            messages = [
                Message(role=Role.USER, content=self.normalizer.normalize_text(prompt)),
                Message(role=Role.ASSISTANT, content=self.normalizer.normalize_text(response)),
            ]

        meta_dict = data.get("metadata", data)
        item_id = meta_dict.get("source_id") or f"synth_{uuid.uuid4().hex[:10]}"
        prov = self._build_provenance(
            item_id=item_id,
            explicit_provenance=meta_dict.get("provenance"),
            source_type=self.source_definition.source_type if self.source_definition else meta_dict.get("source_type", SourceType.SYNTHETIC.value),
            source=self.source_definition.name if self.source_definition else meta_dict.get("source", "synthetic_generator"),
            generator=meta_dict.get("generator") or (self.source_definition.generator if self.source_definition else None),
            generator_version=meta_dict.get("generator_version") or (self.source_definition.generator_version if self.source_definition else None),
            license=meta_dict.get("license"),
            created_at=meta_dict.get("created_at"),
        )

        metadata = RecordMetadata(
            domain=meta_dict.get("domain", "programming"),
            topic=meta_dict.get("topic", "python"),
            task_type=meta_dict.get("task_type", "coding"),
            difficulty=meta_dict.get("difficulty", "intermediate"),
            quality_score=meta_dict.get("quality_score", 0.90),
            source=prov.source,
            source_type=prov.source_type,
            created_at=prov.created_at,
            source_id=prov.source_id,
            license=prov.license,
            generator=prov.generator,
            generator_version=prov.generator_version,
            provenance=prov,
        )

        return DatasetRecord(messages=messages, metadata=metadata)


def create_source_adapter(
    source_definition: SourceDefinition, normalizer: Optional[DatasetNormalizer] = None
) -> SourceAdapter:
    """Factory creating the appropriate SourceAdapter for a given SourceDefinition."""
    stype = source_definition.source_type.lower()

    if stype in (SourceType.EXISTING_DATASET.value, SourceType.CURATED.value, SourceType.RAW.value, SourceType.BENCHMARK.value):
        return ExistingDatasetAdapter(source_definition=source_definition, normalizer=normalizer)
    elif stype == SourceType.DOCUMENTATION.value:
        return DocumentationAdapter(source_definition=source_definition, normalizer=normalizer)
    elif stype in (SourceType.HUMAN_AUTHORED.value, SourceType.INTERNAL.value):
        return HumanAuthoredAdapter(source_definition=source_definition, normalizer=normalizer)
    elif stype == SourceType.SYNTHETIC.value:
        return SyntheticAdapter(source_definition=source_definition, normalizer=normalizer)
    else:
        return ExistingDatasetAdapter(source_definition=source_definition, normalizer=normalizer)
