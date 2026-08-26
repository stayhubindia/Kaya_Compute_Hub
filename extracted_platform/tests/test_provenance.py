from datetime import datetime, timezone
import pytest

from src.dataset.schema import (
    DatasetRecord,
    Message,
    ProvenanceInfo,
    RecordMetadata,
    Role,
    SourceType,
)


def test_valid_provenance_explicit():
    prov = ProvenanceInfo(
        source_type=SourceType.HUMAN_AUTHORED,
        source="Internal Engineering Notes",
        source_id="internal-note-001",
        license="Internal-Confidential",
        created_at="2026-08-11T12:00:00Z",
    )
    meta = RecordMetadata(
        domain="programming",
        topic="python",
        task_type="coding",
        difficulty="intermediate",
        quality_score=0.95,
        provenance=prov,
    )
    record = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Explain context managers in Python."),
            Message(role=Role.ASSISTANT, content="Context managers facilitate resource management using __enter__ and __exit__."),
        ],
        metadata=meta,
    )

    d = record.to_dict()
    assert d["metadata"]["provenance"]["source_type"] == "human_authored"
    assert d["metadata"]["provenance"]["source"] == "Internal Engineering Notes"
    assert d["metadata"]["provenance"]["source_id"] == "internal-note-001"
    assert d["metadata"]["provenance"]["license"] == "Internal-Confidential"

    # Backwards compatibility top-level checks
    assert d["metadata"]["source"] == "Internal Engineering Notes"
    assert d["metadata"]["source_type"] == "human_authored"


def test_synthetic_provenance():
    prov = ProvenanceInfo(
        source_type=SourceType.SYNTHETIC,
        source="synthetic_pipeline",
        source_id="batch-gen-004",
        generator="sample_test_generator",
        generator_version="1.0.0",
        created_at="2026-08-11T12:00:00Z",
    )
    assert prov.source_type == "synthetic"
    assert prov.generator == "sample_test_generator"
    assert prov.generator_version == "1.0.0"
    assert prov.license is None


def test_unknown_provenance_defaults():
    meta = RecordMetadata(
        domain="general_knowledge",
        topic="general",
        task_type="explanation",
        difficulty="beginner",
    )
    assert meta.provenance is not None
    assert meta.provenance.source_type == "unknown"
    assert meta.provenance.source == "unknown"
    assert meta.provenance.source_id is None
    assert meta.provenance.license is None


def test_missing_optional_fields_serialization():
    prov = ProvenanceInfo(
        source_type="documentation",
        source="Linux Docs",
    )
    d = prov.to_dict()
    assert d["source_type"] == "documentation"
    assert d["source"] == "Linux Docs"
    assert d["source_id"] is None
    assert d["license"] is None
    assert d["generator"] is None
