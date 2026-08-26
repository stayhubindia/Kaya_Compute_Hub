"""
Tests for Cross-Split and Source-Group Leakage Guard (Phase 3.5).
"""

import pytest
from src.dataset.leakage_guard import LeakageGuard
from src.dataset.schema import DatasetRecord, Message, ProvenanceInfo, RecordMetadata, Role, SourceType


def make_record_with_source(doc_id: str, text: str, domain: str = "science") -> DatasetRecord:
    return DatasetRecord(
        messages=[
            Message(role=Role.USER, content=text),
            Message(role=Role.ASSISTANT, content=f"Answer for {text}"),
        ],
        metadata=RecordMetadata(
            domain=domain,
            topic="physics",
            difficulty="intermediate",
            task_type="conceptual_explanation",
            source_type=SourceType.DOCUMENTATION.value,
            source="nptel_doc",
            provenance=ProvenanceInfo(
                source_type=SourceType.DOCUMENTATION.value,
                source="nptel_doc",
                source_id=f"{doc_id}::chunk_001",
                license="CC-BY-4.0",
            ),
        ),
    )


def test_source_group_clustering():
    guard = LeakageGuard(seed=42)
    records = [
        make_record_with_source("docA", "Prompt A1"),
        make_record_with_source("docA", "Prompt A2"),
        make_record_with_source("docB", "Prompt B1"),
        make_record_with_source("docB", "Prompt B2"),
        make_record_with_source("docC", "Prompt C1"),
        make_record_with_source("docD", "Prompt D1"),
    ]

    train, val, test = guard.split_with_source_group_isolation(
        records, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, cluster_by_source=True
    )

    assert len(train) + len(val) + len(test) == len(records)

    # Check that docA records are in the same split
    train_sources = {guard._extract_source_group_key(r) for r in train}
    val_sources = {guard._extract_source_group_key(r) for r in val}
    test_sources = {guard._extract_source_group_key(r) for r in test}

    assert len(train_sources.intersection(val_sources)) == 0
    assert len(train_sources.intersection(test_sources)) == 0
    assert len(val_sources.intersection(test_sources)) == 0


def test_cross_split_leakage_detection():
    guard = LeakageGuard()
    r1 = make_record_with_source("docA", "Unique Prompt 1")
    r2 = make_record_with_source("docB", "Unique Prompt 2")
    r_dup = make_record_with_source("docA", "Unique Prompt 1")  # exact duplicate

    train = [r1]
    val = [r_dup]
    test = [r2]

    report = guard.audit_cross_split_leakage(train, val, test)
    assert report.is_clean is False
    assert report.train_val_exact == 1
    assert report.total_exact_leaks == 1
