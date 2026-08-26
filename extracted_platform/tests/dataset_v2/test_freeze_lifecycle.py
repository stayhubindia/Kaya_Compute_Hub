"""
Tests for Dataset Freeze Lifecycle & Cryptographic Hashing (Phase 3.5).
"""

import hashlib
import json
import pytest
from src.dataset.release_qa import DatasetReleaseQAEngine, ReleaseLifecycleState
from src.dataset.schema import DatasetRecord, Message, ProvenanceInfo, RecordMetadata, Role, SourceType


def make_valid_record(idx: int) -> DatasetRecord:
    return DatasetRecord(
        messages=[
            Message(role=Role.USER, content=f"Scientific query number {idx} regarding thermodynamics."),
            Message(
                role=Role.ASSISTANT,
                content=f"Thermodynamic explanation {idx}. Heat transferred: $Q = m c \\Delta T$. Units: J.",
            ),
        ],
        metadata=RecordMetadata(
            domain="science",
            topic="thermodynamics",
            difficulty="intermediate",
            task_type="conceptual_explanation",
            source_type=SourceType.DOCUMENTATION.value,
            source="nptel_thermo",
            provenance=ProvenanceInfo(
                source_type=SourceType.DOCUMENTATION.value,
                source="nptel_thermo",
                source_id=f"doc_thermo_{idx}",
                license="CC-BY-4.0",
            ),
        ),
    )


def test_freeze_lifecycle_execution(tmp_path):
    release_dir = tmp_path / "dataset-v2.0"
    engine = DatasetReleaseQAEngine()
    records = [make_valid_record(i) for i in range(12)]

    report, train, val, test = engine.run_qa_pipeline(
        input_source=records,
        target_size=12,
        output_dir=release_dir,
        dry_run=False,
        freeze=True,
    )

    assert report.is_frozen is True
    assert report.lifecycle_state == ReleaseLifecycleState.FROZEN
    assert (release_dir / "manifest.json").exists()
    assert (release_dir / "checksums.sha256").exists()
    assert (release_dir / "train.jsonl").exists()
    assert (release_dir / "validation.jsonl").exists()
    assert (release_dir / "test.jsonl").exists()
    assert (release_dir / "reports" / "dataset_v2_qa.json").exists()
    assert (release_dir / "reports" / "dataset_v2_qa.md").exists()

    # Verify checksums integrity
    with open(release_dir / "checksums.sha256", "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 4
    for line in lines:
        parts = line.strip().split()
        assert len(parts) == 2
        chk, fname = parts
        fpath = release_dir / fname
        if fpath.exists():
            with open(fpath, "rb") as fl:
                computed = hashlib.sha256(fl.read()).hexdigest()
            assert chk == computed
