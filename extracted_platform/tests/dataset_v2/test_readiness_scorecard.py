"""
Tests for 10-Dimension Readiness Scorecard & Release QA Engine (Phase 3.5).
"""

import pytest
from src.dataset.release_qa import DatasetReleaseQAEngine, GateStatus, ReleaseLifecycleState
from src.dataset.schema import DatasetRecord, Message, ProvenanceInfo, RecordMetadata, Role, SourceType


def make_valid_record(idx: int, domain: str = "science", difficulty: str = "intermediate") -> DatasetRecord:
    return DatasetRecord(
        messages=[
            Message(role=Role.USER, content=f"What is the principle of conservation of energy {idx}?"),
            Message(
                role=Role.ASSISTANT,
                content=f"Energy cannot be created or destroyed, only transformed in system {idx}. $$E_{{total}} = E_k + E_p$$. Units: J.",
            ),
        ],
        metadata=RecordMetadata(
            domain=domain,
            topic="physics",
            difficulty=difficulty,
            task_type="conceptual_explanation",
            source_type=SourceType.DOCUMENTATION.value,
            source="nptel_physics",
            provenance=ProvenanceInfo(
                source_type=SourceType.DOCUMENTATION.value,
                source="nptel_physics",
                source_id=f"doc_{idx}",
                license="CC-BY-4.0",
            ),
        ),
    )


def test_scorecard_evaluation_pass():
    engine = DatasetReleaseQAEngine()
    records = [
        make_valid_record(i, domain="science", difficulty="intermediate")
        for i in range(10)
    ]

    report, train, val, test = engine.run_qa_pipeline(
        input_source=records,
        target_size=10,
        dry_run=True,
    )

    assert len(report.scorecard) == 10
    assert report.all_mandatory_gates_passed is True
    assert report.lifecycle_state == ReleaseLifecycleState.READY

    dimensions = {sc.dimension: sc for sc in report.scorecard}
    assert dimensions["Schema Validity"].status == GateStatus.PASS
    assert dimensions["Provenance Completeness"].status == GateStatus.PASS
    assert dimensions["Scientific Correctness"].status == GateStatus.PASS
    assert dimensions["Split Integrity"].status == GateStatus.PASS


def test_scorecard_evaluation_critical_failure():
    engine = DatasetReleaseQAEngine()
    # Malformed record with broken brackets and placeholder citation
    broken_rec = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Explain quantum optics."),
            Message(role=Role.ASSISTANT, content="Formula: $$E = (h \\nu$$ [Citation Needed]"),
        ],
        metadata=RecordMetadata(
            domain="science",
            topic="quantum_optics",
            difficulty="advanced",
            task_type="conceptual_explanation",
            source_type=SourceType.DOCUMENTATION.value,
            source="nptel_physics",
            provenance=ProvenanceInfo(
                source_type=SourceType.DOCUMENTATION.value,
                source="nptel_physics",
                source_id="doc_broken",
                license="CC-BY-4.0",
            ),
        ),
    )

    report, _, _, _ = engine.run_qa_pipeline(
        input_source=[broken_rec],
        target_size=1,
        dry_run=True,
    )

    assert report.all_mandatory_gates_passed is False
    assert report.lifecycle_state == ReleaseLifecycleState.REJECTED
