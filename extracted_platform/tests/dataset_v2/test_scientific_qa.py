"""
Tests for Scientific Quality and Rigor Auditor (Phase 3.5).
"""

import pytest
from src.dataset.scientific_qa import ScientificQAAuditor, ScientificValidationStatus
from src.dataset.schema import DatasetRecord, Message, ProvenanceInfo, RecordMetadata, Role, SourceType, TaskType


def make_sci_record(
    prompt: str,
    response: str,
    task_type: str = "conceptual_explanation",
    domain: str = "science",
) -> DatasetRecord:
    return DatasetRecord(
        messages=[
            Message(role=Role.USER, content=prompt),
            Message(role=Role.ASSISTANT, content=response),
        ],
        metadata=RecordMetadata(
            domain=domain,
            topic="classical_mechanics",
            difficulty="advanced",
            task_type=task_type,
            source_type=SourceType.DOCUMENTATION.value,
            source="nptel_physics",
            provenance=ProvenanceInfo(
                source_type=SourceType.DOCUMENTATION.value,
                source="nptel_physics",
                source_id="phys_101",
                license="CC-BY-4.0",
            ),
        ),
    )


def test_scientific_qa_valid_equation_and_units():
    auditor = ScientificQAAuditor()
    rec = make_sci_record(
        prompt="Derive the kinetic energy equation for a particle with mass $m$ and velocity $v$.",
        response="The kinetic energy $E_k$ is defined as $$E_k = \\frac{1}{2} m v^2$$. When $v = 10\\text{ m/s}$ and $m = 2\\text{ kg}$, $E_k = 100\\text{ J}$.",
        task_type=TaskType.CALCULATION.value,
    )
    res = auditor.audit_record(rec)
    assert res.is_valid is True
    assert res.status == ScientificValidationStatus.VERIFIED
    assert res.equations_count >= 1
    assert res.balanced_delimiters is True
    assert "m/s" in res.units_detected or "J" in res.units_detected


def test_scientific_qa_unbalanced_brackets():
    auditor = ScientificQAAuditor()
    rec = make_sci_record(
        prompt="Explain energy-momentum relation.",
        response="The formula is $$E^2 = (p c)^2 + (m_0 c^2)^2$ with unmatched brackets: $$E = \\frac{1}{2} (m v^2$$",
        task_type="derivation",
    )
    res = auditor.audit_record(rec)
    assert res.is_valid is False
    assert res.status == ScientificValidationStatus.FAILED


def test_scientific_qa_placeholder_citations():
    auditor = ScientificQAAuditor()
    rec = make_sci_record(
        prompt="What is general relativity?",
        response="General relativity was published by Einstein [Citation Needed].",
        task_type="conceptual_explanation",
    )
    res = auditor.audit_record(rec)
    assert res.is_valid is False
    assert res.status == ScientificValidationStatus.FAILED


def test_audit_dataset_aggregation():
    auditor = ScientificQAAuditor()
    records = [
        make_sci_record(
            "Calculate force for $m=5\\text{ kg}$ and $a=2\\text{ m/s^2}$.",
            "According to Newton's second law, $$F = m a$$. For $m=5\\text{ kg}$ and $a=2\\text{ m/s^2}$, $F = 10\\text{ N}$.",
            task_type=TaskType.CALCULATION.value,
        ),
        make_sci_record(
            "Broken citation check.",
            "This statement has no proof [?].",
        ),
    ]

    passed, failed, result, evals = auditor.audit_dataset(records)
    assert len(passed) == 1
    assert len(failed) == 1
    assert result.total_evaluated == 2
    assert result.verified_count == 1
    assert result.failed_count == 1
