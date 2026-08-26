"""
Unit tests for InstructionQualityAuditor (src/generation/quality.py).
"""

import pytest
from src.dataset.schema import DatasetRecord, Message, RecordMetadata, Role
from src.generation.models import GroundingEvaluation, MathematicalValidation
from src.generation.quality import InstructionQualityAuditor


def test_quality_auditor_high_quality():
    auditor = InstructionQualityAuditor(min_quality_score=0.85)
    rec = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Explain the physical significance of Planck's constant in quantum mechanics."),
            Message(
                role=Role.ASSISTANT,
                content=(
                    "### Physical Significance of Planck's Constant\n\n"
                    "Planck's constant (denoted by $h$) sets the quantum scale of action. "
                    "It relates photon energy to frequency via $E = h\\nu$, defining the granularity of atomic transitions."
                ),
            ),
        ],
        metadata=RecordMetadata(
            domain="physics",
            topic="quantum_mechanics",
            task_type="explanation",
            difficulty="intermediate",
        ),
    )

    grounding = GroundingEvaluation(is_grounded=True, grounding_score=0.95)
    math_eval = MathematicalValidation(is_valid=True)

    score, dims, feedback = auditor.audit_candidate(rec, grounding, math_eval)
    assert score >= 0.85
    assert len(feedback) == 0


def test_quality_auditor_broken_unicode():
    auditor = InstructionQualityAuditor(min_quality_score=0.85)
    rec = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Explain the magnetic field."),
            Message(role=Role.ASSISTANT, content="The field is \ufffd with corrupt character encoding here."),
        ],
        metadata=RecordMetadata(
            domain="physics",
            topic="electromagnetism",
            task_type="explanation",
            difficulty="intermediate",
        ),
    )

    grounding = GroundingEvaluation(is_grounded=True, grounding_score=0.90)
    math_eval = MathematicalValidation(is_valid=True)

    score, dims, feedback = auditor.audit_candidate(rec, grounding, math_eval)
    assert any("broken Unicode" in f for f in feedback)
