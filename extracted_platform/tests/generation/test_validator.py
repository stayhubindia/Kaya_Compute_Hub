"""
Unit tests for InstructionValidator (src/generation/validator.py).
"""

import pytest
from src.dataset.schema import DatasetRecord, Message, RecordMetadata, Role, TaskType
from src.generation.models import KnowledgeUnit
from src.generation.validator import InstructionValidator


def test_validator_grounded_example():
    validator = InstructionValidator(min_grounding_score=0.20, min_lexical_overlap=0.20)
    unit = KnowledgeUnit(
        unit_id="ku_1",
        document_id="doc_1",
        section_id="sec_1",
        text="The photoelectric effect demonstrates the particle nature of light.",
        domain="physics",
        topic="quantum_mechanics",
    )

    rec = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="What is the photoelectric effect?"),
            Message(role=Role.ASSISTANT, content="The photoelectric effect demonstrates the particle nature of light when photons strike a metal."),
        ],
        metadata=RecordMetadata(
            domain="physics",
            topic="quantum_mechanics",
            task_type="explanation",
            difficulty="intermediate",
        ),
    )

    grounding, math_eval, rejections = validator.validate_candidate(rec, unit)
    assert grounding.is_grounded is True
    assert grounding.lexical_overlap >= 0.20
    assert len(rejections) == 0


def test_validator_ungrounded_example():
    validator = InstructionValidator(min_grounding_score=0.50, min_lexical_overlap=0.50)
    unit = KnowledgeUnit(
        unit_id="ku_2",
        document_id="doc_1",
        section_id="sec_1",
        text="Thermodynamic cycles like Carnot cycle operate between reservoirs.",
        domain="physics",
        topic="thermodynamics",
    )

    rec = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Explain baking cookies."),
            Message(role=Role.ASSISTANT, content="Baking cookies requires flour, sugar, butter, and vanilla extract."),
        ],
        metadata=RecordMetadata(
            domain="physics",
            topic="thermodynamics",
            task_type="explanation",
            difficulty="intermediate",
        ),
    )

    grounding, math_eval, rejections = validator.validate_candidate(rec, unit)
    assert grounding.is_grounded is False
    assert any("Insufficient lexical grounding overlap" in r for r in rejections)


def test_validator_unbalanced_brackets():
    validator = InstructionValidator()
    unit = KnowledgeUnit(
        unit_id="ku_3",
        document_id="doc_1",
        section_id="sec_1",
        text="Equation derivation with unbalanced braces.",
        domain="physics",
        topic="quantum_mechanics",
    )

    rec = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Derive equation."),
            Message(role=Role.ASSISTANT, content="Here is derivation: $$\\frac{a}{b$$ with missing closing brace."),
        ],
        metadata=RecordMetadata(
            domain="physics",
            topic="quantum_mechanics",
            task_type=TaskType.PROOF.value,
            difficulty="advanced",
        ),
    )

    grounding, math_eval, rejections = validator.validate_candidate(rec, unit)
    assert math_eval.is_valid is False
    assert math_eval.balanced_delimiters is False
    assert any("Unbalanced mathematical delimiters" in r for r in rejections)
