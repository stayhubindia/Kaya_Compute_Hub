"""
Unit tests for ScientificPromptBuilder (src/generation/prompt_builder.py).
"""

import pytest
from src.dataset.schema import TaskType
from src.generation.models import ContentType, KnowledgeUnit
from src.generation.prompt_builder import ScientificPromptBuilder


def test_prompt_builder_qa():
    pb = ScientificPromptBuilder()
    unit = KnowledgeUnit(
        unit_id="u1",
        document_id="d1",
        section_id="s1",
        title="Schrodinger Equation",
        topic="quantum_mechanics",
        domain="physics",
        text="Quantum mechanics equation.",
        content_types=[ContentType.DEFINITION],
    )

    prompt = pb.build_prompt(unit, TaskType.QUESTION_ANSWERING.value)
    assert "Schrodinger Equation" in prompt
    assert "Quantum Mechanics" in prompt


def test_prompt_builder_derivation():
    pb = ScientificPromptBuilder()
    unit = KnowledgeUnit(
        unit_id="u2",
        document_id="d1",
        section_id="s1",
        title="Diffraction Limit",
        topic="optics",
        domain="physics",
        text="Deriving the diffraction limit.",
    )

    prompt = pb.build_prompt(unit, TaskType.PROOF.value)
    assert "Derive" in prompt
    assert "Diffraction Limit" in prompt


def test_prompt_builder_calculation():
    pb = ScientificPromptBuilder()
    unit = KnowledgeUnit(
        unit_id="u3",
        document_id="d1",
        section_id="s1",
        title="Binding Energy",
        topic="nuclear_physics",
        domain="physics",
        text="Calculate binding energy.",
    )

    prompt = pb.build_prompt(unit, TaskType.CALCULATION.value)
    assert "Calculate" in prompt
