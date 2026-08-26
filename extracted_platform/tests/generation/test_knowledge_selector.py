"""
Unit tests for KnowledgeSelector (src/generation/knowledge_selector.py).
"""

import pytest
from src.dataset.schema import DifficultyLevel, SourceType
from src.generation.knowledge_selector import KnowledgeSelector
from src.generation.models import ContentType, KnowledgeUnit
from src.ingestion.models import Equation, KnowledgeChunk, ProvenanceInfo, Table


def test_knowledge_selector_definition():
    selector = KnowledgeSelector(min_token_estimate=5)
    unit = KnowledgeUnit(
        unit_id="ku_1",
        document_id="doc_1",
        section_id="sec_1",
        text="The refractive index is defined as the ratio of the speed of light in vacuum to that in the medium.",
        domain="physics",
        topic="optics",
    )

    enriched = selector.analyze_and_enrich_unit(unit)
    assert ContentType.DEFINITION in enriched.content_types
    assert enriched.difficulty_estimate in [DifficultyLevel.BEGINNER.value, DifficultyLevel.INTERMEDIATE.value]


def test_knowledge_selector_derivation():
    selector = KnowledgeSelector(min_token_estimate=5)
    unit = KnowledgeUnit(
        unit_id="ku_2",
        document_id="doc_1",
        section_id="sec_1",
        text="We derive the wave equation by substituting equation (1) into the Maxwell equations. Differentiating with respect to time yields $$\\nabla^2 E = \\frac{1}{c^2}\\frac{\\partial^2 E}{\\partial t^2}$$.",
        domain="physics",
        topic="electromagnetism",
        equations=[
            Equation(equation_id="eq_1", latex_content=r"\nabla^2 E = \frac{1}{c^2}\frac{\partial^2 E}{\partial t^2}")
        ],
    )

    enriched = selector.analyze_and_enrich_unit(unit)
    assert ContentType.DERIVATION in enriched.content_types
    assert ContentType.EQUATION in enriched.content_types
    assert enriched.mathematical_density > 0.05
    assert enriched.difficulty_estimate in [DifficultyLevel.ADVANCED.value, DifficultyLevel.EXPERT.value]


def test_knowledge_selector_calculation():
    selector = KnowledgeSelector(min_token_estimate=5)
    unit = KnowledgeUnit(
        unit_id="ku_3",
        document_id="doc_1",
        section_id="sec_1",
        text="Calculate the photon energy when frequency is 5.0 MHz. The velocity is 3.0e8 m/s.",
        domain="physics",
        topic="quantum_physics",
    )

    enriched = selector.analyze_and_enrich_unit(unit)
    assert ContentType.CALCULATION in enriched.content_types


def test_knowledge_selector_table():
    selector = KnowledgeSelector(min_token_estimate=5)
    unit = KnowledgeUnit(
        unit_id="ku_4",
        document_id="doc_1",
        section_id="sec_1",
        text="The measured values are shown below:\n| Wavelength | Frequency |\n|---|---|\n| 500nm | 6.0e14Hz |",
        domain="physics",
        topic="optics",
        tables=[Table(table_id="tbl_1", headers=["Wavelength", "Frequency"])],
    )

    enriched = selector.analyze_and_enrich_unit(unit)
    assert ContentType.TABLE_DATA in enriched.content_types


def test_select_units_filter():
    selector = KnowledgeSelector(min_token_estimate=10)
    u_short = KnowledgeUnit(
        unit_id="ku_short", document_id="d1", section_id="s1", text="Too short."
    )
    u_valid = KnowledgeUnit(
        unit_id="ku_valid",
        document_id="d1",
        section_id="s1",
        text="This is a comprehensive scientific description containing enough words to pass the minimum token check cleanly.",
    )

    selected = selector.select_units([u_short, u_valid])
    assert len(selected) == 1
    assert selected[0].unit_id == "ku_valid"
