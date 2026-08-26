"""
Unit tests for specialized scientific generators and dispatcher (src/generation/).
"""

import pytest
from src.dataset.schema import Role, TaskType
from src.generation.answer_generator import ScientificInstructionDispatcher
from src.generation.equation_generator import EquationGenerator
from src.generation.models import ContentType, KnowledgeUnit
from src.generation.multi_turn_generator import MultiTurnGenerator
from src.generation.problem_generator import ProblemGenerator
from src.generation.reasoning_generator import ReasoningGenerator
from src.generation.scientific_generator import ScientificGenerator
from src.ingestion.models import Equation


@pytest.fixture
def sample_unit():
    return KnowledgeUnit(
        unit_id="ku_test_01",
        document_id="doc_test",
        section_id="sec_test",
        title="Harmonic Oscillator",
        domain="physics",
        topic="quantum_mechanics",
        subtopic="potential_wells",
        source="nptel",
        text=(
            "The quantum harmonic oscillator is a fundamental model system. "
            "The potential energy is given by V(x) = (1/2)m\\omega^2 x^2. "
            "Energy eigenvalues are quantized as E_n = (n + 1/2)\\hbar\\omega."
        ),
        equations=[
            Equation(equation_id="eq1", latex_content=r"E_n = \left(n + \frac{1}{2}\right)\hbar\omega")
        ],
    )


def test_scientific_generator(sample_unit):
    gen = ScientificGenerator()
    rec = gen.generate_candidate(sample_unit, TaskType.QUESTION_ANSWERING.value, "What is a harmonic oscillator?")
    assert len(rec.messages) == 2
    assert rec.messages[0].role == Role.USER
    assert rec.messages[1].role == Role.ASSISTANT
    assert "Harmonic Oscillator" in rec.messages[1].content
    assert rec.metadata.domain == "physics"


def test_equation_generator(sample_unit):
    gen = EquationGenerator()
    rec = gen.generate_candidate(sample_unit, TaskType.PROOF.value, "Derive energy eigenvalues.")
    assert len(rec.messages) == 2
    assert r"E_n" in rec.messages[1].content
    assert "$$" in rec.messages[1].content


def test_problem_generator(sample_unit):
    gen = ProblemGenerator()
    rec = gen.generate_candidate(sample_unit, TaskType.CALCULATION.value, "Calculate ground state energy.")
    assert len(rec.messages) == 2
    assert "Problem Solution" in rec.messages[1].content


def test_reasoning_generator(sample_unit):
    gen = ReasoningGenerator()
    rec = gen.generate_candidate(sample_unit, TaskType.REASONING.value, "Explain the quantization condition.")
    assert len(rec.messages) == 2
    assert "Scientific Reasoning Chain" in rec.messages[1].content


def test_multi_turn_generator(sample_unit):
    gen = MultiTurnGenerator()
    rec = gen.generate_candidate(sample_unit, TaskType.MULTI_TURN.value, "Can you explain the harmonic oscillator?")
    assert len(rec.messages) >= 4
    assert rec.messages[0].role == Role.USER
    assert rec.messages[1].role == Role.ASSISTANT
    assert rec.messages[2].role == Role.USER
    assert rec.messages[3].role == Role.ASSISTANT


def test_dispatcher(sample_unit):
    dispatcher = ScientificInstructionDispatcher()

    # Proof -> EquationGenerator
    rec_proof = dispatcher.dispatch_and_generate(sample_unit, TaskType.PROOF.value, "Derive...")
    assert "Mathematical Derivation" in rec_proof.messages[1].content

    # Calculation -> ProblemGenerator
    rec_calc = dispatcher.dispatch_and_generate(sample_unit, TaskType.CALCULATION.value, "Calculate...")
    assert "Problem Solution" in rec_calc.messages[1].content

    # Multi-turn -> MultiTurnGenerator
    rec_mt = dispatcher.dispatch_and_generate(sample_unit, TaskType.MULTI_TURN.value, "Explain...")
    assert len(rec_mt.messages) >= 4
