"""
Integration tests for InstructionDatasetPipeline (src/generation/pipeline.py).
"""

import json
from pathlib import Path
import pytest

from src.dataset.schema import SourceType
from src.generation.models import KnowledgeUnit
from src.generation.pipeline import InstructionDatasetPipeline
from src.ingestion.models import Equation, KnowledgeChunk, ProvenanceInfo


@pytest.fixture
def sample_units():
    prov1 = ProvenanceInfo(source_type=SourceType.DOCUMENTATION.value, source="nptel", source_id="chunk_01")
    u1 = KnowledgeUnit(
        unit_id="ku_physics_01",
        document_id="doc_nptel_01",
        section_id="sec_01",
        chunk_id="chunk_01",
        title="Schrodinger Wave Mechanics",
        domain="physics",
        topic="quantum_mechanics",
        source="nptel",
        text=(
            "The time-dependent Schrodinger equation governs quantum state evolution. "
            "For a stationary state, substituting the spatial wave function yields the time-independent equation "
            "where energy eigenvalues are real."
        ),
        equations=[Equation(equation_id="eq1", latex_content=r"\hat{H}\psi = E\psi")],
    )

    prov2 = ProvenanceInfo(source_type=SourceType.DOCUMENTATION.value, source="arxiv", source_id="chunk_02")
    u2 = KnowledgeUnit(
        unit_id="ku_physics_02",
        document_id="doc_arxiv_01",
        section_id="sec_02",
        chunk_id="chunk_02",
        title="Optical Cavity Resonance",
        domain="physics",
        topic="optics",
        source="arxiv",
        text=(
            "We analyze optical cavity resonance in Fabry-Perot interferometers. "
            "The resonance frequency is given by f_m = m c / (2 L). "
            "Experimental measurements demonstrate sharp transmission peaks with finesse F > 100."
        ),
        equations=[Equation(equation_id="eq2", latex_content=r"f_m = \frac{m c}{2 L}")],
    )
    return [u1, u2]


def test_pipeline_dry_run(sample_units, tmp_path: Path):
    pipeline = InstructionDatasetPipeline()
    res = pipeline.run(input_dir_or_units=sample_units, output_dir=tmp_path, dry_run=True)

    assert res["dry_run"] is True
    assert res["total_units_evaluated"] == 2
    assert res["total_units_selected"] == 2


def test_pipeline_generation(sample_units, tmp_path: Path):
    pipeline = InstructionDatasetPipeline(max_examples_per_unit=2)
    summary = pipeline.run(input_dir_or_units=sample_units, output_dir=tmp_path, dry_run=False)

    assert summary["total_candidates_generated"] >= 2
    assert summary["total_candidates_accepted"] >= 2
    assert summary["average_quality_score"] >= 0.85

    # Check output files
    combined_file = tmp_path / "combined_candidates.jsonl"
    assert combined_file.is_file()
    assert combined_file.stat().st_size > 0

    # Check reports
    reports_dir = tmp_path / "reports"
    assert (reports_dir / "statistics.json").is_file()
    assert (reports_dir / "generation_report.md").is_file()
    assert (reports_dir / "quality_report.json").is_file()
    assert (reports_dir / "provenance_report.json").is_file()
    assert (reports_dir / "rejection_report.json").is_file()

    # Check manifest
    manifest_file = tmp_path / "manifests" / "generation_manifest.json"
    assert manifest_file.is_file()
