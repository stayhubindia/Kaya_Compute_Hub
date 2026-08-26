"""
Unit tests for Ingestion Data Models and IR (Phase 3.3).
"""

import pytest
from src.dataset.schema import ProvenanceInfo, SourceType
from src.ingestion.models import (
    Equation,
    ExtractionStatus,
    ExtractionTelemetry,
    Figure,
    IngestionDocument,
    IngestionDocumentMetadata,
    KnowledgeChunk,
    LicenseStatus,
    QualityStatus,
    Reference,
    Section,
    Table,
)


def test_document_id_computation():
    content = b"Quantum Mechanics and Statistical Physics Course Notes"
    doc_id = IngestionDocument.compute_document_id(content)
    assert isinstance(doc_id, str)
    assert len(doc_id) == 64
    # Determinism
    assert doc_id == IngestionDocument.compute_document_id(content)


def test_section_full_text_assembly():
    sec = Section(
        section_id="sec_1",
        title="1. Introduction to Quantum Physics",
        section_type="introduction",
        paragraphs=["Quantum mechanics describes physical phenomena at microscopic scales."],
        equations=[
            Equation(
                equation_id="eq_1",
                latex_content="i\\hbar \\frac{\\partial \\psi}{\\partial t} = \\hat{H}\\psi",
                equation_type="display",
            )
        ],
        tables=[
            Table(
                table_id="tab_1",
                headers=["Quantity", "Symbol"],
                rows=[["Planck constant", "h"]],
                markdown="| Quantity | Symbol |\n|---|---|\n| Planck constant | h |",
            )
        ],
    )
    full_text = sec.full_text()
    assert "1. Introduction to Quantum Physics" in full_text
    assert "Quantum mechanics describes physical phenomena" in full_text
    assert "i\\hbar" in full_text
    assert "Planck constant" in full_text


def test_ingestion_document_serialization():
    meta = IngestionDocumentMetadata(
        title="Thermodynamics of Black Holes",
        authors=["S. Hawking", "J. Bekenstein"],
        domain="science",
        topic="physics",
        subtopic="astrophysics",
        license="CC-BY-4.0",
        license_status=LicenseStatus.KNOWN,
        internal_only=False,
    )
    doc = IngestionDocument(
        document_id="a" * 64,
        source_path="/tmp/test_paper.pdf",
        source_file_hash="a" * 64,
        format="pdf",
        metadata=meta,
        sections=[
            Section(
                section_id="sec_0",
                title="Abstract",
                section_type="abstract",
                paragraphs=["We present thermodynamic properties of event horizons."],
            )
        ],
    )

    json_str = doc.to_json()
    loaded_doc = IngestionDocument.from_json(json_str)

    assert loaded_doc.document_id == doc.document_id
    assert loaded_doc.metadata.title == "Thermodynamics of Black Holes"
    assert loaded_doc.metadata.domain == "science"
    assert loaded_doc.metadata.internal_only is False
    assert len(loaded_doc.sections) == 1


def test_knowledge_chunk_creation_and_serialization():
    prov = ProvenanceInfo(
        source_type=SourceType.DOCUMENTATION.value,
        source="arxiv",
        source_id="arxiv_2301.0001",
    )
    chunk = KnowledgeChunk(
        chunk_id="chk_12345",
        document_id="doc_12345",
        section_id="sec_0",
        text="Sample scientific chunk text regarding wave-particle duality.",
        token_estimate=12,
        domain="science",
        topic="physics",
        subtopic="quantum_mechanics",
        source="arxiv",
        source_type="documentation",
        license="CC-BY-4.0",
        license_status="KNOWN",
        internal_only=False,
        quality_score=0.95,
        provenance=prov,
    )

    data = chunk.to_dict()
    assert data["chunk_id"] == "chk_12345"
    assert data["domain"] == "science"
    assert data["provenance"]["source"] == "arxiv"

    # From JSON
    json_repr = chunk.to_json()
    loaded = KnowledgeChunk.from_json(json_repr)
    assert loaded.chunk_id == chunk.chunk_id
    assert loaded.quality_score == 0.95
