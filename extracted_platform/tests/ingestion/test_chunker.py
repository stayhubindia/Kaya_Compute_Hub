"""
Unit tests for Semantic Document Chunker (Phase 3.3).
"""

import pytest
from src.ingestion.chunker import SemanticChunker
from src.ingestion.models import (
    Equation,
    IngestionDocument,
    IngestionDocumentMetadata,
    Section,
    Table,
)


def test_chunker_section_and_equation_binding():
    sec = Section(
        section_id="sec_101",
        title="Electromagnetism and Maxwell Equations",
        section_type="theory",
        paragraphs=[
            "Maxwell's equations represent one of the most elegant achievements in classical physics.",
            "The differential form relates electric and magnetic fields to charge and current densities.",
        ],
        equations=[
            Equation(
                equation_id="eq_m1",
                latex_content=r"\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}",
                equation_type="display",
            )
        ],
    )
    meta = IngestionDocumentMetadata(
        title="Classical Electrodynamics",
        source="nptel",
        domain="science",
        topic="physics",
        subtopic="electromagnetism",
    )
    doc = IngestionDocument(
        document_id="doc_em_123",
        source_path="/tmp/em.pdf",
        source_file_hash="doc_em_123",
        format="pdf",
        metadata=meta,
        sections=[sec],
    )

    chunker = SemanticChunker(min_chunk_tokens=10, max_chunk_tokens=500)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    c = chunks[0]
    assert c.document_id == "doc_em_123"
    assert c.section_id == "sec_101"
    assert c.domain == "science"
    assert c.topic == "physics"
    assert c.subtopic == "electromagnetism"
    assert r"\nabla \cdot \mathbf{E}" in c.text
    assert c.token_estimate > 0


def test_chunk_id_determinism():
    chunker = SemanticChunker()
    doc_id = "doc_test_hash"
    sec_id = "sec_test_0"
    text = "Deterministic physics knowledge segment."

    id1 = chunker.chunk_document.__globals__["KnowledgeChunk"].generate_chunk_id(doc_id, sec_id, 0, text)
    id2 = chunker.chunk_document.__globals__["KnowledgeChunk"].generate_chunk_id(doc_id, sec_id, 0, text)

    assert id1 == id2
    assert len(id1) == 16
