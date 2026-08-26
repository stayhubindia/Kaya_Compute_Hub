"""
Unit tests for Generation data models and schemas (src/generation/models.py).
"""

import pytest
from src.dataset.schema import DatasetRecord, Message, Role, SourceType
from src.generation.models import (
    CandidateRecord,
    ContentType,
    ExtendedProvenance,
    GroundingEvaluation,
    KnowledgeUnit,
    MathematicalValidation,
)
from src.ingestion.models import KnowledgeChunk, ProvenanceInfo


def test_content_type_enum():
    assert ContentType.CONCEPT.value == "concept"
    assert ContentType.DERIVATION.value == "derivation"
    assert ContentType.CALCULATION.value == "calculation"
    assert ContentType.TABLE_DATA.value == "table_data"


def test_knowledge_unit_from_chunk():
    prov = ProvenanceInfo(
        source_type=SourceType.DOCUMENTATION.value,
        source="nptel",
        source_id="chunk_001",
    )
    chunk = KnowledgeChunk(
        chunk_id="chunk_001",
        document_id="doc_123",
        section_id="sec_456",
        text="The Schrodinger equation describes quantum states.",
        domain="physics",
        topic="quantum_mechanics",
        source="nptel",
        source_type=SourceType.DOCUMENTATION.value,
        provenance=prov,
    )

    ku = KnowledgeUnit.from_knowledge_chunk(chunk, title="Quantum Intro")
    assert ku.unit_id == "ku_chunk_001"
    assert ku.document_id == "doc_123"
    assert ku.section_id == "sec_456"
    assert ku.title == "Quantum Intro"
    assert ku.domain == "physics"
    assert ku.topic == "quantum_mechanics"

    d = ku.to_dict()
    assert d["unit_id"] == "ku_chunk_001"
    assert d["domain"] == "physics"


def test_extended_provenance_conversion():
    ext_prov = ExtendedProvenance(
        source="nptel",
        source_id="chunk_999",
        knowledge_document_id="doc_999",
        knowledge_chunk_id="chunk_999",
        knowledge_section_id="sec_999",
        generation_seed=42,
    )

    prov_info = ext_prov.to_provenance_info()
    assert prov_info.source == "nptel"
    assert prov_info.source_id == "chunk_999"
    assert prov_info.generator == "scientific_instruction_engine"


def test_candidate_record_to_dict():
    prov = ProvenanceInfo(
        source_type=SourceType.DOCUMENTATION.value,
        source="nptel",
        source_id="chunk_001",
    )
    chunk = KnowledgeChunk(
        chunk_id="chunk_001",
        document_id="doc_123",
        section_id="sec_456",
        text="Some text about optics.",
        domain="physics",
        topic="optics",
        source="nptel",
        source_type=SourceType.DOCUMENTATION.value,
        provenance=prov,
    )
    ku = KnowledgeUnit.from_knowledge_chunk(chunk)
    ext_prov = ExtendedProvenance(source="nptel", source_id="chunk_001")

    from src.dataset.schema import RecordMetadata
    rec = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Explain Snell's law."),
            Message(role=Role.ASSISTANT, content="Snell's law describes the refraction of light."),
        ],
        metadata=RecordMetadata(
            domain="physics",
            topic="optics",
            task_type="explanation",
            difficulty="intermediate",
            quality_score=0.92,
        ),
    )

    cand = CandidateRecord(
        record_id="cand_001",
        record=rec,
        knowledge_unit=ku,
        task_type="explanation",
        difficulty="intermediate",
        provenance_extended=ext_prov,
        quality_score=0.92,
        is_accepted=True,
    )

    cand_dict = cand.to_dict()
    assert cand_dict["record_id"] == "cand_001"
    assert cand_dict["task_type"] == "explanation"
    assert cand_dict["is_accepted"] is True
