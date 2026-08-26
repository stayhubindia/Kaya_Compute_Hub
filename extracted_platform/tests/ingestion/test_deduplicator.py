"""
Unit tests for Ingestion Deduplicator (Phase 3.3).
"""

import pytest
from src.dataset.schema import ProvenanceInfo
from src.ingestion.deduplicator import IngestionDeduplicator
from src.ingestion.models import KnowledgeChunk


def create_dummy_chunk(chunk_id: str, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id="doc_1",
        section_id="sec_1",
        text=text,
        domain="science",
        topic="physics",
        source="arxiv",
        source_type="documentation",
        provenance=ProvenanceInfo(source="arxiv"),
    )


def test_exact_and_near_duplicate_chunk_filtering():
    dedup = IngestionDeduplicator(enable_near_dedup=True, near_duplicate_threshold=0.85)

    c1 = create_dummy_chunk("c1", "The thermodynamics of black holes relates area to entropy.")
    c2 = create_dummy_chunk("c2", "The thermodynamics of black holes relates area to entropy.")  # exact duplicate
    c3 = create_dummy_chunk("c3", "The thermodynamics of black holes relates area to entropy! ")  # near duplicate
    c4 = create_dummy_chunk("c4", "Quantum entanglement in spin systems exhibits non-local correlations.")  # unique

    unique_chunks, report = dedup.deduplicate_chunks([c1, c2, c3, c4])

    assert len(unique_chunks) == 2
    assert unique_chunks[0].chunk_id == "c1"
    assert unique_chunks[1].chunk_id == "c4"
    assert report.exact_duplicate_chunks == 1
    assert report.near_duplicate_chunks == 1
    assert report.unique_chunks == 2


def test_document_deduplication():
    dedup = IngestionDeduplicator()
    assert dedup.is_duplicate_document("doc_hash_1") is False
    assert dedup.is_duplicate_document("doc_hash_1") is True
    assert dedup.is_duplicate_document("doc_hash_2") is False
