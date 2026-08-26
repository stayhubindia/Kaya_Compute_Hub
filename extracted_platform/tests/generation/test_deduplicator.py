"""
Unit tests for InstructionDeduplicator (src/generation/deduplicator.py).
"""

import pytest
from src.dataset.schema import DatasetRecord, Message, RecordMetadata, Role
from src.generation.deduplicator import InstructionDeduplicator


def test_deduplicator_exact_and_near():
    dedup = InstructionDeduplicator(enable_exact=True, enable_near=True, near_threshold=0.85)

    r1 = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Explain quantum superposition."),
            Message(role=Role.ASSISTANT, content="Superposition allows a quantum state to be a linear combination of basis states."),
        ],
        metadata=RecordMetadata(domain="physics", topic="quantum", task_type="explanation", difficulty="intermediate"),
    )

    # Exact duplicate
    r2 = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Explain quantum superposition."),
            Message(role=Role.ASSISTANT, content="Superposition allows a quantum state to be a linear combination of basis states."),
        ],
        metadata=RecordMetadata(domain="physics", topic="quantum", task_type="explanation", difficulty="intermediate"),
    )

    # Distinct record
    r3 = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="What is quantum entanglement?"),
            Message(role=Role.ASSISTANT, content="Entanglement is a phenomenon where quantum states cannot be described independently."),
        ],
        metadata=RecordMetadata(domain="physics", topic="quantum", task_type="explanation", difficulty="intermediate"),
    )

    unique, report = dedup.deduplicate([r1, r2, r3])
    assert len(unique) == 2
    assert report.exact_duplicates == 1
    assert report.total_records == 3
