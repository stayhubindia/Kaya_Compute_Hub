import pytest
from pydantic import ValidationError

from src.dataset.schema import (
    DatasetRecord,
    DifficultyLevel,
    Message,
    RecordMetadata,
    Role,
)


def test_valid_single_turn_record():
    record_dict = {
        "messages": [
            {"role": "user", "content": "Explain how epoll works in Linux."},
            {"role": "assistant", "content": "epoll is an I/O event notification facility in Linux kernel."},
        ],
        "metadata": {
            "domain": "linux_systems",
            "topic": "kernel_internals",
            "task_type": "explanation",
            "difficulty": "advanced",
            "quality_score": 0.92,
            "source": "curated_tests",
            "source_type": "curated",
            "created_at": "2026-08-11T16:00:00Z",
        },
    }
    record = DatasetRecord.from_dict(record_dict)
    assert record.is_single_turn() is True
    assert record.turn_count() == 1
    assert record.messages[0].role == Role.USER
    assert record.messages[1].role == Role.ASSISTANT
    assert len(record.canonical_content_hash()) == 64


def test_valid_multi_turn_record():
    record_dict = {
        "messages": [
            {"role": "system", "content": "You are an expert systems engineer."},
            {"role": "user", "content": "What is copy-on-write (COW)?"},
            {"role": "assistant", "content": "COW is an optimization strategy used in virtual memory management."},
            {"role": "user", "content": "How does fork() utilize it?"},
            {"role": "assistant", "content": "fork() marks parent pages as read-only rather than duplicating them immediately."},
        ],
        "metadata": {
            "domain": "linux_systems",
            "topic": "kernel_internals",
            "task_type": "multi_turn",
            "difficulty": "advanced",
            "quality_score": 0.95,
            "source": "curated_tests",
            "source_type": "curated",
            "created_at": "2026-08-11T16:00:00Z",
        },
    }
    record = DatasetRecord.from_dict(record_dict)
    assert record.is_single_turn() is False
    assert record.turn_count() == 2
    assert record.messages[0].role == Role.SYSTEM


def test_invalid_first_message_assistant():
    record_dict = {
        "messages": [
            {"role": "assistant", "content": "Hello without user prompt."},
        ],
        "metadata": {
            "domain": "programming",
            "topic": "python",
            "task_type": "explanation",
            "difficulty": "beginner",
            "quality_score": 0.9,
            "source": "tests",
            "source_type": "synthetic",
            "created_at": "2026-08-11T16:00:00Z",
        },
    }
    with pytest.raises(ValidationError):
        DatasetRecord.from_dict(record_dict)


def test_consecutive_same_roles():
    record_dict = {
        "messages": [
            {"role": "user", "content": "First prompt"},
            {"role": "user", "content": "Second prompt without assistant response"},
        ],
        "metadata": {
            "domain": "programming",
            "topic": "python",
            "task_type": "explanation",
            "difficulty": "beginner",
            "quality_score": 0.9,
            "source": "tests",
            "source_type": "synthetic",
            "created_at": "2026-08-11T16:00:00Z",
        },
    }
    with pytest.raises(ValidationError):
        DatasetRecord.from_dict(record_dict)


def test_empty_content_rejected():
    with pytest.raises(ValidationError):
        Message(role=Role.USER, content="   ")


def test_invalid_difficulty_rejected():
    with pytest.raises(ValidationError):
        RecordMetadata(
            domain="programming",
            topic="python",
            task_type="coding",
            difficulty="super_hard",
            quality_score=0.9,
            source="test",
            source_type="synthetic",
            created_at="2026-08-11T16:00:00Z",
        )
