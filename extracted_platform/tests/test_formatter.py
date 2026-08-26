"""
Tests for Conversation Formatter (Phase 4.1).
"""

import pytest

from src.dataset.schema import DatasetRecord, Message, Role, SourceType, TaskType, DifficultyLevel
from src.training.formatter import ConversationFormatter
from src.training.tokenizer import MockQwenTokenizer


@pytest.fixture
def formatter():
    return ConversationFormatter(MockQwenTokenizer())


def test_single_turn_formatting(formatter):
    rec = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="What is dynamic programming?"),
            Message(role=Role.ASSISTANT, content="Dynamic programming breaks problems into overlapping subproblems with optimal substructure."),
        ],
        metadata={
            "domain": "software_engineering",
            "topic": "algorithms",
            "task_type": TaskType.QUESTION_ANSWERING,
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "provenance": {
                "source_name": "test",
                "source_type": SourceType.SYNTHETIC,
                "created_at": "2026-08-12T00:00:00Z",
                "pipeline_version": "2.2",
            },
        },
    )
    formatted = formatter.format_record(rec)
    assert "<|im_start|>user\nWhat is dynamic programming?<|im_end|>\n" in formatted.text
    assert "<|im_start|>assistant\nDynamic programming breaks problems" in formatted.text
    assert len(formatted.turns) == 2
    assert not formatted.turns[0].is_assistant
    assert formatted.turns[1].is_assistant


def test_multi_turn_with_system(formatter):
    rec = DatasetRecord(
        messages=[
            Message(role=Role.SYSTEM, content="You are an expert Linux sysadmin."),
            Message(role=Role.USER, content="How to check open ports?"),
            Message(role=Role.ASSISTANT, content="Use `ss -tuln` or `netstat -tuln`."),
            Message(role=Role.USER, content="What if ss is not installed?"),
            Message(role=Role.ASSISTANT, content="You can check `/proc/net/tcp` or install `iproute2`."),
        ],
        metadata={
            "domain": "linux_systems",
            "topic": "networking",
            "task_type": TaskType.MULTI_TURN,
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "provenance": {
                "source_name": "test",
                "source_type": SourceType.SYNTHETIC,
                "created_at": "2026-08-12T00:00:00Z",
                "pipeline_version": "2.2",
            },
        },
    )
    formatted = formatter.format_record(rec)
    assert len(formatted.turns) == 5
    assert formatted.turns[0].role == "system"
    assert not formatted.turns[0].is_assistant
    assert formatted.turns[1].role == "user"
    assert formatted.turns[2].role == "assistant"
    assert formatted.turns[2].is_assistant
    assert formatted.turns[3].role == "user"
    assert formatted.turns[4].role == "assistant"
    assert formatted.turns[4].is_assistant
