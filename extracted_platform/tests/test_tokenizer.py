"""
Tests for Tokenizer Loading & Sequence Analysis (Phase 4.1).
"""

import pytest

from src.dataset.schema import DatasetRecord, Message, Role, SourceType, TaskType, DifficultyLevel
from src.training.config import TokenizerConfig
from src.training.tokenizer import MockQwenTokenizer, TrainingTokenizerWrapper


@pytest.fixture
def sample_records():
    return [
        DatasetRecord(
            messages=[
                Message(role=Role.SYSTEM, content="You are a coding tutor."),
                Message(role=Role.USER, content="What is a hash table?"),
                Message(role=Role.ASSISTANT, content="A hash table is a data structure implementing associative arrays with O(1) average lookup."),
            ],
            metadata={
                "domain": "software_engineering",
                "topic": "data_structures",
                "task_type": TaskType.QUESTION_ANSWERING,
                "difficulty": DifficultyLevel.BEGINNER,
                "provenance": {
                    "source_name": "test",
                    "source_type": SourceType.SYNTHETIC,
                    "created_at": "2026-08-12T00:00:00Z",
                    "pipeline_version": "2.2",
                },
            },
        ),
        DatasetRecord(
            messages=[
                Message(role=Role.USER, content="Explain merge sort."),
                Message(role=Role.ASSISTANT, content="Merge sort is a divide-and-conquer algorithm with O(n log n) runtime."),
            ],
            metadata={
                "domain": "software_engineering",
                "topic": "algorithms",
                "task_type": TaskType.QUESTION_ANSWERING,
                "difficulty": DifficultyLevel.BEGINNER,
                "provenance": {
                    "source_name": "test",
                    "source_type": SourceType.SYNTHETIC,
                    "created_at": "2026-08-12T00:00:00Z",
                    "pipeline_version": "2.2",
                },
            },
        ),
    ]


def test_mock_tokenizer_basic():
    tok = MockQwenTokenizer()
    assert tok.vocab_size == 151643
    assert tok.eos_token == "<|endoftext|>"
    assert tok.pad_token == "<|endoftext|>"
    assert tok.chat_template is not None

    conv = [{"role": "user", "content": "Hello"}]
    formatted = tok.apply_chat_template(conv, tokenize=False)
    assert "<|im_start|>user\nHello<|im_end|>\n" in formatted


def test_tokenizer_wrapper_loading():
    cfg = TokenizerConfig(fallback_pretrained_id="Qwen/Qwen2.5-3B")
    wrapper = TrainingTokenizerWrapper(cfg)
    tok = wrapper.load()
    assert tok is not None
    assert getattr(tok, "vocab_size", 0) > 0


def test_token_length_analysis(sample_records):
    cfg = TokenizerConfig()
    wrapper = TrainingTokenizerWrapper(cfg)
    report = wrapper.analyze_token_lengths(sample_records, max_seq_length=4096)

    assert report.record_count == 2
    assert report.total_tokens > 0
    assert report.mean > 0
    assert report.truncated_count == 0
    assert report.truncation_rate == 0.0
    assert report.counts_le_1024 == 2
