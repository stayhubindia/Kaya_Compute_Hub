"""
Tests for Data Collator & Assistant-Only Loss Masking (Phase 4.1).
"""

import pytest
import torch

from src.dataset.schema import DatasetRecord, Message, Role, SourceType, TaskType, DifficultyLevel
from src.training.collator import DataCollatorForAssistantOnlyLoss, mask_labels_for_assistant_only
from src.training.tokenizer import MockQwenTokenizer, TrainingTokenizerWrapper
from src.training.config import TokenizerConfig


@pytest.fixture
def tokenizer():
    wrapper = TrainingTokenizerWrapper(TokenizerConfig(fallback_pretrained_id="Qwen/Qwen2.5-3B"))
    return wrapper.load()


@pytest.fixture
def sample_record():
    return DatasetRecord(
        messages=[
            Message(role=Role.SYSTEM, content="You are a coding assistant."),
            Message(role=Role.USER, content="What is recursion?"),
            Message(role=Role.ASSISTANT, content="Recursion is a method where a function calls itself."),
        ],
        metadata={
            "domain": "software_engineering",
            "topic": "programming_concepts",
            "task_type": TaskType.QUESTION_ANSWERING,
            "difficulty": DifficultyLevel.BEGINNER,
            "provenance": {
                "source_name": "test",
                "source_type": SourceType.SYNTHETIC,
                "created_at": "2026-08-12T00:00:00Z",
                "pipeline_version": "2.2",
            },
        },
    )


def test_assistant_only_loss_masking_tokens(tokenizer):
    """Verify system & user tokens are -100 while assistant tokens are preserved."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain binary search."},
        {"role": "assistant", "content": "Binary search is O(log n)."},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    labels = mask_labels_for_assistant_only(input_ids, tokenizer)

    assert len(labels) == len(input_ids)
    assert -100 in labels

    # Count masked vs active tokens
    masked_count = sum(1 for l in labels if l == -100)
    active_count = sum(1 for l in labels if l != -100)

    assert masked_count > 0
    assert active_count > 0
    # First token (<|im_start|>) should be masked
    assert labels[0] == -100


def test_data_collator_batch_padding(tokenizer, sample_record):
    """Verify batch padding and tensor shapes."""
    collator = DataCollatorForAssistantOnlyLoss(
        tokenizer=tokenizer,
        max_seq_length=512,
        assistant_only_loss=True,
    )

    rec2 = DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Short query?"),
            Message(role=Role.ASSISTANT, content="Short answer."),
        ],
        metadata={
            "domain": "software_engineering",
            "topic": "qa",
            "task_type": TaskType.QUESTION_ANSWERING,
            "difficulty": DifficultyLevel.BEGINNER,
            "provenance": {
                "source_name": "test",
                "source_type": SourceType.SYNTHETIC,
                "created_at": "2026-08-12T00:00:00Z",
                "pipeline_version": "2.2",
            },
        },
    )

    batch = collator([sample_record, rec2])

    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["attention_mask"], torch.Tensor)
    assert isinstance(batch["labels"], torch.Tensor)

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
