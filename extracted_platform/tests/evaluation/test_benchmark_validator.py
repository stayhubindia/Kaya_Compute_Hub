"""
Unit tests for Benchmark Validator, Leakage Detection & Quality Auditing (Phase 4.5).
"""

import json
from pathlib import Path
import pytest

from src.dataset.schema import Message, Role
from src.evaluation.benchmark_cases import BenchmarkCase
from src.evaluation.benchmark_validator import BenchmarkValidator


def test_validator_clean_case():
    validator = BenchmarkValidator(excluded_split_files=[])
    case = BenchmarkCase(
        benchmark_id="bench-test-0001",
        domain="programming",
        topic="python",
        difficulty="beginner",
        task_type="coding",
        messages=[
            Message(role=Role.USER, content="Clean test question"),
            Message(role=Role.ASSISTANT, content="Clean test answer"),
        ],
        expected_behavior="Valid code",
        reference_answer="Clean test answer",
        evaluation_type="code_based",
        evaluation_metadata={"required_symbols": ["solve"], "static_checks": ["ast_parse"]},
    )
    report = validator.validate_suite([case])
    assert report.accepted_count == 1
    assert report.rejected_count == 0
    assert report.exact_overlaps == 0
    assert report.near_overlaps == 0


def test_validator_detects_internal_duplicate():
    validator = BenchmarkValidator(excluded_split_files=[])
    case1 = BenchmarkCase(
        benchmark_id="bench-dup-0001",
        domain="programming",
        topic="python",
        difficulty="beginner",
        task_type="explanation",
        messages=[
            Message(role=Role.USER, content="Same exact prompt"),
            Message(role=Role.ASSISTANT, content="Answer 1"),
        ],
        expected_behavior="Valid",
        reference_answer="Answer 1",
        evaluation_type="reference_based",
    )
    case2 = BenchmarkCase(
        benchmark_id="bench-dup-0002",
        domain="programming",
        topic="python",
        difficulty="beginner",
        task_type="explanation",
        messages=[
            Message(role=Role.USER, content="Same exact prompt"),
            Message(role=Role.ASSISTANT, content="Answer 2"),
        ],
        expected_behavior="Valid",
        reference_answer="Answer 2",
        evaluation_type="reference_based",
    )
    report = validator.validate_suite([case1, case2])
    assert report.internal_duplicates >= 1
    assert report.rejected_count >= 1


def test_validator_detects_training_leakage(tmp_path: Path):
    # Create fake training file
    train_record = {
        "messages": [
            {"role": "user", "content": "Explain python GIL in detail."},
            {"role": "assistant", "content": "The Global Interpreter Lock is..."},
        ],
        "metadata": {
            "domain": "programming",
            "topic": "concurrency",
            "task_type": "explanation",
            "difficulty": "intermediate",
            "quality_score": 1.0,
            "source": "unit_test",
            "source_type": "synthetic",
        },
    }
    train_file = tmp_path / "train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(train_record) + "\n")

    validator = BenchmarkValidator(excluded_split_files=[train_file])

    # Leaked case with exact same prompt
    leaked_case = BenchmarkCase(
        benchmark_id="bench-leak-0001",
        domain="programming",
        topic="concurrency",
        difficulty="intermediate",
        task_type="explanation",
        messages=[
            Message(role=Role.USER, content="Explain python GIL in detail."),
            Message(role=Role.ASSISTANT, content="Different assistant completion"),
        ],
        expected_behavior="Explains GIL",
        reference_answer="Different assistant completion",
        evaluation_type="reference_based",
    )

    report = validator.validate_suite([leaked_case])
    assert report.exact_overlaps == 1
    assert report.rejected_count == 1
    assert report.is_valid is False
