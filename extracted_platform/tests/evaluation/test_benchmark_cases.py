"""
Unit tests for Benchmark Case Schema, Safe Evaluators & Builders (Phase 4.5).
"""

import pytest
from src.dataset.schema import Message, Role
from src.evaluation.benchmark_cases import (
    BenchmarkCase,
    BenchmarkEvaluationType,
    BenchmarkSuiteBuilder,
    SafeCodeEvaluator,
)


def test_valid_benchmark_case():
    case = BenchmarkCase(
        benchmark_id="bench-v1-0001",
        domain="programming",
        topic="python_basics",
        difficulty="beginner",
        task_type="coding",
        messages=[
            Message(role=Role.USER, content="Write a function to return True."),
            Message(role=Role.ASSISTANT, content="def test(): return True"),
        ],
        expected_behavior="Returns valid python function",
        reference_answer="def test(): return True",
        evaluation_type="code_based",
        evaluation_metadata={"required_symbols": ["test"], "static_checks": ["ast_parse"]},
    )
    assert case.benchmark_id == "bench-v1-0001"
    assert case.domain == "programming"
    assert case.difficulty == "beginner"
    assert case.evaluation_type == "code_based"
    assert len(case.get_prompt_messages()) == 1
    assert case.canonical_prompt_hash() is not None


def test_invalid_domain_rejection():
    with pytest.raises(Exception):
        BenchmarkCase(
            benchmark_id="bench-v1-0002",
            domain="invalid_nonexistent_domain",
            topic="test",
            difficulty="beginner",
            task_type="coding",
            messages=[
                Message(role=Role.USER, content="Test prompt"),
                Message(role=Role.ASSISTANT, content="Test response"),
            ],
            expected_behavior="Test",
            reference_answer="Test response",
            evaluation_type="reference_based",
        )


def test_invalid_first_message_assistant_rejection():
    with pytest.raises(ValueError, match="First non-system message must be 'user'"):
        BenchmarkCase(
            benchmark_id="bench-v1-0003",
            domain="programming",
            topic="test",
            difficulty="beginner",
            task_type="coding",
            messages=[
                Message(role=Role.ASSISTANT, content="I start first!"),
                Message(role=Role.USER, content="Okay"),
            ],
            expected_behavior="Test",
            reference_answer="Test response",
            evaluation_type="reference_based",
        )


def test_safe_code_evaluator_syntax_and_symbols():
    valid_code = "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)"
    is_valid, err = SafeCodeEvaluator.validate_syntax(valid_code)
    assert is_valid is True
    assert err is None

    has_sym, missing = SafeCodeEvaluator.check_required_constructs(valid_code, ["fib"])
    assert has_sym is True
    assert missing == []

    has_bad_sym, missing_bad = SafeCodeEvaluator.check_required_constructs(valid_code, ["quicksort"])
    assert has_bad_sym is False
    assert "quicksort" in missing_bad

    invalid_code = "def broken(:\n return"
    is_invalid, err_msg = SafeCodeEvaluator.validate_syntax(invalid_code)
    assert is_invalid is False
    assert err_msg is not None


def test_benchmark_suite_builder():
    cases = BenchmarkSuiteBuilder.generate_benchmark_suite(target_count=20, seed=42)
    assert len(cases) == 20
    seen_ids = set()
    for c in cases:
        assert isinstance(c, BenchmarkCase)
        assert c.benchmark_id not in seen_ids
        seen_ids.add(c.benchmark_id)
        assert c.domain in [
            "programming", "software_engineering", "cybersecurity", "linux_systems",
            "networking", "ai_ml", "mathematics", "science", "psychology",
            "human_behavior", "reasoning", "technology", "general_knowledge",
        ]
