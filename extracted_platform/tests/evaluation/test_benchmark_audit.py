"""
Unit tests for Benchmark Semantic & Reference-Answer Quality Audit Engine (Phase 4.6).
"""

import json
from pathlib import Path
import pytest

from src.dataset.schema import Message, Role
from src.evaluation.benchmark_cases import BenchmarkCase
from src.evaluation.benchmark_audit import (
    AnswerLeakageAuditor,
    AuditStatus,
    BenchmarkAuditor,
    CaseAuditResult,
    CodeAuditor,
    DifficultyAuditor,
    DuplicateClass,
    FactCategory,
    FactualClaimAuditor,
    MathematicsAuditor,
    MultiTurnAuditor,
    ReasoningAuditor,
    SemanticDuplicateAuditor,
    StructuralAuditor,
    SuiteAuditReport,
    TaskTypeAuditor,
)


def create_sample_case(
    benchmark_id: str = "bench-test-0001",
    domain: str = "programming",
    difficulty: str = "intermediate",
    task_type: str = "coding",
    eval_type: str = "code_based",
    prompt: str = "Write a function solve() in Python.",
    ref: str = "```python\ndef solve():\n    return True\n```",
    meta: dict = None,
) -> BenchmarkCase:
    return BenchmarkCase(
        benchmark_id=benchmark_id,
        domain=domain,
        topic="python_basics",
        difficulty=difficulty,
        task_type=task_type,
        messages=[
            Message(role=Role.USER, content=prompt),
            Message(role=Role.ASSISTANT, content=ref),
        ],
        expected_behavior="Returns valid solve()",
        reference_answer=ref,
        evaluation_type=eval_type,
        evaluation_metadata=meta or {"required_symbols": ["solve"], "static_checks": ["ast_parse"]},
    )


def test_structural_auditor():
    valid_case = create_sample_case()
    ok, issues = StructuralAuditor.audit(valid_case)
    assert ok is True
    assert len(issues) == 0


def test_mathematics_auditor_correct_and_incorrect():
    # Correct math case
    case_correct = create_sample_case(
        domain="mathematics",
        task_type="calculation",
        eval_type="numerical",
        prompt="Calculate 2 + 2.",
        ref="The computed result is 4.0.",
        meta={"expected_numerical_values": {"ans": 4.0}},
    )
    is_math, ok, issues = MathematicsAuditor.audit(case_correct)
    assert is_math is True
    assert ok is True
    assert len(issues) == 0

    # Incorrect/missing value in math case
    case_incorrect = create_sample_case(
        domain="mathematics",
        task_type="calculation",
        eval_type="numerical",
        prompt="Calculate 5 * 5.",
        ref="The result is unclear.",
        meta={"expected_numerical_values": {"ans": 25.0}},
    )
    is_math, ok, issues = MathematicsAuditor.audit(case_incorrect)
    assert is_math is True
    assert ok is False
    assert len(issues) > 0


def test_code_auditor_syntax_and_symbols():
    # Valid syntax & symbol
    case_valid = create_sample_case()
    is_code, ok, issues = CodeAuditor.audit(case_valid)
    assert is_code is True
    assert ok is True

    # Invalid Python syntax
    case_broken = create_sample_case(
        ref="```python\ndef broken(:\n  return\n```"
    )
    is_code, ok, issues = CodeAuditor.audit(case_broken)
    assert is_code is True
    assert ok is False
    assert any("Syntax error" in i for i in issues)

    # Missing required symbol
    case_missing_sym = create_sample_case(
        ref="```python\ndef wrong_name():\n    return 42\n```",
        meta={"required_symbols": ["solve"]},
    )
    is_code, ok, issues = CodeAuditor.audit(case_missing_sym)
    assert is_code is True
    assert ok is False
    assert any("Missing required symbols" in i for i in issues)


def test_reasoning_auditor():
    case_reasoning = create_sample_case(
        domain="reasoning",
        task_type="reasoning",
        eval_type="reasoning",
        prompt="Deduce the invariant for the scenario.",
        ref="Step 1: Check invariants. Therefore the optimal invariant is maintained.",
        meta={"expected_conclusion": "optimal invariant"},
    )
    is_reason, ok, issues = ReasoningAuditor.audit(case_reasoning)
    assert is_reason is True
    assert ok is True


def test_factual_claim_auditor():
    case_time = create_sample_case(
        prompt="What is currently the latest version of framework X as of this year?"
    )
    cat, req_verify = FactualClaimAuditor.audit(case_time)
    assert cat == FactCategory.TIME_SENSITIVE_FACT
    assert req_verify is True

    case_tech = create_sample_case(
        domain="networking",
        prompt="Explain TCP handshake."
    )
    cat, req_verify = FactualClaimAuditor.audit(case_tech)
    assert cat == FactCategory.TECHNICAL_FACT
    assert req_verify is False


def test_answer_leakage_auditor():
    # Clean case
    clean_case = create_sample_case()
    leaked, _ = AnswerLeakageAuditor.audit(clean_case)
    assert leaked is False

    # Leaked calculation answer in prompt
    leaked_case = create_sample_case(
        task_type="calculation",
        eval_type="numerical",
        prompt="Since the result is 42.0, calculate 42.0.",
        ref="The computed answer is 42.0.",
        meta={"expected_numerical_values": {"res": 42.0}},
    )
    leaked, msg = AnswerLeakageAuditor.audit(leaked_case)
    assert leaked is True
    assert "reveals expected calculation answer" in msg


def test_semantic_duplicate_auditor():
    case1 = create_sample_case(benchmark_id="b1", prompt="Explain the TCP handshake mechanism in networking.")
    case2 = create_sample_case(benchmark_id="b2", prompt="Explain the TCP handshake mechanism in networking.")
    case3 = create_sample_case(benchmark_id="b3", prompt="Derive Schwarzschild radius in astrophysics.")

    res = SemanticDuplicateAuditor.audit_suite([case1, case2, case3])
    assert res["b1"][0] == DuplicateClass.DUPLICATE
    assert res["b2"][0] == DuplicateClass.DUPLICATE
    assert res["b3"][0] == DuplicateClass.UNIQUE


def test_benchmark_auditor_suite_and_critical_override(tmp_path: Path):
    clean_case = create_sample_case(benchmark_id="b-clean")
    broken_case = create_sample_case(
        benchmark_id="b-broken",
        ref="```python\ndef bad(:\n```"
    )

    auditor = BenchmarkAuditor()
    report, results = auditor.audit_suite([clean_case, broken_case], benchmark_sha256="test_sha256")

    assert report.case_count == 2
    assert report.quality_scores.mean > 0
    assert len(report.critical_failures) == 1
    assert report.critical_failures[0]["benchmark_id"] == "b-broken"
    assert report.release_decision == "NEEDS_REVISION"

    # Test report saving
    BenchmarkAuditor.save_reports(report, results, tmp_path)
    assert (tmp_path / "benchmark_audit.json").exists()
    assert (tmp_path / "benchmark_audit.md").exists()
    assert (tmp_path / "benchmark_case_audit.jsonl").exists()
