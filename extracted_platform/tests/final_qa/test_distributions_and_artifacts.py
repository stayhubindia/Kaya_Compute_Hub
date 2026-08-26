"""
Test Suite: Task Distributions, Difficulty, Token Lengths, and Artifacts (Phase 3.5).
"""

from pathlib import Path
import pytest

from src.dataset.final_qa_auditor import FinalQAAuditor, GateStatus


@pytest.fixture
def dataset_root():
    return Path("data/instruction_dataset/v2.0").resolve()


def test_task_and_difficulty_distributions(dataset_root):
    """Verifies task type diversity and multi-tier difficulty representation."""
    auditor = FinalQAAuditor(dataset_dir=dataset_root)
    report = auditor.run_full_audit()

    dist = report.distribution_audit
    assert len(dist.task_distribution) >= 10
    assert len(dist.difficulty_distribution) >= 3

    g12 = next(g for g in report.gate_matrix if g.gate_id == "G12")
    assert g12.status == GateStatus.PASS


def test_token_budget_and_artifacts(dataset_root):
    """Verifies token sequence lengths and absence of critical tokenizer artifacts."""
    auditor = FinalQAAuditor(dataset_dir=dataset_root)
    report = auditor.run_full_audit()

    tok = report.token_and_artifacts
    assert tok.total_lengths.get("mean", 0) > 50
    assert tok.total_lengths.get("max", 0) <= 4096
    assert len(tok.tokenizer_artifacts_detected) == 0

    g13 = next(g for g in report.gate_matrix if g.gate_id == "G13")
    assert g13.status == GateStatus.PASS
