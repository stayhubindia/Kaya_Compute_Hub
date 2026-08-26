"""
Test Suite: Scientific Quality, Grounding, Equation & Table Fidelity Audit (Phase 3.5).
Verifies that dataset-v2.0 maintains high scientific rigor, equation fidelity, and zero hallucinations.
"""

import json
from pathlib import Path
import pytest

from src.dataset.final_qa_auditor import FinalQAAuditor, GateStatus


@pytest.fixture
def dataset_root():
    return Path("data/instruction_dataset/v2.0").resolve()


def test_scientific_quality_scores(dataset_root):
    """Verifies that all records meet or exceed quality score thresholds."""
    auditor = FinalQAAuditor(dataset_dir=dataset_root)
    report = auditor.run_full_audit()

    qa = report.quality_audit
    assert qa.mean_score >= 0.85
    assert qa.min_score >= 0.85
    assert qa.pct_ge_085 == 100.0
    assert qa.pct_ge_090 >= 95.0

    # Gate G8 check
    g8 = next(g for g in report.gate_matrix if g.gate_id == "G8")
    assert g8.status == GateStatus.PASS


def test_source_grounding_audit(dataset_root):
    """Verifies that records are verified grounded against source NPTEL chunks."""
    auditor = FinalQAAuditor(dataset_dir=dataset_root)
    report = auditor.run_full_audit()

    qa = report.quality_audit
    assert qa.unsupported_count == 0
    assert qa.grounding_rate >= 95.0

    # Gate G5 check
    g5 = next(g for g in report.gate_matrix if g.gate_id == "G5")
    assert g5.status == GateStatus.PASS


def test_equation_and_table_fidelity(dataset_root):
    """Verifies equation syntax correctness and Markdown table validity."""
    auditor = FinalQAAuditor(dataset_dir=dataset_root)
    report = auditor.run_full_audit()

    eq = report.equation_audit
    assert eq.equation_records_count > 0
    assert eq.equation_fidelity_rate >= 95.0

    tb = report.table_audit
    assert tb.table_records_count > 0
    assert tb.table_fidelity_rate >= 90.0

    # Gate G9 and G10 checks
    g9 = next(g for g in report.gate_matrix if g.gate_id == "G9")
    g10 = next(g for g in report.gate_matrix if g.gate_id == "G10")
    assert g9.status == GateStatus.PASS
    assert g10.status == GateStatus.PASS
