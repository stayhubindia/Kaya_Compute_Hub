"""
Test Suite: Count Reconciliation Audit (Phase 3.5).
Verifies strict mathematical identity across all stages of the synthesis and split pipeline.
"""

import json
from pathlib import Path
import pytest

from src.dataset.final_qa_auditor import FinalQAAuditor, GateStatus


@pytest.fixture
def dataset_root():
    return Path("data/instruction_dataset/v2.0").resolve()


def test_count_reconciliation_math(dataset_root):
    """Verifies the exact mathematical accounting identities for dataset-v2.0."""
    raw_lines = len((dataset_root / "raw" / "candidates.jsonl").read_text(encoding="utf-8").splitlines())
    rej_lines = len((dataset_root / "processed" / "rejected.jsonl").read_text(encoding="utf-8").splitlines())
    acc_lines = len((dataset_root / "processed" / "accepted.jsonl").read_text(encoding="utf-8").splitlines())
    tr_lines = len((dataset_root / "splits" / "train.jsonl").read_text(encoding="utf-8").splitlines())
    val_lines = len((dataset_root / "splits" / "validation.jsonl").read_text(encoding="utf-8").splitlines())
    te_lines = len((dataset_root / "splits" / "test.jsonl").read_text(encoding="utf-8").splitlines())

    assert raw_lines == 4822, f"Expected 4,822 raw candidates, got {raw_lines}"
    assert rej_lines == 1011, f"Expected 1,011 rejected candidates, got {rej_lines}"
    assert acc_lines == 2452, f"Expected 2,452 accepted records, got {acc_lines}"
    assert tr_lines == 2206, f"Expected 2,206 train records, got {tr_lines}"
    assert val_lines == 123, f"Expected 123 val records, got {val_lines}"
    assert te_lines == 123, f"Expected 123 test records, got {te_lines}"

    # Split identity
    assert tr_lines + val_lines + te_lines == acc_lines

    # Dedup accounting identity
    exact_duplicates = 817
    near_duplicates = 542
    total_dedup = exact_duplicates + near_duplicates
    accepted_before_dedup = acc_lines + total_dedup

    assert accepted_before_dedup == 3811
    assert raw_lines == rej_lines + accepted_before_dedup


def test_auditor_count_reconciliation_method(dataset_root):
    """Tests that FinalQAAuditor generates a verified count reconciliation report."""
    auditor = FinalQAAuditor(dataset_dir=dataset_root)
    report = auditor.run_full_audit()

    cr = report.count_reconciliation
    assert cr.raw_candidates == 4822
    assert cr.rejected_candidates == 1011
    assert cr.accepted_before_dedup == 3811
    assert cr.total_duplicates_removed == 1359
    assert cr.final_unique_records == 2452
    assert cr.train_records == 2206
    assert cr.validation_records == 123
    assert cr.test_records == 123

    assert cr.raw_reconciled is True
    assert cr.dedup_reconciled is True
    assert cr.split_reconciled is True
    assert cr.is_fully_reconciled is True

    # Gate G2 check
    g2 = next(g for g in report.gate_matrix if g.gate_id == "G2")
    assert g2.status == GateStatus.PASS
    assert g2.score == 1.0
    assert g2.is_critical is True
