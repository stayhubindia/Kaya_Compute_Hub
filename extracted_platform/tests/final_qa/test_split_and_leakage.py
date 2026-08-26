"""
Test Suite: Split Integrity and Cross-Split Leakage Audit (Phase 3.5).
Verifies strict split isolation with zero content and chunk leakage between train, val, and test splits.
"""

import hashlib
import json
from pathlib import Path
import pytest

from src.dataset.final_qa_auditor import FinalQAAuditor, GateStatus


@pytest.fixture
def dataset_root():
    return Path("data/instruction_dataset/v2.0").resolve()


def test_split_proportions_and_counts(dataset_root):
    """Verifies train/val/test split counts and approximate 90/5/5 ratio."""
    tr = len((dataset_root / "splits" / "train.jsonl").read_text(encoding="utf-8").splitlines())
    val = len((dataset_root / "splits" / "validation.jsonl").read_text(encoding="utf-8").splitlines())
    te = len((dataset_root / "splits" / "test.jsonl").read_text(encoding="utf-8").splitlines())
    total = tr + val + te

    assert tr == 2206
    assert val == 123
    assert te == 123
    assert total == 2452

    assert 88.0 <= (tr / total) * 100 <= 92.0
    assert 4.0 <= (val / total) * 100 <= 6.0
    assert 4.0 <= (te / total) * 100 <= 6.0


def test_zero_cross_split_content_leakage(dataset_root):
    """Verifies zero overlap of canonical message content hashes across splits."""
    def get_hashes(file_name):
        hashes = set()
        for line in (dataset_root / "splits" / file_name).read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            full_text = " ".join([m["content"] for m in d["messages"]])
            hashes.add(hashlib.sha256(full_text.strip().encode("utf-8")).hexdigest())
        return hashes

    tr_h = get_hashes("train.jsonl")
    val_h = get_hashes("validation.jsonl")
    te_h = get_hashes("test.jsonl")

    assert len(tr_h.intersection(val_h)) == 0, "Train-Validation content overlap detected!"
    assert len(tr_h.intersection(te_h)) == 0, "Train-Test content overlap detected!"
    assert len(val_h.intersection(te_h)) == 0, "Validation-Test content overlap detected!"


def test_zero_cross_split_chunk_leakage(dataset_root):
    """Verifies zero overlap of source chunk IDs across splits."""
    def get_chunk_ids(file_name):
        cids = set()
        for line in (dataset_root / "splits" / file_name).read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            cid = d.get("metadata", {}).get("extra", {}).get("chunk_id") or d.get("metadata", {}).get("source_id")
            if cid:
                cids.add(str(cid))
        return cids

    tr_c = get_chunk_ids("train.jsonl")
    val_c = get_chunk_ids("validation.jsonl")
    te_c = get_chunk_ids("test.jsonl")

    assert len(tr_c.intersection(val_c)) == 0, "Train-Validation chunk overlap detected!"
    assert len(tr_c.intersection(te_c)) == 0, "Train-Test chunk overlap detected!"
    assert len(val_c.intersection(te_c)) == 0, "Validation-Test chunk overlap detected!"


def test_leakage_gate_pass(dataset_root):
    """Verifies Gate G4 PASS status in FinalQAAuditor."""
    auditor = FinalQAAuditor(dataset_dir=dataset_root)
    report = auditor.run_full_audit()

    g4 = next(g for g in report.gate_matrix if g.gate_id == "G4")
    assert g4.status == GateStatus.PASS
    assert g4.score == 1.0
    assert report.leakage_audit.is_leak_free is True
