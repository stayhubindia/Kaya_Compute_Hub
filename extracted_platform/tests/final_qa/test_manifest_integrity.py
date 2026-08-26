"""
Test Suite: Manifest Integrity Audit (Phase 3.5).
Verifies metadata, checksums, split records, and schema completeness of manifests.
"""

import json
from pathlib import Path
import pytest


@pytest.fixture
def dataset_root():
    return Path("data/instruction_dataset/v2.0").resolve()


def test_dataset_manifest_integrity(dataset_root):
    """Validates structure and exact figures in dataset_manifest.json."""
    manifest_file = dataset_root / "manifests" / "dataset_manifest.json"
    assert manifest_file.is_file(), f"Manifest file missing at {manifest_file}"

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_version"] == "dataset-v2.0"
    assert data["seed"] == 42
    assert "counts" in data
    assert "quality" in data
    assert "leakage" in data
    assert "files" in data

    counts = data["counts"]
    assert counts["raw_generated"] == 4822
    assert counts["rejected"] == 1011
    assert counts["total_unique_records"] == 2452
    assert counts["train_records"] == 2206
    assert counts["validation_records"] == 123
    assert counts["test_records"] == 123


def test_final_qa_manifest_integrity(dataset_root):
    """Validates structure and fields in final_qa_manifest.json."""
    qa_manifest_file = dataset_root / "manifests" / "final_qa_manifest.json"
    assert qa_manifest_file.is_file(), f"QA manifest missing at {qa_manifest_file}"

    with open(qa_manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_version"] == "dataset-v2.0"
    assert data["all_critical_gates_passed"] is True
    assert "counts" in data
    assert "quality" in data
    assert "leakage" in data
    assert "provenance" in data
    assert "equations" in data
    assert "tables" in data
    assert "distributions" in data
    assert "token_budget" in data
    assert "reproducibility" in data
    assert "checksums" in data
    assert "gates" in data
    assert len(data["gates"]) == 15
