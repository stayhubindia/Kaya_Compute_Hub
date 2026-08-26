"""
Tests for scripts/build_dataset_v2.py CLI (Phase 3.5).
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest
from src.dataset.schema import DatasetRecord, Message, ProvenanceInfo, RecordMetadata, Role, SourceType


def test_cli_dry_run(tmp_path):
    # Create candidate file
    cand_file = tmp_path / "candidates.jsonl"
    with open(cand_file, "w", encoding="utf-8") as f:
        for i in range(10):
            rec = DatasetRecord(
                messages=[
                    Message(role=Role.USER, content=f"Question on quantum mechanics {i}?"),
                    Message(role=Role.ASSISTANT, content=f"Answer {i}: Energy $E = h \\nu$. Units: eV."),
                ],
                metadata=RecordMetadata(
                    domain="science",
                    topic="physics",
                    difficulty="intermediate",
                    task_type="conceptual_explanation",
                    source_type=SourceType.DOCUMENTATION.value,
                    source="nptel_doc",
                    provenance=ProvenanceInfo(
                        source_type=SourceType.DOCUMENTATION.value,
                        source="nptel_doc",
                        source_id=f"doc_{i}",
                        license="CC-BY-4.0",
                    ),
                ),
            )
            f.write(rec.model_dump_json() + "\n")

    cmd = [
        sys.executable,
        "scripts/build_dataset_v2.py",
        "--input", str(cand_file),
        "--target", "10",
        "--seed", "42",
        "--dry-run",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Phase 3.5 — Scientific Dataset-v2.0 Release Engine" in res.stdout
    assert "QA & Scorecard Summary" in res.stdout


def test_cli_build_and_freeze(tmp_path):
    cand_file = tmp_path / "candidates.jsonl"
    out_dir = tmp_path / "release_v2"

    with open(cand_file, "w", encoding="utf-8") as f:
        for i in range(10):
            rec = DatasetRecord(
                messages=[
                    Message(role=Role.USER, content=f"Question on optics {i}?"),
                    Message(role=Role.ASSISTANT, content=f"Answer {i}: Snell's Law $n_1 \\sin\\theta_1 = n_2 \\sin\\theta_2$."),
                ],
                metadata=RecordMetadata(
                    domain="science",
                    topic="optics",
                    difficulty="intermediate",
                    task_type="conceptual_explanation",
                    source_type=SourceType.DOCUMENTATION.value,
                    source="nptel_doc",
                    provenance=ProvenanceInfo(
                        source_type=SourceType.DOCUMENTATION.value,
                        source="nptel_doc",
                        source_id=f"doc_{i}",
                        license="CC-BY-4.0",
                    ),
                ),
            )
            f.write(rec.model_dump_json() + "\n")

    cmd = [
        sys.executable,
        "scripts/build_dataset_v2.py",
        "--input", str(cand_file),
        "--target", "10",
        "--seed", "42",
        "--version", "dataset-v2.0",
        "--output-dir", str(out_dir),
        "--freeze",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "checksums.sha256").exists()
    assert (out_dir / "train.jsonl").exists()
