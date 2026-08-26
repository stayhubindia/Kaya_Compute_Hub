"""
CLI tests for scripts/generate_instruction_dataset.py.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest

from src.dataset.schema import SourceType
from src.ingestion.models import KnowledgeChunk, ProvenanceInfo


def test_cli_dry_run_and_execution(tmp_path: Path):
    in_dir = tmp_path / "input_ingested"
    out_dir = tmp_path / "output_candidates"
    in_dir.mkdir()

    # Create dummy chunks.jsonl
    chunks_file = in_dir / "chunks.jsonl"
    prov = ProvenanceInfo(source_type=SourceType.DOCUMENTATION.value, source="nptel", source_id="c1")
    chunk = KnowledgeChunk(
        chunk_id="c1",
        document_id="d1",
        section_id="s1",
        text="Newton's second law states that force equals mass times acceleration $F = ma$.",
        domain="physics",
        topic="classical_mechanics",
        source="nptel",
        source_type=SourceType.DOCUMENTATION.value,
        provenance=prov,
    )
    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write(chunk.to_json() + "\n")

    # 1. Dry run CLI call
    cmd_dry = [
        sys.executable,
        "scripts/generate_instruction_dataset.py",
        "--input",
        str(in_dir),
        "--output-dir",
        str(out_dir),
        "--dry-run",
    ]
    res_dry = subprocess.run(cmd_dry, capture_output=True, text=True)
    assert res_dry.returncode == 0
    assert "DRY RUN SUMMARY" in res_dry.stderr or "DRY RUN SUMMARY" in res_dry.stdout

    # 2. Real generation CLI call
    cmd_run = [
        sys.executable,
        "scripts/generate_instruction_dataset.py",
        "--input",
        str(in_dir),
        "--output-dir",
        str(out_dir),
        "--seed",
        "42",
    ]
    res_run = subprocess.run(cmd_run, capture_output=True, text=True)
    assert res_run.returncode == 0
    assert "GENERATION COMPLETE" in res_run.stderr or "GENERATION COMPLETE" in res_run.stdout

    assert (out_dir / "combined_candidates.jsonl").is_file()
    assert (out_dir / "reports" / "generation_report.md").is_file()
