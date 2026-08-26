"""
Test Suite for Phase 2.3.5: Pilot Dataset Assembly & Validation.
Verifies pilot configuration, domain/difficulty coverage, pipeline integration,
provenance preservation, cross-split leakage detection, deterministic reproducibility,
readiness evaluations, and shortage handling.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.dataset.pilot import (
    PilotAssembler,
    PilotManifest,
    PilotReadinessReport,
    PilotResult,
    ReadinessDimensionScore,
)
from src.dataset.schema import DatasetRecord, Message, RecordMetadata, Role


# ============================================================================
# 1. CONFIGURATION & INITIALIZATION TESTS
# ============================================================================

def test_pilot_assembler_initialization_valid_config() -> None:
    """Verifies that PilotAssembler initializes with valid authoritative configs."""
    assembler = PilotAssembler()
    assert len(assembler.domain_targets) == 13
    assert len(assembler.difficulty_targets) == 4
    assert assembler.pilot_cfg.get("target_count") == 1000
    assert assembler.pilot_cfg.get("seed") == 42


def test_pilot_assembler_invalid_config_raises() -> None:
    """Verifies that missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        PilotAssembler(config_path="configs/non_existent.yaml")


# ============================================================================
# 2. CANDIDATE GENERATION & COVERAGE TESTS
# ============================================================================

def test_candidate_generation_domain_and_difficulty_coverage() -> None:
    """Verifies candidate pool covers all 13 domains and 4 difficulty tiers."""
    assembler = PilotAssembler()
    candidates = assembler.generate_candidate_pool(
        target_candidate_count=50,
        seed=42,
        include_fixtures=False,
    )
    assert len(candidates) >= 40

    domains_present = {r.metadata.domain for r in candidates}
    difficulties_present = {r.metadata.difficulty for r in candidates}

    assert len(domains_present) == 13, f"Missing domains: {set(assembler.domain_targets.keys()) - domains_present}"
    assert len(difficulties_present) == 4, f"Missing difficulties: {set(assembler.difficulty_targets.keys()) - difficulties_present}"


def test_candidate_generation_provenance_attached() -> None:
    """Verifies that every generated candidate has valid immutable provenance."""
    assembler = PilotAssembler()
    candidates = assembler.generate_candidate_pool(target_candidate_count=20, seed=42)
    for r in candidates:
        assert r.metadata.provenance is not None
        assert r.metadata.provenance.source_type in ["synthetic", "human_authored", "documentation", "existing_dataset"]
        assert r.metadata.provenance.source_id is not None


# ============================================================================
# 3. END-TO-END PILOT ASSEMBLY & VALIDATION TESTS
# ============================================================================

def test_pilot_assembly_small_target(tmp_path: Path) -> None:
    """Verifies end-to-end assembly on a scaled target with output persistence."""
    assembler = PilotAssembler()
    out_dir = tmp_path / "pilot_test_v1"

    result = assembler.assemble(
        target_count=30,
        seed=42,
        version="test-pilot-v1",
        output_dir=out_dir,
        candidate_multiplier=1.2,
        save_outputs=True,
    )

    assert isinstance(result, PilotResult)
    assert result.manifest.pilot_version == "test-pilot-v1"
    assert result.manifest.target_count == 30
    assert result.manifest.actual_count > 0
    assert result.manifest.train_count > 0

    # Verify directory structure
    assert (out_dir / "raw" / "pilot_candidates.jsonl").is_file()
    assert (out_dir / "processed" / "train.jsonl").is_file()
    assert (out_dir / "processed" / "validation.jsonl").is_file()
    assert (out_dir / "processed" / "test.jsonl").is_file()
    assert (out_dir / "manifests" / "pilot_manifest.json").is_file()
    assert (out_dir / "reports" / "pilot_readiness_report.json").is_file()
    assert (out_dir / "reports" / "pilot_readiness_report.md").is_file()

    # Verify no cross-split leakage
    assert not result.readiness_report.leakage_detected
    assert result.is_ready


def test_pilot_quality_and_readiness_dimensions(tmp_path: Path) -> None:
    """Verifies that all 9 readiness dimensions are evaluated and reported."""
    assembler = PilotAssembler()
    result = assembler.assemble(
        target_count=20,
        seed=42,
        output_dir=tmp_path / "pilot_dim_test",
        save_outputs=False,
    )

    dims = {d.dimension: d for d in result.readiness_report.dimensions}
    expected_dims = {
        "schema_validity",
        "domain_coverage",
        "difficulty_coverage",
        "quality",
        "deduplication",
        "provenance",
        "split_integrity",
        "leakage",
        "source_diversity",
    }
    assert expected_dims.issubset(set(dims.keys()))
    assert dims["leakage"].status == "PASS"
    assert dims["provenance"].status == "PASS"


# ============================================================================
# 4. LEAKAGE DETECTION & FAILING INTEGRITY TEST
# ============================================================================

def test_deliberate_leakage_detection(monkeypatch: Any) -> None:
    """Verifies that cross-split content overlap triggers leakage detection and FAIL status."""
    assembler = PilotAssembler()

    # Mock splitter to force duplicate records into train and test
    def mock_split(records: List[DatasetRecord]) -> Any:
        from src.dataset.splitter import SplitResult
        shared_record = records[0]
        return SplitResult(
            train=records,
            validation=[records[1]] if len(records) > 1 else [],
            test=[shared_record],  # Intentional duplicate in test
            split_summary={},
        )

    monkeypatch.setattr(assembler.splitter, "split", mock_split)

    result = assembler.assemble(
        target_count=10,
        seed=42,
        save_outputs=False,
    )

    assert result.readiness_report.leakage_detected is True
    leakage_dim = next(d for d in result.readiness_report.dimensions if d.dimension == "leakage")
    assert leakage_dim.status == "FAIL"
    assert result.readiness_report.overall_status == "FAIL"
    assert result.is_ready is False


# ============================================================================
# 5. DETERMINISTIC REPRODUCIBILITY TEST
# ============================================================================

def test_pilot_deterministic_reproducibility(tmp_path: Path) -> None:
    """Verifies that running the pilot twice with identical seed/config yields identical outputs."""
    assembler = PilotAssembler()

    run1 = assembler.assemble(
        target_count=25,
        seed=12345,
        version="pilot-rep-v1",
        output_dir=tmp_path / "run1",
        save_outputs=False,
    )

    run2 = assembler.assemble(
        target_count=25,
        seed=12345,
        version="pilot-rep-v1",
        output_dir=tmp_path / "run2",
        save_outputs=False,
    )

    # Hashes and record counts must match exactly
    run1_hashes = [r.canonical_content_hash() for r in run1.train_records]
    run2_hashes = [r.canonical_content_hash() for r in run2.train_records]
    assert run1_hashes == run2_hashes

    assert run1.manifest.actual_count == run2.manifest.actual_count
    assert run1.manifest.train_count == run2.manifest.train_count
    assert run1.manifest.validation_count == run2.manifest.validation_count
    assert run1.manifest.test_count == run2.manifest.test_count


# ============================================================================
# 6. SHORTAGE & OVERSAMPLING HANDLING TESTS
# ============================================================================

def test_shortage_handling_without_silent_fabrication() -> None:
    """Verifies that shortage is explicitly recorded when candidate data is scarce."""
    assembler = PilotAssembler()

    # Force candidate generator to return only 5 records when 100 are requested
    def mock_candidates(*args: Any, **kwargs: Any) -> List[DatasetRecord]:
        return assembler.generator.generate(
            domain="programming",
            topic="python",
            task_type="coding",
            difficulty="intermediate",
            number_of_examples=5,
        )

    assembler.generate_candidate_pool = mock_candidates  # type: ignore

    result = assembler.assemble(
        target_count=100,
        seed=42,
        save_outputs=False,
    )

    assert result.manifest.actual_count <= 5
    assert len(result.readiness_report.shortages) > 0


# ============================================================================
# 7. CLI SCRIPT EXECUTION TEST
# ============================================================================

def test_cli_run_pilot_script(tmp_path: Path) -> None:
    """Verifies CLI execution of scripts/run_pilot.py."""
    out_dir = tmp_path / "cli_pilot_run"
    cmd = [
        sys.executable,
        "scripts/run_pilot.py",
        "--count", "20",
        "--seed", "42",
        "--version", "cli-test-v1",
        "--output-dir", str(out_dir),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI stderr: {proc.stderr}"
    assert "Qwen Pilot Dataset Assembly & Validation Engine" in proc.stdout
    assert "Pilot Execution Summary" in proc.stdout
    assert "Leakage:            NONE DETECTED ✅" in proc.stdout

    # Verify manifest generated on disk
    manifest_path = out_dir / "manifests" / "pilot_manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["pilot_version"] == "cli-test-v1"
