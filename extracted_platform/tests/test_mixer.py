"""
Comprehensive Test Suite for Dataset Mixing & Balancing Engine (Phase 2.3.4).
Tests configuration validation, proportional & balanced strategies, deterministic selection,
shortage tracking, oversampling & undersampling, provenance preservation,
report generation, CLI execution, and Phase 2.2 pipeline integration.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest
import yaml

from src.dataset.mixer import (
    BalancedMixingStrategy,
    DatasetMixer,
    DistributionReport,
    MixingRequest,
    MixingResult,
    OversamplingDetail,
    ProportionalMixingStrategy,
    ShortageDetail,
)
from src.dataset.pipeline import DatasetPipeline
from src.dataset.schema import (
    DatasetRecord,
    DifficultyLevel,
    Message,
    ProvenanceInfo,
    RecordMetadata,
    Role,
    SourceType,
    TaskType,
)


# Helper function to create synthetic test records
def _create_mock_record(
    domain: str,
    difficulty: str,
    task_type: str = "coding",
    source_type: str = "synthetic",
    source: str = "synthetic_generator",
    source_id: str = "rec_001",
    unique_idx: int = 1,
) -> DatasetRecord:
    return DatasetRecord(
        messages=[
            Message(role=Role.USER, content=f"Sample prompt for {domain} #{unique_idx}?"),
            Message(role=Role.ASSISTANT, content=f"Sample response for {domain} difficulty {difficulty} #{unique_idx}."),
        ],
        metadata=RecordMetadata(
            domain=domain,
            topic="general",
            task_type=task_type,
            difficulty=difficulty,
            quality_score=0.95,
            source=source,
            source_type=source_type,
            source_id=f"{source_id}_{unique_idx}",
            created_at="2026-08-11T12:00:00Z",
            generator="sample_test_generator" if source_type == "synthetic" else None,
            generator_version="1.0.0" if source_type == "synthetic" else None,
        ),
    )


@pytest.fixture
def sample_candidates_pool() -> List[DatasetRecord]:
    """Generates a rich, controlled candidate pool across multiple domains and difficulties."""
    records: List[DatasetRecord] = []
    domains = [
        "programming", "software_engineering", "cybersecurity", "linux_systems",
        "networking", "ai_ml", "mathematics", "science", "psychology",
        "human_behavior", "reasoning", "technology", "general_knowledge",
    ]
    difficulties = ["beginner", "intermediate", "advanced", "expert"]

    idx = 1
    for dom in domains:
        # Vary candidate count per domain
        count = 10 if dom in ["programming", "cybersecurity", "linux_systems"] else 4
        for i in range(count):
            diff = difficulties[i % len(difficulties)]
            src_type = "synthetic" if i % 2 == 0 else "human_authored"
            records.append(
                _create_mock_record(
                    domain=dom,
                    difficulty=diff,
                    source_type=src_type,
                    source=f"{src_type}_source",
                    unique_idx=idx,
                )
            )
            idx += 1
    return records


# ============================================================================
# 1. CONFIGURATION & VALIDATION TESTS
# ============================================================================

def test_mixer_initialization_valid_config():
    mixer = DatasetMixer(config_path="configs/dataset.yaml")
    assert mixer.domain_targets is not None
    assert len(mixer.domain_targets) == 13
    assert abs(sum(mixer.domain_targets.values()) - 1.0) < 1e-4
    assert mixer.difficulty_targets["intermediate"] == 0.40


def test_mixer_initialization_invalid_domain_sum(tmp_path: Path):
    bad_config = {
        "domain_targets": {"programming": 0.5, "linux_systems": 0.3},  # Sum = 0.8
        "difficulty": {"targets": {"beginner": 0.25, "intermediate": 0.40, "advanced": 0.25, "expert": 0.10}},
    }
    cfg_file = tmp_path / "bad_dataset.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(bad_config, f)

    with pytest.raises(ValueError, match="Authoritative domain_targets must sum to 1.000"):
        DatasetMixer(config_path=cfg_file)


def test_mixer_initialization_invalid_difficulty_sum(tmp_path: Path):
    bad_config = {
        "domain_targets": {
            "programming": 0.182, "software_engineering": 0.091, "cybersecurity": 0.136,
            "linux_systems": 0.091, "networking": 0.073, "ai_ml": 0.073, "mathematics": 0.045,
            "science": 0.045, "psychology": 0.045, "human_behavior": 0.045, "reasoning": 0.064,
            "technology": 0.045, "general_knowledge": 0.065,
        },
        "difficulty": {"targets": {"beginner": 0.5, "intermediate": 0.6}},  # Sum = 1.1
    }
    cfg_file = tmp_path / "bad_diff.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(bad_config, f)

    with pytest.raises(ValueError, match="Difficulty targets must sum to 1.000"):
        DatasetMixer(config_path=cfg_file)


def test_mixing_request_invalid_strategy():
    with pytest.raises(ValueError, match="Unsupported mixing strategy"):
        MixingRequest(target_count=50, strategy="quantum_random")


def test_mixing_request_invalid_target_count():
    with pytest.raises(ValueError):
        MixingRequest(target_count=0)


# ============================================================================
# 2. PROPORTIONAL & BALANCED MIXING TESTS
# ============================================================================

def test_proportional_mixing_deterministic_selection(sample_candidates_pool: List[DatasetRecord]):
    mixer = DatasetMixer(config_path="configs/dataset.yaml")

    req1 = MixingRequest(target_count=20, strategy="proportional", seed=42)
    res1 = mixer.mix(req1, candidate_records=sample_candidates_pool)

    req2 = MixingRequest(target_count=20, strategy="proportional", seed=42)
    res2 = mixer.mix(req2, candidate_records=sample_candidates_pool)

    assert len(res1.records) == len(res2.records)
    hashes1 = [r.canonical_content_hash() for r in res1.records]
    hashes2 = [r.canonical_content_hash() for r in res2.records]
    assert hashes1 == hashes2


def test_proportional_mixing_different_seeds_vary(sample_candidates_pool: List[DatasetRecord]):
    mixer = DatasetMixer(config_path="configs/dataset.yaml")

    req1 = MixingRequest(target_count=20, strategy="proportional", seed=42)
    res1 = mixer.mix(req1, candidate_records=sample_candidates_pool)

    req2 = MixingRequest(target_count=20, strategy="proportional", seed=999)
    res2 = mixer.mix(req2, candidate_records=sample_candidates_pool)

    hashes1 = [r.canonical_content_hash() for r in res1.records]
    hashes2 = [r.canonical_content_hash() for r in res2.records]
    assert hashes1 != hashes2


def test_balanced_mixing_strategy(sample_candidates_pool: List[DatasetRecord]):
    mixer = DatasetMixer(config_path="configs/dataset.yaml")
    req = MixingRequest(target_count=26, strategy="balanced", seed=42)
    res = mixer.mix(req, candidate_records=sample_candidates_pool)

    assert res.strategy == "balanced"
    assert res.selected_count > 0
    # Check that representation across domains is balanced
    domain_counts = res.domain_distribution.counts
    active_domains = [d for d, c in domain_counts.items() if c > 0]
    assert len(active_domains) >= 10


# ============================================================================
# 3. SHORTAGE & OVERSAMPLING TESTS
# ============================================================================

def test_shortage_handling_without_oversampling():
    mixer = DatasetMixer(config_path="configs/dataset.yaml")

    # Only provide 2 candidates for programming, 0 for other domains
    scarce_pool = [
        _create_mock_record("programming", "intermediate", unique_idx=1),
        _create_mock_record("programming", "intermediate", unique_idx=2),
    ]

    req = MixingRequest(target_count=50, strategy="proportional", seed=42, allow_oversampling=False)
    res = mixer.mix(req, candidate_records=scarce_pool)

    # Should not duplicate or fabricate
    assert res.selected_count == 2
    assert res.total_candidates == 2
    assert res.oversampling is None
    assert len(res.shortages) > 0

    # Verify shortage records
    dom_shortages = [s for s in res.shortages if s.dimension == "domain"]
    assert len(dom_shortages) > 0
    prog_shortage = next((s for s in dom_shortages if s.category == "programming"), None)
    assert prog_shortage is not None
    assert prog_shortage.available == 2
    assert prog_shortage.shortage > 0


def test_oversampling_enabled_controlled_duplication():
    mixer = DatasetMixer(config_path="configs/dataset.yaml")

    # 4 records in programming
    scarce_pool = [
        _create_mock_record("programming", "intermediate", unique_idx=1),
        _create_mock_record("programming", "beginner", unique_idx=2),
        _create_mock_record("programming", "advanced", unique_idx=3),
        _create_mock_record("programming", "expert", unique_idx=4),
    ]

    # Target 20 examples with oversampling
    # Force single-domain targets for testing oversampling ratio
    single_domain_targets = {"programming": 1.0}
    for d in mixer.domain_targets:
        if d != "programming":
            single_domain_targets[d] = 0.0

    req = MixingRequest(
        target_count=20,
        strategy="proportional",
        seed=42,
        allow_oversampling=True,
        domain_targets=single_domain_targets,
    )
    res = mixer.mix(req, candidate_records=scarce_pool)

    assert res.selected_count == 20
    assert res.oversampling is not None
    assert res.oversampling.oversampled_records == 16  # 20 - 4 = 16 copies
    assert res.oversampling.oversampling_ratio > 0.0

    # Verify mixing metadata on oversampled records
    oversampled_items = [r for r in res.records if r.metadata.mixing and r.metadata.mixing.get("oversampled") is True]
    assert len(oversampled_items) == 16
    for r in oversampled_items:
        assert r.metadata.mixing["strategy"] == "proportional"
        assert r.metadata.mixing["copy_index"] >= 1
        # Provenance is intact
        assert r.metadata.source_type == "synthetic"
        assert r.metadata.source == "synthetic_generator"


def test_undersampling_excess_candidates():
    mixer = DatasetMixer(config_path="configs/dataset.yaml")

    # Pool of 52 records spanning difficulties
    difficulties = ["beginner", "intermediate", "advanced", "expert"]
    pool = [_create_mock_record("programming", difficulties[i % 4], unique_idx=i) for i in range(52)]
    single_domain_targets = {"programming": 1.0}
    for d in mixer.domain_targets:
        if d != "programming":
            single_domain_targets[d] = 0.0

    req = MixingRequest(
        target_count=12,
        strategy="proportional",
        seed=42,
        allow_oversampling=False,
        domain_targets=single_domain_targets,
    )
    res = mixer.mix(req, candidate_records=pool)

    assert res.selected_count == 12
    assert res.total_candidates == 52
    assert res.discarded_count == 40


# ============================================================================
# 4. PROVENANCE PRESERVATION & MIXING METADATA
# ============================================================================

def test_provenance_preservation_across_sources():
    mixer = DatasetMixer(config_path="configs/dataset.yaml")

    pool = [
        _create_mock_record("programming", "intermediate", source_type="synthetic", source="synth_bot", unique_idx=1),
        _create_mock_record("networking", "advanced", source_type="human_authored", source="human_expert", unique_idx=2),
        _create_mock_record("linux_systems", "beginner", source_type="documentation", source="linux_man", unique_idx=3),
    ]

    req = MixingRequest(target_count=3, strategy="proportional", seed=42)
    res = mixer.mix(req, candidate_records=pool)

    for r in res.records:
        assert r.metadata.mixing is not None
        assert "strategy" in r.metadata.mixing
        assert "seed" in r.metadata.mixing
        assert "batch_id" in r.metadata.mixing
        # Verify provenance was not overwritten or destroyed
        assert r.metadata.provenance is not None
        assert r.metadata.source_type in ["synthetic", "human_authored", "documentation"]


# ============================================================================
# 5. AUDIT REPORT GENERATION & PERSISTENCE
# ============================================================================

def test_mixing_result_report_generation(sample_candidates_pool: List[DatasetRecord], tmp_path: Path):
    mixer = DatasetMixer(config_path="configs/dataset.yaml")
    req = MixingRequest(target_count=15, strategy="proportional", seed=42)
    res = mixer.mix(req, candidate_records=sample_candidates_pool)

    md_report = res.generate_markdown_report()
    assert "# Dataset Mixing & Balancing Engine Audit Report" in md_report
    assert "## 1. Domain Distribution & Target Deviations" in md_report
    assert "## 2. Difficulty Distribution" in md_report

    # Test saving reports to disk
    json_path, md_path = res.save_reports(tmp_path / "reports")
    assert json_path.is_file()
    assert md_path.is_file()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["requested_count"] == 15
        assert "domain_distribution" in data
        assert "shortages" in data

    # Test saving JSONL
    jsonl_out = tmp_path / "mixed.jsonl"
    saved_count = res.save_jsonl(jsonl_out)
    assert saved_count == len(res.records)
    assert jsonl_out.is_file()

    # Overwrite protection test
    with pytest.raises(FileExistsError):
        res.save_jsonl(jsonl_out, overwrite=False)


# ============================================================================
# 6. INTEGRATION WITH PHASE 2.2 PROCESSING PIPELINE
# ============================================================================

def test_pipeline_integration_mixed_dataset(tmp_path: Path):
    mixer = DatasetMixer(config_path="configs/dataset.yaml")

    # Ingest fixture files
    fixture_files = [
        "datasets/fixtures/synthetic.jsonl",
        "datasets/fixtures/human.jsonl",
        "datasets/fixtures/documentation.jsonl",
        "datasets/fixtures/existing_dataset.jsonl",
    ]

    req = MixingRequest(
        input_sources=fixture_files,
        target_count=8,
        strategy="proportional",
        seed=42,
        allow_oversampling=False,
    )
    mix_res = mixer.mix(req)
    assert mix_res.selected_count > 0

    # Save mixed raw dataset
    mixed_file = tmp_path / "mixed_raw.jsonl"
    mix_res.save_jsonl(mixed_file, overwrite=True)

    # Run through DatasetPipeline (Phase 2.2)
    pipeline = DatasetPipeline(config_path="configs/dataset.yaml")
    out_dir = tmp_path / "processed_pipeline"
    pipe_res = pipeline.run(input_path=mixed_file, output_dir=out_dir, save_outputs=True)

    assert pipe_res.total_raw == mix_res.selected_count
    assert pipe_res.accepted_count > 0
    assert pipe_res.split_result is not None
    assert (out_dir / "train.jsonl").is_file()
    assert (out_dir / "dataset_report.json").is_file()
    assert (out_dir / "source_report.json").is_file()


# ============================================================================
# 7. CLI SCRIPT EXECUTION TEST
# ============================================================================

def test_cli_mix_dataset_execution(tmp_path: Path):
    out_jsonl = tmp_path / "cli_mixed.jsonl"
    rep_dir = tmp_path / "cli_reports"
    pipe_dir = tmp_path / "cli_pipeline_out"

    cmd = [
        sys.executable,
        "scripts/mix_dataset.py",
        "--input", "datasets/fixtures/synthetic.jsonl",
        "--input", "datasets/fixtures/human.jsonl",
        "--count", "5",
        "--strategy", "proportional",
        "--seed", "42",
        "--output", str(out_jsonl),
        "--report-dir", str(rep_dir),
        "--run-pipeline",
        "--pipeline-output-dir", str(pipe_dir),
        "--overwrite",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI command failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"

    assert "Qwen Dataset Mixing & Balancing Engine (Phase 2.3.4)" in res.stdout
    assert "Mixing Execution Summary" in res.stdout
    assert out_jsonl.is_file()
    assert (rep_dir / "dataset_mix_report.json").is_file()
    assert (rep_dir / "dataset_mix_report.md").is_file()
    assert (pipe_dir / "train.jsonl").is_file()
