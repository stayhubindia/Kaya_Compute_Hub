"""
Unit and Integration Tests for Phase 3.2 Production Dataset Generation Engine.
Tests batch generation, template diversity, provenance tracking, atomic persistence,
checkpoint recovery, deduplication, global mixing, deficit tracking, and CLI utilities.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from src.dataset.generator import GenerationRequest, SampleSyntheticGenerator, SyntheticGeneratorInterface
from src.dataset.production import (
    BatchCheckpoint,
    BatchPlan,
    BatchStatus,
    DatasetFreezeState,
    ProductionCheckpointManager,
    ProductionManifest,
    ProductionPlan,
    ProductionPlanner,
    derive_batch_seed,
)
from src.dataset.production_generator import (
    BatchGenerationResult,
    BatchYieldMetrics,
    GlobalGenerationResult,
    ProductionGenerationEngine,
    TemplateUsageStats,
    atomic_write_jsonl,
)
from src.dataset.schema import DatasetRecord, DifficultyLevel, Role, SourceType, TaskType


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config_path() -> Path:
    return Path("configs/dataset.yaml").resolve()


@pytest.fixture
def templates_path() -> Path:
    return Path("configs/domain_templates.yaml").resolve()


@pytest.fixture
def sources_path() -> Path:
    return Path("configs/sources.yaml").resolve()


@pytest.fixture
def engine(config_path, templates_path, sources_path) -> ProductionGenerationEngine:
    return ProductionGenerationEngine(
        config_path=config_path,
        templates_path=templates_path,
        sources_path=sources_path,
    )


# ============================================================================
# 1. ATOMIC WRITE & UTILITY TESTS
# ============================================================================

def test_atomic_write_jsonl(tmp_path: Path):
    """Verifies atomic write functionality and SHA-256 computation."""
    generator = SampleSyntheticGenerator()
    req = GenerationRequest(
        domain="programming",
        topic="python",
        task_type="coding",
        difficulty="intermediate",
        number_of_examples=5,
        seed=42,
    )
    records = generator.generate_batch(req).records
    target_file = tmp_path / "subdir" / "test_records.jsonl"

    saved_path, checksum = atomic_write_jsonl(records, target_file)

    assert saved_path.is_file()
    assert len(checksum) == 64  # Valid sha256 hex length

    loaded_records = []
    with open(saved_path, "r", encoding="utf-8") as f:
        for line in f:
            loaded_records.append(DatasetRecord.model_validate_json(line))

    assert len(loaded_records) == 5
    assert loaded_records[0].metadata.domain == "programming"


# ============================================================================
# 2. BATCH GENERATION & PROVENANCE TESTS
# ============================================================================

def test_generate_single_batch(engine: ProductionGenerationEngine, tmp_path: Path):
    """Verifies single batch execution, atomic writes, yield metrics, and provenance."""
    planner = engine.planner
    plan = planner.plan(target_count=100, batch_size=25, candidate_multiplier=1.2, seed=42)
    batch_plan = plan.batch_plans[0]

    ckpt_mgr = ProductionCheckpointManager(tmp_path / "checkpoints")
    result = engine.generate_batch(
        batch_plan=batch_plan,
        checkpoint_mgr=ckpt_mgr,
        output_dir=tmp_path,
        dataset_version="test-v1",
    )

    assert result.status == BatchStatus.COMPLETED.value, f"Batch failed with error: {result.error_message}"

    assert result.batch_id == batch_plan.batch_id
    assert result.generated_count == batch_plan.candidate_target
    assert result.accepted_count > 0
    assert result.raw_file is not None and Path(result.raw_file).is_file()
    assert result.processed_file is not None and Path(result.processed_file).is_file()
    assert result.yield_metrics is not None
    assert result.yield_metrics.final_batch_yield_pct > 0.0

    # Verify Checkpoint
    ckpt = ckpt_mgr.load_checkpoint(batch_plan.batch_id)
    assert ckpt is not None
    assert ckpt.status == BatchStatus.COMPLETED.value
    assert ckpt.generated_count == result.generated_count
    assert ckpt.accepted_count == result.accepted_count

    # Verify Record Provenance
    records = []
    with open(result.processed_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(DatasetRecord.model_validate_json(line))

    for r in records:
        assert r.metadata.provenance is not None
        assert r.metadata.provenance.source_type == SourceType.SYNTHETIC.value or r.metadata.provenance.source_type == SourceType.SYNTHETIC
        assert r.metadata.provenance.generator == "sample_synthetic_generator"
        assert r.metadata.provenance.created_at is not None
        assert batch_plan.batch_id in r.metadata.provenance.source_id




def test_batch_resume_skips_completed(engine: ProductionGenerationEngine, tmp_path: Path):
    """Verifies that an already completed batch is skipped on resume unless force=True."""
    planner = engine.planner
    plan = planner.plan(target_count=50, batch_size=25, seed=42)
    bp = plan.batch_plans[0]
    ckpt_mgr = ProductionCheckpointManager(tmp_path / "checkpoints")

    # First run
    res1 = engine.generate_batch(bp, ckpt_mgr, tmp_path)
    assert res1.status == BatchStatus.COMPLETED.value

    # Second run without force (should resume and return existing)
    res2 = engine.generate_batch(bp, ckpt_mgr, tmp_path, force=False)
    assert res2.status == BatchStatus.COMPLETED.value
    assert res2.processed_sha256 == res1.processed_sha256

    # Third run with force=True (should re-generate)
    res3 = engine.generate_batch(bp, ckpt_mgr, tmp_path, force=True)
    assert res3.status == BatchStatus.COMPLETED.value
    assert res3.processed_sha256 == res1.processed_sha256


# ============================================================================
# 3. FAILURE HANDLING & RECOVERY TESTS
# ============================================================================

class FailingGenerator(SyntheticGeneratorInterface):
    """Fault-injection generator for testing failure modes."""
    def generate(self, *args, **kwargs):
        raise RuntimeError("Simulated API rate limit or synthesis failure.")

    def generate_batch(self, *args, **kwargs):
        raise RuntimeError("Simulated API rate limit or synthesis failure.")

    def generate_from_template(self, *args, **kwargs):
        raise RuntimeError("Simulated template generation failure.")


def test_batch_failure_non_fail_fast(config_path, templates_path, sources_path, tmp_path: Path):
    """Verifies that batch failure records FAILED status in checkpoint when fail_fast=False."""
    failing_engine = ProductionGenerationEngine(
        config_path=config_path,
        templates_path=templates_path,
        sources_path=sources_path,
        generator_backend=FailingGenerator(),
    )
    plan = failing_engine.planner.plan(target_count=50, batch_size=25, seed=42)
    bp = plan.batch_plans[0]
    ckpt_mgr = ProductionCheckpointManager(tmp_path / "checkpoints")

    res = failing_engine.generate_batch(bp, ckpt_mgr, tmp_path, fail_fast=False)
    assert res.status == BatchStatus.FAILED.value
    assert "Simulated" in str(res.error_message)

    ckpt = ckpt_mgr.load_checkpoint(bp.batch_id)
    assert ckpt is not None
    assert ckpt.status == BatchStatus.FAILED.value
    assert "Simulated" in str(ckpt.error_message)


def test_batch_failure_fail_fast_raises(config_path, templates_path, sources_path, tmp_path: Path):
    """Verifies that fail_fast=True immediately raises a RuntimeError."""
    failing_engine = ProductionGenerationEngine(
        config_path=config_path,
        templates_path=templates_path,
        sources_path=sources_path,
        generator_backend=FailingGenerator(),
    )
    plan = failing_engine.planner.plan(target_count=50, batch_size=25, seed=42)
    bp = plan.batch_plans[0]
    ckpt_mgr = ProductionCheckpointManager(tmp_path / "checkpoints")

    with pytest.raises(RuntimeError) as excinfo:
        failing_engine.generate_batch(bp, ckpt_mgr, tmp_path, fail_fast=True)
    assert "fail_fast=True" in str(excinfo.value)


# ============================================================================
# 4. GLOBAL GENERATION, MIXING & DEFICIT TESTS
# ============================================================================

def test_generate_all_staged_run(engine: ProductionGenerationEngine, tmp_path: Path):
    """Verifies end-to-end staged generation run (100 target, 4 batches)."""
    out_dir = tmp_path / "prod_staged"
    result = engine.generate_all(
        target_count=100,
        batch_size=25,
        candidate_multiplier=1.2,
        seed=42,
        version="staged-v1.0",
        output_dir=out_dir,
        max_batches=4,
    )

    assert result.dataset_version == "staged-v1.0"
    assert result.target_count == 100
    assert result.total_generated > 0
    assert result.final_selected_count > 0
    assert result.candidate_dataset_file is not None
    assert Path(result.candidate_dataset_file).is_file()
    assert result.manifest_file is not None
    assert Path(result.manifest_file).is_file()

    # Verify Manifest State
    manifest = ProductionManifest.load(result.manifest_file)
    assert manifest.status == DatasetFreezeState.VALIDATING.value
    assert manifest.actual_final_count == result.final_selected_count
    assert "candidate_dataset.jsonl" in manifest.checksums

    # Verify Reports
    for r_key, r_path in result.report_files.items():
        assert Path(r_path).is_file()

    # Verify Quality Metrics
    assert result.quality_summary["mean_score"] >= 0.85


def test_dry_run_generation(engine: ProductionGenerationEngine, tmp_path: Path):
    """Verifies dry-run generation creates plan and manifest without synthesizing records."""
    out_dir = tmp_path / "dry_run"
    result = engine.generate_all(
        target_count=10000,
        batch_size=500,
        candidate_multiplier=1.2,
        seed=42,
        version="dryrun-v1.0",
        output_dir=out_dir,
        dry_run=True,
    )

    assert result.total_generated == 0
    assert result.final_selected_count == 0
    assert result.manifest_file is not None
    assert Path(result.manifest_file).is_file()

    manifest = ProductionManifest.load(result.manifest_file)
    assert manifest.status == DatasetFreezeState.PLANNED.value
    assert manifest.target_count == 10000


def test_deterministic_reproducibility(engine: ProductionGenerationEngine, tmp_path: Path):
    """Verifies that two runs with identical seeds produce identical checksums."""
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    res1 = engine.generate_all(target_count=50, batch_size=25, seed=999, output_dir=dir1, max_batches=2)
    res2 = engine.generate_all(target_count=50, batch_size=25, seed=999, output_dir=dir2, max_batches=2)

    assert res1.candidate_dataset_sha256 == res2.candidate_dataset_sha256
    assert res1.final_selected_count == res2.final_selected_count


# ============================================================================
# 5. CLI UTILITY INTEGRATION TESTS
# ============================================================================

def test_cli_dry_run(tmp_path: Path):
    """Verifies scripts/generate_production.py execution in --dry-run mode."""
    out_dir = tmp_path / "cli_dry"
    cmd = [
        sys.executable,
        "scripts/generate_production.py",
        "--dry-run",
        "--target", "1000",
        "--output-dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "DRY-RUN MODE" in proc.stdout
    assert (out_dir / "manifests" / "production_manifest.json").is_file()


def test_cli_staged_run(tmp_path: Path):
    """Verifies scripts/generate_production.py execution with staged batch limits."""
    out_dir = tmp_path / "cli_staged"
    cmd = [
        sys.executable,
        "scripts/generate_production.py",
        "--target", "60",
        "--batch-size", "20",
        "--max-batches", "2",
        "--version", "cli-test-v1",
        "--output-dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "SUCCESS: Production generation run completed." in proc.stdout
    assert (out_dir / "processed" / "candidate_dataset.jsonl").is_file()
    assert (out_dir / "reports" / "generation_report.json").is_file()
