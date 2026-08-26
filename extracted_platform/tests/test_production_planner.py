"""
Test suite for Production Dataset Specification & Scaling Architecture (Phase 3.1).
Tests quota apportionment algorithms, 2D matrix distribution, batch planning,
checkpoint manager, manifest lifecycle, and dry-run CLI planner.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.dataset.production import (
    BatchCheckpoint,
    BatchPlan,
    BatchStatus,
    DatasetFreezeState,
    DomainDifficultyMatrix,
    ProductionCheckpointManager,
    ProductionManifest,
    ProductionPlan,
    ProductionPlanner,
    apportion_quotas_hare_niemeyer,
    build_domain_difficulty_matrix,
    derive_batch_seed,
)


@pytest.fixture
def sample_domain_weights():
    return {
        "programming": 0.182,
        "software_engineering": 0.091,
        "cybersecurity": 0.136,
        "linux_systems": 0.091,
        "networking": 0.073,
        "ai_ml": 0.073,
        "mathematics": 0.045,
        "science": 0.045,
        "psychology": 0.045,
        "human_behavior": 0.045,
        "reasoning": 0.064,
        "technology": 0.045,
        "general_knowledge": 0.065,
    }


@pytest.fixture
def sample_diff_weights():
    return {
        "beginner": 0.25,
        "intermediate": 0.40,
        "advanced": 0.25,
        "expert": 0.10,
    }


# ============================================================================
# 1. QUOTA APPORTIONMENT & EXACT SUM PRESERVATION TESTS
# ============================================================================

@pytest.mark.parametrize("target_count", [100, 1000, 10000, 25000, 50000, 100000])
def test_domain_quotas_sum_exactly_to_target(sample_domain_weights, target_count):
    breakdowns = apportion_quotas_hare_niemeyer(target_count, sample_domain_weights)
    total_quota = sum(b.integer_quota for b in breakdowns)
    assert total_quota == target_count
    assert len(breakdowns) == len(sample_domain_weights)
    for b in breakdowns:
        assert b.integer_quota >= 0
        assert b.category in sample_domain_weights


@pytest.mark.parametrize("target_count", [100, 1000, 10000, 25000, 50000, 100000])
def test_difficulty_quotas_sum_exactly_to_target(sample_diff_weights, target_count):
    breakdowns = apportion_quotas_hare_niemeyer(target_count, sample_diff_weights)
    total_quota = sum(b.integer_quota for b in breakdowns)
    assert total_quota == target_count
    assert len(breakdowns) == 4


def test_deterministic_rounding_apportionment(sample_domain_weights):
    b1 = apportion_quotas_hare_niemeyer(10000, sample_domain_weights)
    b2 = apportion_quotas_hare_niemeyer(10000, sample_domain_weights)
    assert [b.to_dict() for b in b1] == [b.to_dict() for b in b2]


def test_invalid_apportionment_inputs():
    with pytest.raises(ValueError, match="Target total count must be positive"):
        apportion_quotas_hare_niemeyer(0, {"a": 0.5, "b": 0.5})

    with pytest.raises(ValueError, match="Target total count must be positive"):
        apportion_quotas_hare_niemeyer(-100, {"a": 0.5, "b": 0.5})

    with pytest.raises(ValueError, match="Weights dictionary must not be empty"):
        apportion_quotas_hare_niemeyer(100, {})

    with pytest.raises(ValueError, match="Sum of weights must be positive"):
        apportion_quotas_hare_niemeyer(100, {"a": 0.0, "b": 0.0})


# ============================================================================
# 2. 2D DOMAIN x DIFFICULTY MATRIX TESTS
# ============================================================================

@pytest.mark.parametrize("target_count", [50, 500, 1000, 10000, 25000, 100000])
def test_domain_difficulty_matrix_properties(sample_domain_weights, sample_diff_weights, target_count):
    matrix_obj = build_domain_difficulty_matrix(target_count, sample_domain_weights, sample_diff_weights)

    # 1. Grand total must equal target_count exactly
    assert matrix_obj.grand_total == target_count

    # 2. Each row total must equal the sum of cell values for that domain
    for dom, row_total in matrix_obj.row_totals.items():
        cells = matrix_obj.matrix[dom]
        assert sum(cells.values()) == row_total

    # 3. Sum of row totals must equal target_count
    assert sum(matrix_obj.row_totals.values()) == target_count

    # 4. Sum of column totals must equal target_count
    assert sum(matrix_obj.col_totals.values()) == target_count


# ============================================================================
# 3. BATCH PLANNING & DETERMINISTIC SEED TESTS
# ============================================================================

def test_deterministic_batch_seed_generation():
    seed1 = derive_batch_seed(42, 1)
    seed2 = derive_batch_seed(42, 1)
    seed3 = derive_batch_seed(42, 2)
    seed4 = derive_batch_seed(100, 1)

    assert seed1 == seed2
    assert seed1 != seed3
    assert seed1 != seed4
    assert isinstance(seed1, int)
    assert seed1 > 0


def test_batch_planning_counts_and_partials(tmp_path, sample_domain_weights, sample_diff_weights):
    cfg_file = tmp_path / "dataset.yaml"
    cfg_file.write_text(
        yaml.dump({
            "domain_targets": sample_domain_weights,
            "difficulty": {"targets": sample_diff_weights},
            "production": {
                "target_count": 10000,
                "candidate_multiplier": 1.20,
                "batch_size": 500,
                "seed": 42,
                "version": "dataset-v1.0",
            },
        })
    )

    planner = ProductionPlanner(config_path=cfg_file)
    plan = planner.plan(target_count=10000, candidate_multiplier=1.20, batch_size=500)

    assert plan.target_count == 10000
    assert plan.candidate_target == 12000
    assert plan.batch_size == 500
    assert plan.estimated_batches == 24
    assert len(plan.batch_plans) == 24

    # Verify all batch IDs are sequential
    for i, bp in enumerate(plan.batch_plans, 1):
        assert bp.batch_id == f"dataset-v1.0-batch-{i:04d}"
        assert bp.batch_index == i
        assert bp.candidate_target == 500


def test_custom_scale_targets(tmp_path, sample_domain_weights, sample_diff_weights):
    cfg_file = tmp_path / "dataset.yaml"
    cfg_file.write_text(
        yaml.dump({
            "domain_targets": sample_domain_weights,
            "difficulty": {"targets": sample_diff_weights},
            "production": {"target_count": 10000},
        })
    )

    planner = ProductionPlanner(config_path=cfg_file)

    # Test 25K target
    plan_25k = planner.plan(target_count=25000, candidate_multiplier=1.20, batch_size=1000)
    assert plan_25k.target_count == 25000
    assert plan_25k.candidate_target == 30000
    assert plan_25k.estimated_batches == 30

    # Test 100K target
    plan_100k = planner.plan(target_count=100000, candidate_multiplier=1.20, batch_size=2000)
    assert plan_100k.target_count == 100000
    assert plan_100k.candidate_target == 120000
    assert plan_100k.estimated_batches == 60


# ============================================================================
# 4. CONFIGURATION VALIDATION TESTS
# ============================================================================

def test_production_planner_invalid_config_raises(tmp_path):
    # Invalid domain targets sum != 1.0
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(
        yaml.dump({
            "domain_targets": {"programming": 0.5},
            "difficulty": {"targets": {"beginner": 0.5, "intermediate": 0.5}},
        })
    )
    with pytest.raises(ValueError, match="Domain targets must sum to 1.00"):
        ProductionPlanner(config_path=bad_cfg)


def test_production_planner_invalid_arguments(tmp_path, sample_domain_weights, sample_diff_weights):
    cfg_file = tmp_path / "dataset.yaml"
    cfg_file.write_text(
        yaml.dump({
            "domain_targets": sample_domain_weights,
            "difficulty": {"targets": sample_diff_weights},
        })
    )
    planner = ProductionPlanner(config_path=cfg_file)

    with pytest.raises(ValueError, match="Target count must be a positive integer"):
        planner.plan(target_count=0)

    with pytest.raises(ValueError, match="Candidate multiplier must be >= 1.0"):
        planner.plan(target_count=1000, candidate_multiplier=0.8)

    with pytest.raises(ValueError, match="Batch size must be a positive integer"):
        planner.plan(target_count=1000, batch_size=0)


# ============================================================================
# 5. CHECKPOINT ARCHITECTURE & RESUME TESTS
# ============================================================================

def test_checkpoint_manager_lifecycle(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    mgr = ProductionCheckpointManager(ckpt_dir)

    # Initial state
    assert len(mgr.list_checkpoints()) == 0
    assert not mgr.is_batch_completed("batch-0001")
    assert mgr.should_process_batch("batch-0001")

    # Create and save a pending checkpoint
    ckpt1 = BatchCheckpoint(
        batch_id="batch-0001",
        batch_index=1,
        status=BatchStatus.PENDING.value,
        seed=12345,
        requested_count=500,
    )
    mgr.save_checkpoint(ckpt1)

    assert not mgr.is_batch_completed("batch-0001")
    assert mgr.should_process_batch("batch-0001")

    # Mark as completed
    ckpt1.status = BatchStatus.COMPLETED.value
    ckpt1.generated_count = 500
    ckpt1.accepted_count = 490
    mgr.save_checkpoint(ckpt1)

    assert mgr.is_batch_completed("batch-0001")
    assert not mgr.should_process_batch("batch-0001", force=False)
    assert mgr.should_process_batch("batch-0001", force=True)
    assert len(mgr.get_completed_batches()) == 1


def test_checkpoint_failure_handling(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    mgr = ProductionCheckpointManager(ckpt_dir)

    ckpt = BatchCheckpoint(
        batch_id="batch-0002",
        batch_index=2,
        status=BatchStatus.GENERATING.value,
        seed=67890,
        requested_count=500,
    )
    mgr.save_checkpoint(ckpt)

    # Test failure with fail_fast=False
    mgr.handle_batch_failure("batch-0002", ValueError("Generation timeout"), fail_fast=False)
    failed_ckpts = mgr.get_failed_batches()
    assert len(failed_ckpts) == 1
    assert failed_ckpts[0].batch_id == "batch-0002"
    assert "Generation timeout" in failed_ckpts[0].error_message

    # Test failure with fail_fast=True
    with pytest.raises(RuntimeError, match="Batch 'batch-0002' failed with fail_fast=True"):
        mgr.handle_batch_failure("batch-0002", RuntimeError("Fatal API error"), fail_fast=True)


# ============================================================================
# 6. PRODUCTION MANIFEST & FREEZE STATES TESTS
# ============================================================================

def test_production_manifest_freeze_lifecycle(tmp_path, sample_domain_weights, sample_diff_weights):
    manifest = ProductionManifest(
        dataset_version="dataset-v1.0",
        target_count=10000,
        candidate_target=12000,
        seed=42,
        domain_targets=sample_domain_weights,
        difficulty_targets=sample_diff_weights,
        batch_size=500,
        batch_count=24,
    )

    assert manifest.status == DatasetFreezeState.PLANNED.value

    # State transitions
    manifest.transition_state(DatasetFreezeState.GENERATING)
    assert manifest.status == DatasetFreezeState.GENERATING.value

    manifest.transition_state(DatasetFreezeState.VALIDATING)
    assert manifest.status == DatasetFreezeState.VALIDATING.value

    manifest.transition_state(DatasetFreezeState.READY)
    assert manifest.status == DatasetFreezeState.READY.value

    manifest.transition_state(DatasetFreezeState.FROZEN)
    assert manifest.status == DatasetFreezeState.FROZEN.value

    # Invalid state transition raises ValueError
    with pytest.raises(ValueError, match="Invalid freeze state"):
        manifest.transition_state("INVALID_STATE")

    # Save and reload
    save_path = tmp_path / "manifest.json"
    manifest.save(save_path)
    loaded = ProductionManifest.load(save_path)
    assert loaded.dataset_version == "dataset-v1.0"
    assert loaded.status == DatasetFreezeState.FROZEN.value


# ============================================================================
# 7. DRY-RUN PLANNER & CLI TESTS
# ============================================================================

def test_dry_run_planner_generates_reports(tmp_path, sample_domain_weights, sample_diff_weights):
    cfg_file = tmp_path / "dataset.yaml"
    cfg_file.write_text(
        yaml.dump({
            "domain_targets": sample_domain_weights,
            "difficulty": {"targets": sample_diff_weights},
            "production": {
                "target_count": 10000,
                "candidate_multiplier": 1.20,
                "batch_size": 500,
                "seed": 42,
                "version": "dataset-v1.0",
                "output_dir": str(tmp_path / "prod_out"),
            },
        })
    )

    planner = ProductionPlanner(config_path=cfg_file)
    plan = planner.plan()

    reports_dir = tmp_path / "prod_out" / "reports"
    json_path, md_path = plan.save_reports(reports_dir)

    assert json_path.is_file()
    assert md_path.is_file()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["target_count"] == 10000
    assert data["candidate_target"] == 12000
    assert len(data["domain_quotas"]) == 13
    assert len(data["difficulty_quotas"]) == 4

    md_content = md_path.read_text(encoding="utf-8")
    assert "# Production Dataset Specification & Plan" in md_content
    assert "`10,000` examples" in md_content


def test_cli_plan_production_dry_run(tmp_path):
    out_dir = tmp_path / "cli_prod"
    cmd = [
        sys.executable,
        "scripts/plan_production.py",
        "--target", "10000",
        "--seed", "42",
        "--output-dir", str(out_dir),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    assert result.returncode == 0
    assert "SUCCESS: Production plan established for 10,000 examples" in result.stdout
    assert (out_dir / "reports" / "production_plan.json").is_file()
    assert (out_dir / "reports" / "production_plan.md").is_file()
    assert (out_dir / "manifests" / "production_manifest.json").is_file()
