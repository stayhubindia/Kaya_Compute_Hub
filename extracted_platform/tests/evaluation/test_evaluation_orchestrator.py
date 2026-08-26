"""
Unit tests for Production Evaluation Execution & Model Comparison Orchestrator (Phase 4.7).
"""

import json
from pathlib import Path
import pytest

from src.dataset.schema import Message, Role
from src.evaluation.benchmark_cases import BenchmarkCase
from src.evaluation.execution import (
    CaseExecutionResult,
    EvaluationExecutionEngine,
    GenerationConfig,
    GPUReadinessGate,
)
from src.evaluation.experiment import (
    CaseChangeStatus,
    EvaluationManifest,
    ExperimentManager,
    ExperimentStatus,
    ModelComparisonEngine,
)


def test_generation_config_hashing_and_yaml(tmp_path: Path):
    cfg = GenerationConfig(temperature=0.0, top_p=1.0, seed=42)
    h1 = cfg.compute_hash()
    assert len(h1) == 64

    # Saving and loading
    cfg_file = tmp_path / "test_generation.yaml"
    cfg.save_to_yaml(cfg_file)
    loaded_cfg = GenerationConfig.load_from_yaml(cfg_file)
    assert loaded_cfg.compute_hash() == h1

    # Modification alters hash
    cfg2 = GenerationConfig(temperature=0.7, top_p=1.0, seed=42)
    assert cfg2.compute_hash() != h1


def test_gpu_readiness_gate():
    ok, msg = GPUReadinessGate.check()
    # On host without GPU, must cleanly return False with informative blocked message
    assert isinstance(ok, bool)
    assert isinstance(msg, str)
    if not ok:
        assert "MODEL INFERENCE BLOCKED" in msg


def test_native_chat_prompt_formatting():
    case = BenchmarkCase(
        benchmark_id="test-001",
        domain="programming",
        topic="python",
        difficulty="beginner",
        task_type="coding",
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Write hello world in Python."),
            Message(role=Role.ASSISTANT, content="print('hello world')"),
        ],
        expected_behavior="Valid python code",
        reference_answer="print('hello world')",
        evaluation_type="code_based",
    )

    engine = EvaluationExecutionEngine(
        model_type="base",
        benchmark_dir="benchmarks/benchmark-v1.0",
        dry_run=True,
    )
    prompt_str = engine.format_chat_prompt(case)
    assert "<|im_start|>system\nYou are a helpful assistant.<|im_end|>" in prompt_str
    assert "<|im_start|>user\nWrite hello world in Python.<|im_end|>" in prompt_str
    assert prompt_str.endswith("<|im_start|>assistant\n")


def test_experiment_manager_manifest_and_resume(tmp_path: Path):
    mgr = ExperimentManager(base_dir=tmp_path)
    exp_id = mgr.create_experiment_id("benchmark-v1.0", "base")
    assert "eval-benchmark-v1.0-base-" in exp_id

    gen_cfg = GenerationConfig()
    manifest = mgr.init_manifest(
        experiment_id=exp_id,
        model_type="base",
        benchmark_version="benchmark-v1.0",
        benchmark_sha256="dummy_sha",
        gen_config=gen_cfg,
        case_count=10,
    )
    assert manifest.status == ExperimentStatus.PLANNED
    assert manifest.case_count == 10

    # Save manifest atomically
    man_path = tmp_path / exp_id / "evaluation_manifest.json"
    manifest.save_atomic(man_path)
    assert man_path.exists()

    loaded_man = EvaluationManifest.load(man_path)
    assert loaded_man.experiment_id == exp_id

    # Test resume tracking
    res_path = tmp_path / exp_id / "evaluation_results.jsonl"
    r1 = CaseExecutionResult(
        experiment_id=exp_id,
        benchmark_id="b-001",
        model="base",
        domain="ai_ml",
        topic="lora",
        difficulty="intermediate",
        task_type="coding",
        evaluation_type="code_based",
        status="COMPLETED",
    )
    r2 = CaseExecutionResult(
        experiment_id=exp_id,
        benchmark_id="b-002",
        model="base",
        domain="ai_ml",
        topic="lora",
        difficulty="intermediate",
        task_type="coding",
        evaluation_type="code_based",
        status="FAILED",
    )
    mgr.append_case_result(res_path, r1)
    mgr.append_case_result(res_path, r2)

    completed = mgr.load_completed_case_ids(res_path)
    assert "b-001" in completed
    assert "b-002" not in completed


def test_model_comparison_engine_and_regression(tmp_path: Path):
    gen_cfg = GenerationConfig()
    cfg_hash = gen_cfg.compute_hash()

    base_manifest = EvaluationManifest(
        experiment_id="exp-base",
        model="base",
        benchmark_version="benchmark-v1.0",
        benchmark_sha256="bench_sha",
        generation_config_sha256=cfg_hash,
        case_count=2,
    )
    adapt_manifest = EvaluationManifest(
        experiment_id="exp-adapter",
        model="adapter",
        benchmark_version="benchmark-v1.0",
        benchmark_sha256="bench_sha",
        generation_config_sha256=cfg_hash,
        case_count=2,
    )

    base_res = [
        CaseExecutionResult(
            experiment_id="exp-base",
            benchmark_id="case-1",
            model="base",
            domain="programming",
            topic="python",
            difficulty="intermediate",
            task_type="coding",
            evaluation_type="code_based",
            metrics={"keyword_overlap": 0.4, "formatting_score": 0.8},
            status="COMPLETED",
        ),
        CaseExecutionResult(
            experiment_id="exp-base",
            benchmark_id="case-2",
            model="base",
            domain="science",
            topic="physics",
            difficulty="advanced",
            task_type="explanation",
            evaluation_type="reasoning",
            metrics={"keyword_overlap": 0.8, "formatting_score": 1.0},
            status="COMPLETED",
        ),
    ]

    # Adapter improves case-1, regresses case-2
    adapt_res = [
        CaseExecutionResult(
            experiment_id="exp-adapter",
            benchmark_id="case-1",
            model="adapter",
            domain="programming",
            topic="python",
            difficulty="intermediate",
            task_type="coding",
            evaluation_type="code_based",
            metrics={"keyword_overlap": 0.9, "formatting_score": 1.0},
            status="COMPLETED",
        ),
        CaseExecutionResult(
            experiment_id="exp-adapter",
            benchmark_id="case-2",
            model="adapter",
            domain="science",
            topic="physics",
            difficulty="advanced",
            task_type="explanation",
            evaluation_type="reasoning",
            metrics={"keyword_overlap": 0.2, "formatting_score": 0.5},
            status="COMPLETED",
        ),
    ]

    report, cmps = ModelComparisonEngine.compare_experiments(
        base_manifest, adapt_manifest, base_res, adapt_res
    )

    assert report.cases_total == 2
    assert report.cases_improved == 1
    assert report.cases_regressed == 1
    assert report.domain_deltas["programming"]["improved"] == 1
    assert report.domain_deltas["science"]["regressed"] == 1

    # Report saving
    ModelComparisonEngine.save_comparison_reports(report, tmp_path)
    assert (tmp_path / "model_comparison.json").exists()
    assert (tmp_path / "model_comparison.md").exists()


def test_model_comparison_rejects_mismatched_config():
    base_man = EvaluationManifest(
        experiment_id="exp-base",
        model="base",
        benchmark_version="benchmark-v1.0",
        benchmark_sha256="bench_sha",
        generation_config_sha256="hash_1",
    )
    adapt_man = EvaluationManifest(
        experiment_id="exp-adapter",
        model="adapter",
        benchmark_version="benchmark-v1.0",
        benchmark_sha256="bench_sha",
        generation_config_sha256="hash_2",
    )

    with pytest.raises(ValueError, match="Generation config hash mismatch"):
        ModelComparisonEngine.compare_experiments(base_man, adapt_man, [], [])


def test_engine_preflight_and_dry_run():
    engine = EvaluationExecutionEngine(
        model_type="base",
        benchmark_dir="benchmarks/benchmark-v1.0",
        dry_run=True,
    )
    ok, issues = engine.preflight()
    assert ok is True
    assert len(issues) == 0

    dry_res = engine.execute_dry_run([], manifest_sha="test_sha256")
    assert dry_res["dry_run"] is True
    assert dry_res["status"] == "VALIDATED"


def test_cli_dry_run_execution():
    from scripts.run_evaluation import parse_args, run_evaluation_mode
    import argparse

    args_base = argparse.Namespace(
        model="base",
        compare=False,
        benchmark="benchmark-v1.0",
        generation_config="configs/generation.yaml",
        output_dir="experiments",
        reports_dir="reports",
        dry_run=True,
        resume=False,
        base_experiment=None,
        adapter_experiment=None,
    )
    ret_base = run_evaluation_mode(args_base)
    assert ret_base == 0

    args_adapter = argparse.Namespace(
        model="adapter",
        compare=False,
        benchmark="benchmark-v1.0",
        generation_config="configs/generation.yaml",
        output_dir="experiments",
        reports_dir="reports",
        dry_run=True,
        resume=False,
        base_experiment=None,
        adapter_experiment=None,
    )
    ret_adapter = run_evaluation_mode(args_adapter)
    assert ret_adapter == 0
