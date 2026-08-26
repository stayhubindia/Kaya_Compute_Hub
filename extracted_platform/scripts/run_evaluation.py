#!/usr/bin/env python3
"""
CLI Utility to Execute Benchmark Evaluation & Model Comparison (Phase 4.7).
Supports BASELINE, ADAPTER, and COMPARE modes with strict GPU gating and atomic outputs.

Usage:
    python scripts/run_evaluation.py --model base --benchmark benchmark-v1.0 [--dry-run]
    python scripts/run_evaluation.py --model adapter --benchmark benchmark-v1.0 [--dry-run]
    python scripts/run_evaluation.py --compare --base-experiment experiments/base-001 --adapter-experiment experiments/adapter-001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_dataset import BenchmarkDatasetManager
from src.evaluation.execution import (
    EvaluationExecutionEngine,
    GenerationConfig,
    GPUReadinessGate,
)
from src.evaluation.experiment import (
    EvaluationManifest,
    ExperimentManager,
    ExperimentStatus,
    ModelComparisonEngine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4.7 — Production Evaluation Execution & Model Comparison Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "adapter"],
        help="Model type to evaluate ('base' for Qwen3-4B-Base, 'adapter' for fine-tuned LoRA)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparative regression between baseline and adapter experiments",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="benchmark-v1.0",
        help="Benchmark suite version name or path",
    )
    parser.add_argument(
        "--generation-config",
        type=str,
        default="configs/generation.yaml",
        help="Path to locked generation parameters YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Directory to store experiment results and manifests",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to store comparative reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pipeline, paths, and configurations without executing model inference",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted evaluation run",
    )
    parser.add_argument(
        "--base-experiment",
        type=str,
        help="Path to baseline experiment directory or manifest (for --compare)",
    )
    parser.add_argument(
        "--adapter-experiment",
        type=str,
        help="Path to adapter experiment directory or manifest (for --compare)",
    )
    return parser.parse_args()


def run_evaluation_mode(args: argparse.Namespace) -> int:
    bench_arg = args.benchmark
    bench_dir = Path(bench_arg) if Path(bench_arg).is_dir() else Path("benchmarks") / bench_arg
    if not bench_dir.exists():
        print(f"[✗] Benchmark directory not found at: {bench_dir}")
        return 1

    print("=" * 70)
    print(f"Phase 4.7 Evaluation Orchestrator: {args.model.upper()} Mode")
    print(f"Benchmark: {bench_dir.name} ({bench_dir})")
    print(f"Generation Config: {args.generation_config}")
    print(f"Dry Run: {args.dry_run}")
    print("=" * 70)

    # 1. Load Generation Config and Checksum
    try:
        gen_cfg = GenerationConfig.load_from_yaml(args.generation_config)
        gen_hash = gen_cfg.compute_hash()
        print(f"[✓] Generation Configuration loaded (SHA-256: {gen_hash[:16]}...)")
    except Exception as e:
        print(f"[✗] Failed to load generation config: {e}")
        return 1

    # 2. Load Benchmark Cases and Manifest
    try:
        cases, manifest, stats = BenchmarkDatasetManager.load_benchmark(bench_dir)
        print(f"[✓] Loaded {len(cases)} benchmark cases (SHA-256: {manifest.benchmark_sha256})")
    except Exception as e:
        print(f"[✗] Failed to load benchmark suite: {e}")
        return 1

    # 3. Setup Execution Engine & Preflight
    engine = EvaluationExecutionEngine(
        model_type=args.model,
        benchmark_dir=bench_dir,
        generation_config_path=args.generation_config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    preflight_ok, issues = engine.preflight()
    if not preflight_ok:
        print("\n[✗] Preflight issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        if not args.dry_run:
            return 1

    # 4. Dry-Run Execution
    if args.dry_run:
        dry_res = engine.execute_dry_run(cases, manifest.benchmark_sha256)
        print("\n[✓] DRY-RUN VALIDATION COMPLETE")
        print(f"  - Model: {dry_res['model_type']}")
        print(f"  - Cases: {dry_res['benchmark_cases_count']}")
        print(f"  - Benchmark SHA-256: {dry_res['benchmark_sha256']}")
        print(f"  - Generation Hash: {dry_res['generation_config_hash']}")
        print(f"  - Message: {dry_res['message']}")
        return 0

    # 5. Hardware Gating
    gpu_ok, gpu_msg = GPUReadinessGate.check()
    if not gpu_ok:
        print(f"\n[!] {gpu_msg}")
        print("MODEL INFERENCE NOT EXECUTED — GPU UNAVAILABLE")
        return 0

    print(f"\n[✓] {gpu_msg}")
    print("[*] Starting model inference pipeline...")
    return 0


def run_comparison_mode(args: argparse.Namespace) -> int:
    print("=" * 70)
    print("Phase 4.7 Model Comparison Orchestrator (--compare)")
    print("=" * 70)

    if not args.base_experiment or not args.adapter_experiment:
        print("[✗] Error: --compare requires both --base-experiment and --adapter-experiment paths.")
        return 1

    base_dir = Path(args.base_experiment)
    adapt_dir = Path(args.adapter_experiment)

    base_manifest_file = base_dir / "evaluation_manifest.json" if base_dir.is_dir() else base_dir
    adapt_manifest_file = adapt_dir / "evaluation_manifest.json" if adapt_dir.is_dir() else adapt_dir

    if not base_manifest_file.exists() or not adapt_manifest_file.exists():
        print("[✗] Error: One or both evaluation manifests not found.")
        return 1

    try:
        base_manifest = EvaluationManifest.load(base_manifest_file)
        adapt_manifest = EvaluationManifest.load(adapt_manifest_file)

        # Load case results
        base_results_file = base_dir / "evaluation_results.jsonl" if base_dir.is_dir() else base_manifest_file.parent / "evaluation_results.jsonl"
        adapt_results_file = adapt_dir / "evaluation_results.jsonl" if adapt_dir.is_dir() else adapt_manifest_file.parent / "evaluation_results.jsonl"

        from src.evaluation.execution import CaseExecutionResult
        base_results = []
        if base_results_file.exists():
            with open(base_results_file, "r") as f:
                for line in f:
                    if line.strip():
                        base_results.append(CaseExecutionResult(**json.loads(line)))

        adapt_results = []
        if adapt_results_file.exists():
            with open(adapt_results_file, "r") as f:
                for line in f:
                    if line.strip():
                        adapt_results.append(CaseExecutionResult(**json.loads(line)))

        report, comparisons = ModelComparisonEngine.compare_experiments(
            base_manifest, adapt_manifest, base_results, adapt_results
        )

        ModelComparisonEngine.save_comparison_reports(report, args.reports_dir)
        print(f"\n[✓] Comparison reports saved to '{args.reports_dir}/':")
        print(f"  - {Path(args.reports_dir) / 'model_comparison.json'}")
        print(f"  - {Path(args.reports_dir) / 'model_comparison.md'}")
        print(f"\nOVERALL VERDICT: {report.overall_verdict}")
        return 0

    except Exception as e:
        print(f"[✗] Comparison failed: {e}")
        return 1


def main() -> int:
    args = parse_args()
    if args.compare:
        return run_comparison_mode(args)
    elif args.model:
        return run_evaluation_mode(args)
    else:
        print("Please specify either --model [base|adapter] or --compare.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
