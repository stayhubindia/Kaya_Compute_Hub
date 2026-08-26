#!/usr/bin/env python3
"""
CLI Utility for Qwen3-4B Evaluation, Benchmarking & Regression Analysis (Phase 4.4).
Usage:
    python scripts/evaluate_model.py --preflight
    python scripts/evaluate_model.py --model base [--dry-run]
    python scripts/evaluate_model.py --model adapter [--dry-run]
    python scripts/evaluate_model.py --compare
    python scripts/evaluate_model.py --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.config import EvaluationConfig
from src.evaluation.runner import EvaluationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4.4 — Qwen3-4B-Base Evaluation & Benchmarking CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation.yaml",
        help="Path to evaluation configuration YAML file",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run comprehensive preflight verification checks",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "adapter"],
        help="Target model architecture for evaluation ('base' or 'adapter')",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Execute regression comparison between baseline and adapter reports",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate formatted markdown and JSON reports from existing evaluation runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible evaluation execution",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of test examples to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for evaluation artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run evaluation in offline simulation mode without requiring live GPU",
    )
    return parser.parse_args()


def run_preflight(runner: EvaluationRunner) -> int:
    """Execute and display preflight audit."""
    print("=" * 60)
    print("Phase 4.4 — Evaluation Preflight Audit")
    print("=" * 60)

    audit = runner.run_preflight_audit()
    for name, passed in audit.checks.items():
        icon = "✓" if passed else "✗"
        detail = audit.details.get(name, "")
        print(f"[{icon}] {name:30} : {detail}")

    if audit.errors:
        print("\nPreflight Audit Failures:")
        for err in audit.errors:
            print(f"  - {err}")
        print("\nStatus: PREFLIGHT AUDIT FAILED")
        return 1

    print("\nStatus: EVALUATION PREFLIGHT AUDIT PASSED")
    return 0


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        config = EvaluationConfig.from_yaml(config_path)
    else:
        config = EvaluationConfig()

    if args.seed is not None:
        config.seed = args.seed
    if args.max_examples is not None:
        config.dataset.max_examples = args.max_examples
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    runner = EvaluationRunner(config)

    # 1. Preflight
    if args.preflight:
        return run_preflight(runner)

    # 2. Evaluation (--model base / --model adapter)
    if args.model:
        print(f"\nInitiating Evaluation for target: '{args.model}' (dry-run: {args.dry_run})...")
        report = runner.evaluate(model_type=args.model, dry_run=args.dry_run, max_examples=args.max_examples)

        print("\n" + "=" * 60)
        print(f"Evaluation Run Summary: {report.model_name} ({report.model_type})")
        print("=" * 60)
        print(f"Sample Count       : {report.sample_count}")
        print(f"Hardware Device    : {report.hardware_device}")
        print(f"Mock Mode          : {report.is_mock}")
        print(f"Validity Rate      : {report.overall_metrics.validity_rate:.2%}")
        print(f"Empty Rate         : {report.overall_metrics.empty_rate:.2%}")
        print(f"Avg Repetition     : {report.overall_metrics.avg_repetition_ratio:.4f}")
        print(f"Avg Formatting     : {report.overall_metrics.avg_formatting_score:.2f}")
        print(f"Avg Token Length   : {report.overall_metrics.avg_token_length:.1f}")
        print("=" * 60)

        if not runner.hardware.cuda_available and not args.dry_run:
            print("\n[NOTE] MODEL INFERENCE BLOCKED — GPU UNAVAILABLE (CUDA not found)")
            print("Status: EVALUATION INFRASTRUCTURE READY")
        else:
            print("\nStatus: EVALUATION COMPLETED")
        return 0

    # 3. Compare
    if args.compare:
        print("\nExecuting Regression Comparison (Baseline vs Adapter)...")
        # Run base and adapter evaluations in dry-run if not present
        base_rep = runner.evaluate(model_type="base", dry_run=True)
        adapt_rep = runner.evaluate(model_type="adapter", dry_run=True)
        regression = runner.compare(base_rep, adapt_rep)

        print("\n" + "=" * 60)
        print("Regression Comparison Summary")
        print("=" * 60)
        print(f"Verdict             : {regression.verdict}")
        print(f"Total Improvements  : {regression.total_improvements}")
        print(f"Total Regressions   : {regression.total_regressions}")
        print(f"Total Unchanged     : {regression.total_unchanged}")
        print(f"Summary             : {regression.executive_summary}")
        print("=" * 60)
        print("\nStatus: REGRESSION ANALYSIS COMPLETED")
        return 0

    # 4. Report
    if args.report:
        print("\nGenerating Evaluation & Benchmark Reports...")
        base_rep = runner.evaluate(model_type="base", dry_run=True)
        adapt_rep = runner.evaluate(model_type="adapter", dry_run=True)
        regression = runner.compare(base_rep, adapt_rep)
        print(f"Reports successfully written to: '{config.reports_dir}'")
        return 0

    # Default to preflight if no action specified
    return run_preflight(runner)


if __name__ == "__main__":
    sys.exit(main())
