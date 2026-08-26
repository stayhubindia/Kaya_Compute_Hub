#!/usr/bin/env python3
"""
CLI Utility to Validate Benchmark Suite Integrity & Independence (Phase 4.5).
Usage:
    python scripts/validate_benchmark.py --benchmark benchmark-v1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_dataset import BenchmarkDatasetManager
from src.evaluation.benchmark_validator import BenchmarkValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4.5 — Validate Benchmark Suite Integrity & Independence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="benchmark-v1.0",
        help="Benchmark version name or path to benchmark directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark.yaml",
        help="Path to benchmark configuration YAML file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    bench_arg = args.benchmark
    if Path(bench_arg).is_dir():
        bench_dir = Path(bench_arg)
    else:
        bench_dir = Path("benchmarks") / bench_arg

    if not bench_dir.exists():
        print(f"[✗] Error: Benchmark directory not found at '{bench_dir}'")
        return 1

    config_path = Path(args.config)
    cfg: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    print("=" * 65)
    print(f"Validating Benchmark Suite at '{bench_dir}'")
    print("=" * 65)

    # 1. Load and verify SHA-256
    try:
        cases, manifest, stats = BenchmarkDatasetManager.load_benchmark(bench_dir)
        print(f"[✓] Successfully loaded {len(cases)} cases from {bench_dir / 'benchmark.jsonl'}")
        print(f"[✓] SHA-256 Checksum Verified : {manifest.benchmark_sha256}")
        print(f"[✓] Lifecycle State Verified   : {manifest.lifecycle_status}")
    except Exception as e:
        print(f"[✗] Integrity failure: {e}")
        return 1

    # 2. Run Leakage & Quality Validation
    validator = BenchmarkValidator(
        excluded_split_files=cfg.get("leakage_prevention", {}).get("excluded_dataset_splits")
    )
    report = validator.validate_suite(cases)

    print("\n--- Validation & Independence Audit ---")
    print(f"Accepted Cases          : {report.accepted_count} / {report.total_cases}")
    print(f"Exact Overlaps (Leakage): {report.exact_overlaps}")
    print(f"Near Overlaps (Leakage) : {report.near_overlaps}")
    print(f"Internal Duplicates     : {report.internal_duplicates}")
    print(f"Domains Represented     : {len(report.domain_counts)} / 13")
    print(f"Difficulties Covered    : {len(report.difficulty_counts)} / 4")
    print(f"Evaluation Types Covered: {len(report.evaluation_type_counts)} / 7")

    if not report.is_valid:
        print("\n[✗] VALIDATION FAILED:")
        for err in report.schema_errors + report.leakage_errors + report.quality_errors:
            print(f"  - {err}")
        return 1

    print("\n" + "=" * 65)
    print("BENCHMARK VALIDATION PASSED — ZERO LEAKAGE — FROZEN & READY")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
