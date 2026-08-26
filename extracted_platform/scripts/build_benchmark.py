#!/usr/bin/env python3
"""
CLI Utility to Build Independent Evaluation Benchmark Suite (Phase 4.5).
Usage:
    python scripts/build_benchmark.py --target 500 [--seed 42] [--output-dir benchmarks/benchmark-v1.0] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_cases import BenchmarkSuiteBuilder
from src.evaluation.benchmark_dataset import BenchmarkDatasetManager, BenchmarkManifest
from src.evaluation.benchmark_validator import BenchmarkValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4.5 — Build Independent Benchmark Suite (benchmark-v1.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark.yaml",
        help="Path to benchmark configuration YAML file",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=500,
        help="Target number of benchmark cases to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic benchmark generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/benchmark-v1.0",
        help="Target directory for benchmark artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate generation and validation without writing files to disk",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    cfg: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    target_count = args.target or cfg.get("target_count", 500)
    seed = args.seed or cfg.get("seed", 42)
    output_dir = Path(args.output_dir or cfg.get("storage", {}).get("base_dir", "benchmarks/benchmark-v1.0"))

    print("=" * 65)
    print(f"Building Independent Benchmark Suite: benchmark-v1.0 (Target: {target_count})")
    print("=" * 65)

    # 1. Generate cases
    cases = BenchmarkSuiteBuilder.generate_benchmark_suite(target_count=target_count, seed=seed)
    print(f"[✓] Generated {len(cases)} initial benchmark candidate cases.")

    # 2. Validate cases and audit training data leakage
    validator = BenchmarkValidator(
        excluded_split_files=cfg.get("leakage_prevention", {}).get("excluded_dataset_splits")
    )
    val_report = validator.validate_suite(cases)

    print("\n--- Benchmark Quality & Leakage Audit ---")
    print(f"Total Cases Evaluated   : {val_report.total_cases}")
    print(f"Accepted Cases          : {val_report.accepted_count}")
    print(f"Rejected Cases          : {val_report.rejected_count}")
    print(f"Exact Overlaps (Leakage): {val_report.exact_overlaps}")
    print(f"Near Overlaps (Leakage) : {val_report.near_overlaps}")
    print(f"Internal Duplicates     : {val_report.internal_duplicates}")
    print(f"Domains Represented     : {len(val_report.domain_counts)} / 13")
    print(f"Difficulties Covered    : {len(val_report.difficulty_counts)} / 4")
    print(f"Evaluation Types Covered: {len(val_report.evaluation_type_counts)} / 7")

    if not val_report.is_valid:
        print("\n[✗] BENCHMARK VALIDATION FAILED:")
        for err in val_report.schema_errors + val_report.leakage_errors + val_report.quality_errors:
            print(f"  - {err}")
        return 1

    print("[✓] Zero leakage detected. 100% independence from training splits verified.")

    # 3. Dry-run gate
    if args.dry_run:
        print("\n[DRY RUN] Benchmark validation successful. Skipping disk writes.")
        return 0

    # 4. Save benchmark artifacts
    gen_cfg = cfg.get("generation", {})
    jsonl_path, manifest, stats = BenchmarkDatasetManager.save_benchmark(
        cases=cases,
        base_dir=output_dir,
        config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest(),
        generation_config=gen_cfg,
    )

    print(f"\n[✓] Benchmark bundle successfully written to '{output_dir}':")
    print(f"  - Benchmark Cases: {jsonl_path} ({len(cases)} records)")
    print(f"  - SHA-256 Hash   : {manifest.benchmark_sha256}")
    print(f"  - Manifest       : {output_dir / 'manifest.json'}")
    print(f"  - Statistics     : {output_dir / 'statistics.json'}")
    print(f"  - Documentation  : {output_dir / 'README.md'}")
    print(f"  - Lifecycle State: {manifest.lifecycle_status}")
    print("=" * 65)
    print("BENCHMARK BUILD COMPLETED SUCCESSFULLY")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
