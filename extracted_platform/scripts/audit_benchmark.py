#!/usr/bin/env python3
"""
CLI Utility to Audit Benchmark Suite Quality & Reference Answers (Phase 4.6).
Usage:
    python scripts/audit_benchmark.py --benchmark benchmark-v1.0 [--output-dir reports/]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_audit import BenchmarkAuditor
from src.evaluation.benchmark_dataset import BenchmarkDatasetManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4.6 — Benchmark Semantic & Reference-Answer Quality Audit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="benchmark-v1.0",
        help="Benchmark version name or path to benchmark directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for audit reports",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.90,
        help="Threshold score for PASS classification",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.75,
        help="Threshold score for WARN classification",
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

    print("=" * 70)
    print(f"Auditing Benchmark Suite: {bench_dir.name} ({bench_dir})")
    print("=" * 70)

    # 1. Load Benchmark Suite
    try:
        cases, manifest, stats = BenchmarkDatasetManager.load_benchmark(bench_dir)
        print(f"[✓] Loaded {len(cases)} benchmark cases from '{bench_dir / 'benchmark.jsonl'}'")
        print(f"[✓] Verified SHA-256 Checksum: {manifest.benchmark_sha256}")
    except Exception as e:
        print(f"[✗] Failed to load benchmark suite: {e}")
        return 1

    # 2. Run Comprehensive Quality Audit
    auditor = BenchmarkAuditor(
        pass_threshold=args.pass_threshold,
        warn_threshold=args.warn_threshold,
    )
    report, case_results = auditor.audit_suite(cases, manifest.benchmark_sha256)

    # 3. Output Telemetry
    print("\n--- Sub-Auditor Breakdown ---")
    print(f"Structural Audit   : {report.structural_audit['passed']} Passed | {report.structural_audit['warnings']} Warnings | {report.structural_audit['failures']} Failures")
    print(f"Semantic Audit     : {report.semantic_audit['passed']} Passed | {report.semantic_audit['warnings']} Warnings | {report.semantic_audit['failures']} Failures")
    print(f"Reference Audit    : {report.reference_audit['passed']} Passed | {report.reference_audit['warnings']} Warnings | {report.reference_audit['failures']} Failures")
    print(f"Mathematics Audit  : {report.mathematical_audit['checked']} Checked | {report.mathematical_audit['passed']} Passed | {report.mathematical_audit['failed']} Failed")
    print(f"Code Audit         : {report.code_audit['checked']} Checked | {report.code_audit['passed']} Passed | {report.code_audit['warnings']} Warnings | {report.code_audit['failed']} Failed")
    print(f"Reasoning Audit    : {report.reasoning_audit['checked']} Checked | {report.reasoning_audit['passed']} Passed | {report.reasoning_audit['warnings']} Warnings | {report.reasoning_audit['failed']} Failed")
    print(f"Difficulty Check   : {report.difficulty_audit['mismatches']} Mismatches")
    print(f"Task-Type Check    : {report.task_type_audit['mismatches']} Mismatches")
    print(f"Prompt Answer Leaks: {report.answer_leakage_audit['detected']} Detected")
    print(f"Semantic Clusters  : {report.semantic_duplicate_audit['unique']} Unique | {report.semantic_duplicate_audit['similar']} Similar | {report.semantic_duplicate_audit['duplicate']} Duplicate")

    print("\n--- Benchmark Quality Scores ---")
    print(f"Mean Score: {report.quality_scores.mean:.4f}")
    print(f"P50 Score : {report.quality_scores.p50:.4f}")
    print(f"P90 Score : {report.quality_scores.p90:.4f}")
    print(f"P95 Score : {report.quality_scores.p95:.4f}")
    print(f"Min / Max : {report.quality_scores.min:.4f} / {report.quality_scores.max:.4f}")

    # 4. Save Reports
    out_dir = Path(args.output_dir)
    BenchmarkAuditor.save_reports(report, case_results, out_dir)
    print(f"\n[✓] Audit reports saved to '{out_dir}/':")
    print(f"  - {out_dir / 'benchmark_audit.json'}")
    print(f"  - {out_dir / 'benchmark_audit.md'}")
    print(f"  - {out_dir / 'benchmark_case_audit.jsonl'}")

    print("\n" + "=" * 70)
    print(f"RELEASE DECISION: {report.release_decision}")
    print("=" * 70)

    if report.critical_failures:
        print("\n[!] Critical Failures:")
        for cf in report.critical_failures:
            print(f"  - {cf['benchmark_id']} ({cf['domain']}): {', '.join(cf['issues'])}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
