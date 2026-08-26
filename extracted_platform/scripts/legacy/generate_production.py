#!/usr/bin/env python3
"""
Production Dataset Generation CLI (Phase 3.2).
Executes batch-based synthetic candidate generation, atomic file persistence,
batch-local cleaning, quality evaluation, deduplication, checkpoint recovery,
and global stratified balancing.

Usage:
  python scripts/generate_production.py --dry-run
  python scripts/generate_production.py --target 100 --batch-size 25 --max-batches 4
  python scripts/generate_production.py --resume
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.production import BatchStatus
from src.dataset.production_generator import ProductionGenerationEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-4B-Base Production Dataset Generation & Scaling Engine (Phase 3.2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to authoritative dataset YAML configuration.",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="configs/domain_templates.yaml",
        help="Path to domain task templates YAML manifest.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="configs/sources.yaml",
        help="Path to source registry YAML manifest.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Dataset version identifier (e.g. 'dataset-v1.0').",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target final dataset example count (e.g. 10000, 100 for test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed for deterministic generation.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=None,
        help="Oversampling multiplier for raw candidate generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of candidate records per generation batch.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root output directory for production artifacts and batches.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically skip batches already recorded as completed in checkpoints.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=False,
        help="Retry batches marked as failed in checkpoints.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit execution to first N batches (useful for staged verification).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute plan and manifest without generating any synthetic records.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Halt execution immediately on first batch failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 80)
    print(" QWEN3-4B PRODUCTION DATASET GENERATION ENGINE (Phase 3.2)")
    print("=" * 80)
    print()

    try:
        engine = ProductionGenerationEngine(
            config_path=args.config,
            templates_path=args.templates,
            sources_path=args.sources,
        )

        result = engine.generate_all(
            target_count=args.target,
            seed=args.seed,
            candidate_multiplier=args.candidate_multiplier,
            batch_size=args.batch_size,
            version=args.version,
            output_dir=args.output_dir,
            resume=args.resume,
            retry_failed=args.retry_failed,
            max_batches=args.max_batches,
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
        )

        if args.dry_run:
            print("[DRY-RUN MODE] Production plan established without synthetic record generation.")
            print(f"Manifest written to: {result.manifest_file}")
            return 0

        print(f"Dataset Version:          {result.dataset_version}")
        print(f"Target Final Count:       {result.target_count:,}")
        print(f"Candidate Target Pool:    {result.candidate_target:,}")
        print(f"Total Raw Generated:      {result.total_generated:,}")
        print(f"Clean Accepted:           {result.total_clean_accepted:,}")
        print(f"Quality Accepted:         {result.total_quality_accepted:,}")
        print(f"Global Deduped:           {result.global_deduped_count:,}")
        print(f"Final Selected Count:     {result.final_selected_count:,}")
        print(f"Shortage Deficit:         {result.shortage_deficit:,}")
        print(f"Replenishment Needed:     {'YES' if result.replenishment_needed else 'NO'}")
        print(f"Overall Candidate Yield:  {result.yield_overall_pct:.2f}%")
        print()

        completed = sum(1 for b in result.batch_results if b.status == BatchStatus.COMPLETED.value)
        failed = sum(1 for b in result.batch_results if b.status == BatchStatus.FAILED.value)
        print(f"Batch Execution:          {completed} completed, {failed} failed (total {len(result.batch_results)})")
        print()

        print(f"[+] Candidate Dataset:    {result.candidate_dataset_file}")
        print(f"[+] Dataset Manifest:     {result.manifest_file}")
        for k, p in sorted(result.report_files.items()):
            print(f"[+] Report ({k}): {p}")

        print()
        print("=" * 80)
        print(" SUCCESS: Production generation run completed.")
        print("=" * 80)
        return 0

    except Exception as e:
        print(f"\n[ERROR] Generation engine execution failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
