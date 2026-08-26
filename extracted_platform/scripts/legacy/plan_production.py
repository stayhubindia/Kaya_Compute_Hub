#!/usr/bin/env python3
"""
CLI Utility for Production Dataset Planning & Dry-Run Specification (Phase 3.1).
Calculates exact mathematical domain/difficulty quotas, 2D joint matrix,
batch allocations, deterministic seeds, and manifests without generating dataset records.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.production import ProductionPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan Production Dataset Generation & Scaling Architecture (Phase 3.1)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML (default: configs/dataset.yaml)",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="configs/domain_templates.yaml",
        help="Path to domain templates YAML (default: configs/domain_templates.yaml)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="configs/sources.yaml",
        help="Path to source manifest YAML (default: configs/sources.yaml)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target number of final accepted dataset examples (e.g. 10000, 25000, 50000, 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global deterministic random seed (default: 42)",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=None,
        help="Candidate multiplier to absorb filtering/dedup (default: 1.20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for parallel/resumable generation (default: 500)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Dataset version identifier (e.g. dataset-v1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for production dataset outputs",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save plan reports or manifests to disk",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 80)
    print(" QWEN3-4B PRODUCTION DATASET PLANNING & SCALING ENGINE")
    print("=" * 80)

    try:
        planner = ProductionPlanner(
            config_path=args.config,
            template_manifest_path=args.templates,
            source_manifest_path=args.sources,
        )

        plan = planner.plan(
            target_count=args.target,
            seed=args.seed,
            candidate_multiplier=args.candidate_multiplier,
            batch_size=args.batch_size,
            version=args.version,
            output_dir=args.output_dir,
        )

        print("\nProduction Dataset Plan")
        print("-----------------------")
        print(f"Version:              {plan.version}")
        print(f"Target Count:         {plan.target_count:,} examples")
        print(f"Candidate Target:     {plan.candidate_target:,} examples ({plan.candidate_multiplier:.2f}x multiplier)")
        print(f"Global Seed:          {plan.seed}")
        print(f"Batch Size:           {plan.batch_size:,} candidates/batch")
        print(f"Estimated Batches:    {plan.estimated_batches} batches")
        print(f"Task Strategy:        {plan.task_strategy}")
        print(f"Source Strategy:      {plan.source_strategy}")

        print("\nDomain Quotas (Hare-Niemeyer / Largest Remainder):")
        print("--------------------------------------------------")
        print(f"{'Domain':<24} {'Weight':<8} {'Target %':<10} {'Exact':<10} {'Integer Quota':<14} {'Adj':<6}")
        print("-" * 74)
        for q in plan.domain_quotas:
            adj = f"+{q.rounding_adjustment}" if q.rounding_adjustment > 0 else "0"
            print(
                f"{q.category:<24} {q.weight:<8.3f} {q.target_percentage:<9.2f}% {q.exact_quota:<10.2f} {q.integer_quota:<14,d} {adj:<6}"
            )
        total_dom = sum(q.integer_quota for q in plan.domain_quotas)
        print("-" * 74)
        print(f"{'TOTAL':<24} {'1.000':<8} {'100.00%':<10} {plan.target_count:<10.2f} {total_dom:<14,d}")

        print("\nDifficulty Quotas:")
        print("------------------")
        print(f"{'Difficulty':<16} {'Weight':<8} {'Target %':<10} {'Exact':<10} {'Integer Quota':<14} {'Adj':<6}")
        print("-" * 66)
        for q in plan.difficulty_quotas:
            adj = f"+{q.rounding_adjustment}" if q.rounding_adjustment > 0 else "0"
            print(
                f"{q.category:<16} {q.weight:<8.2f} {q.target_percentage:<9.1f}% {q.exact_quota:<10.2f} {q.integer_quota:<14,d} {adj:<6}"
            )
        total_diff = sum(q.integer_quota for q in plan.difficulty_quotas)
        print("-" * 66)
        print(f"{'TOTAL':<16} {'1.000':<8} {'100.00%':<10} {plan.target_count:<10.2f} {total_diff:<14,d}")

        print("\nDomain x Difficulty Quota Matrix:")
        print("---------------------------------")
        header = f"{'Domain':<22} | {'Beginner':<9} | {'Intermed':<9} | {'Advanced':<9} | {'Expert':<9} | {'Row Total':<10}"
        print(header)
        print("-" * len(header))
        for dom, row_total in sorted(plan.matrix.row_totals.items()):
            cells = plan.matrix.matrix.get(dom, {})
            b = cells.get("beginner", 0)
            i = cells.get("intermediate", 0)
            a = cells.get("advanced", 0)
            e = cells.get("expert", 0)
            print(f"{dom:<22} | {b:<9,d} | {i:<9,d} | {a:<9,d} | {e:<9,d} | {row_total:<10,d}")
        print("-" * len(header))
        cb = plan.matrix.col_totals.get("beginner", 0)
        ci = plan.matrix.col_totals.get("intermediate", 0)
        ca = plan.matrix.col_totals.get("advanced", 0)
        ce = plan.matrix.col_totals.get("expert", 0)
        print(f"{'TOTAL':<22} | {cb:<9,d} | {ci:<9,d} | {ca:<9,d} | {ce:<9,d} | {plan.matrix.grand_total:<10,d}")

        print("\nBatch Breakdown:")
        print("----------------")
        for bp in plan.batch_plans:
            print(
                f"  [{bp.batch_index:03d}/{plan.estimated_batches:03d}] {bp.batch_id} "
                f"(seed={bp.seed}, target={bp.target_count:,}, candidate_target={bp.candidate_target:,})"
            )

        if not args.no_save:
            reports_dir = Path(plan.storage_layout["reports"])
            manifests_dir = Path(plan.storage_layout["manifests"])

            json_rep, md_rep = plan.save_reports(reports_dir)
            print(f"\n[+] Production Plan JSON:     {json_rep}")
            print(f"[+] Production Plan Markdown: {md_rep}")

            manifest = planner.create_initial_manifest(plan)
            manifest_path = manifests_dir / "production_manifest.json"
            manifest.save(manifest_path)
            print(f"[+] Production Manifest:      {manifest_path}")

        print("\n" + "=" * 80)
        print(f" SUCCESS: Production plan established for {plan.target_count:,} examples (0 records synthesized).")
        print("=" * 80)
        return 0

    except Exception as e:
        print(f"\n[ERROR] Production planning failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
