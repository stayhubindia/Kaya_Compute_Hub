#!/usr/bin/env python3
"""
CLI Utility for Pilot Dataset Assembly & Validation (Phase 2.3.5).
Executes candidate pool generation, multi-stage processing, stratified mixing,
splitting, cross-split leakage verification, and readiness reporting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.pilot import PilotAssembler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble and validate ~1,000-example Pilot Dataset (Phase 2.3.5)."
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
        help="Path to domain dataset templates YAML manifest.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Target number of conversational examples in final pilot (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic pseudo-random seed (default: 42).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="pilot-v1",
        help="Pilot release version tag (default: 'pilot-v1').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/pilot/v1",
        help="Target directory for processed splits, manifests, and reports.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=1.2,
        help="Multiplier for initial candidate pool sizing (default: 1.2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 64)
    print("Qwen Pilot Dataset Assembly & Validation Engine (Phase 2.3.5)")
    print("=" * 64)
    print(f"Target Count:         {args.count}")
    print(f"Candidate Multiplier: {args.candidate_multiplier} (~{int(args.count * args.candidate_multiplier)} candidates)")
    print(f"Deterministic Seed:   {args.seed}")
    print(f"Pilot Version:        {args.version}")
    print(f"Config File:          {args.config}")
    print(f"Output Directory:     {args.output_dir}")
    print("=" * 64)
    print("\nAssembling candidate pool across 13 domains and executing pipeline...\n")

    try:
        assembler = PilotAssembler(
            config_path=args.config,
            templates_path=args.templates,
        )

        result = assembler.assemble(
            target_count=args.count,
            seed=args.seed,
            version=args.version,
            output_dir=args.output_dir,
            candidate_multiplier=args.candidate_multiplier,
            save_outputs=True,
        )

        manifest = result.manifest
        report = result.readiness_report

        print("=" * 64)
        print("Pilot Execution Summary")
        print("=" * 64)
        print(f"Pilot version:      {manifest.pilot_version}")
        print(f"Target:             {manifest.target_count}")
        print(f"Candidates:         {manifest.candidate_count}")
        print(f"Accepted:           {manifest.accepted_count}")
        print(f"Rejected:           {manifest.rejected_count}")
        print(f"Duplicates:         {manifest.exact_duplicate_count + manifest.near_duplicate_count} (exact: {manifest.exact_duplicate_count}, near: {manifest.near_duplicate_count})")
        print(f"Final:              {manifest.actual_count} (train: {manifest.train_count}, val: {manifest.validation_count}, test: {manifest.test_count})")
        print()
        print("Domain balance:")
        for dom, stats in sorted(report.domain_distribution.items()):
            print(f"  - {dom:<22}: {stats.get('count', 0):>4} ({stats.get('percentage', 0.0):>5.1f}% | target: {stats.get('target_percentage', 0.0):>5.1f}%)")

        print()
        print("Difficulty balance:")
        for diff, stats in sorted(report.difficulty_distribution.items()):
            print(f"  - {diff:<22}: {stats.get('count', 0):>4} ({stats.get('percentage', 0.0):>5.1f}% | target: {stats.get('target_percentage', 0.0):>5.1f}%)")

        print()
        qual = report.quality_summary
        print(f"Quality:            mean: {qual.get('mean_score')}, median: {qual.get('median_score')}, min: {qual.get('min_score')}, max: {qual.get('max_score')}")
        print(f"                    >=0.85: {qual.get('pct_ge_085', 0.0):.1f}%, >=0.90: {qual.get('pct_ge_090', 0.0):.1f}%")
        prov = report.provenance_summary
        print(f"Provenance:         {prov.get('with_provenance')}/{manifest.actual_count} ({prov.get('provenance_rate', 0.0):.1f}%) complete")
        print(f"Leakage:            {'DETECTED ❌' if report.leakage_detected else 'NONE DETECTED ✅'}")
        print(f"Readiness:          {report.overall_status}")
        print()
        print(f"Output:             {args.output_dir}")
        print("=" * 64)

    except Exception as e:
        print(f"ERROR during pilot execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
