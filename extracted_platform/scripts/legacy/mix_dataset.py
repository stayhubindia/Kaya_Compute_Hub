#!/usr/bin/env python3
"""
CLI Utility for Dataset Mixing & Balancing Engine (Phase 2.3.4).
Combines multiple source pools into a unified dataset according to configurable distributions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.mixer import DatasetMixer, MixingRequest
from src.dataset.pipeline import DatasetPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen Conversational Dataset Mixing & Balancing CLI (Phase 2.3.4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        dest="inputs",
        action="append",
        required=True,
        help="Input JSONL source dataset file or directory. Can be specified multiple times.",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        required=True,
        help="Target total number of examples in the mixed dataset.",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        default="proportional",
        choices=["proportional", "balanced"],
        help="Mixing strategy to apply.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for candidate selection and shuffling.",
    )
    parser.add_argument(
        "--allow-oversampling",
        action="store_true",
        help="Allow controlled duplication when candidate stratum count < target quota.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Custom identifier for the mixing batch.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for the mixed JSONL dataset.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory to save dataset_mix_report.json and dataset_mix_report.md.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if they already exist.",
    )
    parser.add_argument(
        "--dedup-before-mix",
        action="store_true",
        help="Run exact deduplication across input candidates before mixing.",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run Phase 2.2 processing pipeline on the mixed dataset immediately.",
    )
    parser.add_argument(
        "--pipeline-output-dir",
        type=str,
        default="datasets/processed/mixed",
        help="Output directory for Phase 2.2 processing pipeline splits and telemetry.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Determine output path
    output_path = args.output
    if not output_path:
        default_dir = Path("datasets/raw/mixed")
        default_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(default_dir / f"mixed_{args.strategy}_n{args.count}_s{args.seed}.jsonl")

    report_dir = args.report_dir or str(Path(output_path).parent / "reports")

    print("=" * 60)
    print("Qwen Dataset Mixing & Balancing Engine (Phase 2.3.4)")
    print("=" * 60)
    print(f"Inputs:           {args.inputs}")
    print(f"Strategy:         {args.strategy}")
    print(f"Target Count:     {args.count}")
    print(f"Seed:             {args.seed}")
    print(f"Oversampling:     {args.allow_oversampling}")
    print(f"Output Path:      {output_path}")
    print("=" * 60)

    try:
        mixer = DatasetMixer(
            config_path=args.config,
            deduplicate_before_mix=args.dedup_before_mix,
        )

        request = MixingRequest(
            input_sources=args.inputs,
            target_count=args.count,
            strategy=args.strategy,
            seed=args.seed,
            allow_oversampling=args.allow_oversampling,
            batch_id=args.batch_id,
        )

        print("\nIngesting candidate source pools and executing mixing algorithm...")
        result = mixer.mix(request)

        # Save mixed records to JSONL
        saved_count = result.save_jsonl(output_path, overwrite=args.overwrite)
        json_rep, md_rep = result.save_reports(report_dir)

        print("\n" + "=" * 60)
        print("Mixing Execution Summary")
        print("=" * 60)
        print(f"Strategy:         {result.strategy}")
        print(f"Seed:             {result.seed}")
        print(f"Candidates:       {result.total_candidates}")
        print(f"Requested:        {result.requested_count}")
        print(f"Selected:         {result.selected_count}")
        print(f"Discarded:        {result.discarded_count}")
        print(f"Shortages:        {len(result.shortages)}")
        if result.oversampling:
            print(f"Oversampled:      {result.oversampling.oversampled_records} (ratio: {result.oversampling.oversampling_ratio:.3f})")
        else:
            print(f"Oversampled:      0")

        print("\nDomain distribution:")
        for dom, count in sorted(result.domain_distribution.counts.items()):
            if count > 0:
                pct = result.domain_distribution.percentages.get(dom, 0.0)
                t_pct = result.domain_distribution.targets.get(dom, 0.0)
                print(f"  - {dom:<24}: {count:>3} ({pct:>5.1f}% | target: {t_pct:>4.1f}%)")

        print("\nDifficulty distribution:")
        for diff, count in sorted(result.difficulty_distribution.counts.items()):
            if count > 0:
                pct = result.difficulty_distribution.percentages.get(diff, 0.0)
                t_pct = result.difficulty_distribution.targets.get(diff, 0.0)
                print(f"  - {diff:<24}: {count:>3} ({pct:>5.1f}% | target: {t_pct:>4.1f}%)")

        print("\nSource distribution:")
        for src, count in sorted(result.source_distribution.counts.items()):
            if count > 0:
                pct = result.source_distribution.percentages.get(src, 0.0)
                print(f"  - {src:<24}: {count:>3} ({pct:>5.1f}%)")

        print(f"\nOutput JSONL:     {output_path}")
        print(f"Audit Reports:    {json_rep} | {md_rep}")
        print("=" * 60)

        # Optional Phase 2.2 Pipeline Execution
        if args.run_pipeline:
            print("\nExecuting Phase 2.2 Processing Pipeline on mixed dataset...")
            print("-" * 60)
            pipeline = DatasetPipeline(config_path=args.config)
            pipeline_result = pipeline.run(
                input_path=output_path,
                output_dir=args.pipeline_output_dir,
                save_outputs=True,
            )

            print("Phase 2.2 Pipeline Summary:")
            print(f"- Total Raw Ingested:    {pipeline_result.total_raw}")
            print(f"- Clean Accepted:        {pipeline_result.accepted_count}")
            print(f"- Clean Rejected:        {pipeline_result.rejected_count}")
            print(f"- Exact Duplicates:      {pipeline_result.exact_duplicates}")
            print(f"- Near Duplicates:       {pipeline_result.near_duplicates}")
            if pipeline_result.split_result:
                print(
                    f"- Splits:                Train: {len(pipeline_result.split_result.train)}, "
                    f"Val: {len(pipeline_result.split_result.validation)}, "
                    f"Test: {len(pipeline_result.split_result.test)}"
                )
            print(f"- Pipeline Reports Dir:  {args.pipeline_output_dir}")
            print("-" * 60)

        return 0

    except Exception as exc:
        print(f"\n[ERROR] Mixing failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
