#!/usr/bin/env python3
"""
CLI Driver for Dataset Engineering Pipeline.
Executes ingestion, normalization, cleaning, deduplication, quality scoring, splitting, and reporting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.pipeline import DatasetPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Qwen Conversational Dataset Engineering Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input path (single JSON/JSONL file or directory). Overrides config paths.raw if specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for train/val/test splits and reports. Overrides config paths.processed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible train/val/test splitting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute pipeline and compute metrics without saving split files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"[Error] Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Qwen Dataset Pipeline Initializing ===")
    print(f"Configuration: {config_path}")

    pipeline = DatasetPipeline(config_path=config_path)

    # Determine input path
    input_path = args.input
    if not input_path:
        input_path = pipeline.paths_cfg.get("raw")
        if not input_path or not Path(input_path).exists():
            # Fallback to local default if drive path not mounted
            local_fallback = PROJECT_ROOT / "data" / "fixtures" / "raw"
            if local_fallback.exists():
                input_path = str(local_fallback)
            else:
                print(
                    f"[Error] No input path provided and default raw path does not exist: {input_path}",
                    file=sys.stderr,
                )
                print("Specify input with: --input <path_to_json_or_dir>", file=sys.stderr)
                sys.exit(1)

    # Override seed if provided
    if args.seed is not None:
        pipeline.splitter.random_seed = args.seed

    output_dir = args.output_dir
    save_outputs = not args.dry_run

    print(f"Input Path:  {input_path}")
    print(f"Output Dir:  {output_dir or pipeline.paths_cfg.get('processed', 'datasets/processed')}")
    print(f"Dry Run:     {args.dry_run}")
    print("\nExecuting Pipeline Stages...")

    result = pipeline.run(
        input_path=input_path,
        output_dir=output_dir,
        save_outputs=save_outputs,
    )

    print("\n=== Pipeline Execution Summary ===")
    print(f"Total Raw Inputs Ingested:    {result.total_raw}")
    print(f"Accepted & Enriched Records:  {result.accepted_count}")
    print(f"Rejected by Cleaning:         {result.rejected_count}")
    print(f"Exact Duplicates Removed:     {result.exact_duplicates}")
    print(f"Near Duplicates Removed:      {result.near_duplicates}")
    print(f"\n--- Split Distribution ---")
    print(f"Train Set:                    {len(result.split_result.train)} records")
    print(f"Validation Set:               {len(result.split_result.validation)} records")
    print(f"Test Set (Isolated):          {len(result.split_result.test)} records")

    if save_outputs:
        print(f"\n--- Output Files Generated ---")
        for key, path in result.output_files.items():
            print(f"  {key:15s}: {path}")

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
