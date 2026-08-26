#!/usr/bin/env python3
"""
Scientific Instruction Dataset Generation CLI (Phase 3.4).

Transforms ingested scientific knowledge chunks into a high-fidelity
instruction-tuning dataset (dataset-v2.0 candidate release).

Usage:
  python scripts/generate_instruction_dataset.py --dry-run
  python scripts/generate_instruction_dataset.py --max-chunks 25 --seed 42
  python scripts/generate_instruction_dataset.py --resume --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.models import CandidateGenerationPolicy
from src.generation.pipeline import ScientificGenerationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_instruction_dataset")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3.4 — Source-Grounded Scientific Instruction Dataset Generation Engine"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/ingested/nptel_corpus/chunks.jsonl",
        help="Path to input chunks.jsonl file",
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="data/ingested/nptel_corpus/documents.jsonl",
        help="Path to documents.jsonl file for metadata lookups",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/instruction_dataset/v2.0",
        help="Directory to output generated dataset and reports",
    )
    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=None,
        help="Target total number of accepted examples to produce",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of input chunks to process (for pilot runs)",
    )
    parser.add_argument(
        "--candidates-per-chunk",
        type=int,
        default=None,
        help="Override max candidates generated per chunk",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry-run analysis without generating records",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing checkpoint (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Overwrite checkpoint and reprocess from scratch",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed chunks recorded in checkpoint",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed execution report to stdout",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Perform strict post-generation schema and grounding validation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Phase 3.4 — Scientific Instruction Dataset Generation Engine")
    logger.info(f"Input Chunks:     {args.input}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Seed:             {args.seed}")
    logger.info(f"Max Chunks:       {args.max_chunks}")
    logger.info(f"Dry Run:          {args.dry_run}")
    logger.info(f"Resume:           {args.resume}")
    logger.info("=" * 60)

    # Initialize policy
    policy = CandidateGenerationPolicy(deterministic_seed=args.seed)
    if args.candidates_per_chunk:
        policy.max_candidates_per_chunk = args.candidates_per_chunk

    pipeline = ScientificGenerationPipeline(
        input_chunks_path=args.input,
        documents_path=args.documents,
        output_dir=args.output_dir,
        policy=policy,
        seed=args.seed,
    )

    if args.dry_run:
        logger.info("Executing Dry-Run Analysis...")
        summary = pipeline.execute_dry_run(max_chunks=args.max_chunks)
        logger.info("DRY RUN SUMMARY")
        logger.info(f"Chunks Discovered:        {summary.chunks_discovered:,}")
        logger.info(f"Chunks with Equations:    {summary.chunks_with_equations:,}")
        logger.info(f"Chunks with Tables:       {summary.chunks_with_tables:,}")
        logger.info(f"Estimated Candidates:     {summary.quality_summary.get('estimated_candidates', 0):,}")
        logger.info("Top Task Types Eligible:")
        for t, cnt in sorted(summary.task_distribution.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  - {t}: {cnt}")
        return 0

    logger.info("Executing Scientific Instruction Generation & Validation...")
    summary = pipeline.run(
        max_chunks=args.max_chunks,
        target_count=args.count,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )

    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE - Scientific Instruction Generation Completed!")
    logger.info(f"Execution ID:          {summary.execution_id}")
    logger.info(f"Lifecycle State:       {summary.lifecycle}")
    logger.info(f"Chunks Processed:      {summary.chunks_processed:,}")
    logger.info(f"Candidates Generated:  {summary.candidates_generated:,}")
    logger.info(f"Candidates Accepted:   {summary.candidates_accepted:,}")
    logger.info(f"Candidates Rejected:   {summary.candidates_rejected:,}")
    logger.info(f"Exact Duplicates:      {summary.exact_duplicates:,}")
    logger.info(f"Near Duplicates:       {summary.near_duplicates:,}")
    logger.info(f"Unique Final Records:  {summary.unique_candidates:,}")
    logger.info(f"  - Train Split (90%): {summary.train_count:,}")
    logger.info(f"  - Val Split (5%):    {summary.validation_count:,}")
    logger.info(f"  - Test Split (5%):   {summary.test_count:,}")
    logger.info(f"Mean Quality Score:    {summary.quality_summary.get('mean_quality_score')}")
    logger.info(f"Manifest Path:         {summary.manifest_path}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
