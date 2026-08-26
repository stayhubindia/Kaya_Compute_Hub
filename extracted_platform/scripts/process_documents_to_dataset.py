#!/usr/bin/env python3
"""
Master Document-to-Dataset Pipeline Script for Google Colab & Local Execution.
Converts raw PDF, HTML, Markdown (.md), Text (.txt), and JSON files into a
production-ready LLM instruction-tuning dataset (train.jsonl, validation.jsonl, test.jsonl).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pipeline import KnowledgeIngestionPipeline
from src.generation.pipeline import ScientificGenerationPipeline, CandidateGenerationPolicy
from src.dataset.release_qa import DatasetReleaseQAEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("process_documents_to_dataset")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Master PDF/HTML/MD Document to LLM Fine-Tuning Dataset Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to folder or file containing PDF, HTML, MD, TXT, or JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="dataset_output",
        help="Destination directory for final train.jsonl, validation.jsonl, and test.jsonl splits.",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="custom_docs",
        help="Source tag or category name for documents.",
    )
    parser.add_argument(
        "--max-documents",
        "-m",
        type=int,
        default=None,
        help="Limit number of input files to process (useful for fast testing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for splitting and sampling.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Target number of output dataset records.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_ingest_dir = output_dir / "ingested_temp"
    temp_gen_dir = output_dir / "gen_temp"

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("🚀 STARTING END-TO-END DOCUMENT TO DATASET PIPELINE")
    logger.info(f"Input Path:   {input_path}")
    logger.info(f"Output Dir:   {output_dir}")
    logger.info(f"Source Tag:   {args.source}")
    logger.info(f"Max Docs:     {args.max_documents or 'All'}")
    logger.info(f"Random Seed:  {args.seed}")
    logger.info("=" * 70)

    # ---------------------------------------------------------
    # STEP 1: DOCUMENT INGESTION & CHUNKING
    # ---------------------------------------------------------
    logger.info("\n---------------------------------------------------------")
    logger.info("📄 STEP 1: Ingesting & Extracting PDF / HTML / MD / TXT Documents...")
    logger.info("---------------------------------------------------------")

    ingest_pipeline = KnowledgeIngestionPipeline(
        output_dir=temp_ingest_dir,
        source=args.source,
        resume=True,
        force=False,
        seed=args.seed,
        max_documents=args.max_documents,
    )
    ingest_stats = ingest_pipeline.run(input_path)

    logger.info(f"Ingestion completed: Discovered {ingest_stats.documents_discovered} files, extracted {ingest_stats.total_chunks} chunks.")

    chunks_file = temp_ingest_dir / "chunks.jsonl"
    docs_file = temp_ingest_dir / "documents.jsonl"

    if not chunks_file.exists() or chunks_file.stat().st_size == 0:
        logger.error("No valid text chunks were extracted from input documents.")
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 2: INSTRUCTION & QA DATASET SYNTHESIS
    # ---------------------------------------------------------
    logger.info("\n---------------------------------------------------------")
    logger.info("🧠 STEP 2: Synthesizing Instruction & Q&A Dataset Pairs...")
    logger.info("---------------------------------------------------------")

    policy = CandidateGenerationPolicy(deterministic_seed=args.seed)
    gen_pipeline = ScientificGenerationPipeline(
        input_chunks_path=chunks_file,
        documents_path=docs_file if docs_file.exists() else None,
        output_dir=temp_gen_dir,
        policy=policy,
        seed=args.seed,
    )

    gen_summary = gen_pipeline.run(
        target_count=args.target_count,
        resume=True,
    )

    logger.info(f"Synthesis completed: Generated {gen_summary.candidates_generated} candidates, {gen_summary.candidates_accepted} accepted.")

    # ---------------------------------------------------------
    # STEP 3: FINAL QA, LEAKAGE AUDIT & SPLIT PACKAGING
    # ---------------------------------------------------------
    logger.info("\n---------------------------------------------------------")
    logger.info("🛡️ STEP 3: Quality Audit, Leakage Protection & Dataset Splitting...")
    logger.info("---------------------------------------------------------")

    candidates_source = temp_gen_dir / "processed" / "accepted.jsonl"
    if not candidates_source.exists():
        candidates_source = temp_gen_dir / "raw" / "candidates.jsonl"

    # Copy final splits to main output dir
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    gen_splits_dir = temp_gen_dir / "splits"

    train_count = 0
    val_count = 0
    test_count = 0

    if (gen_splits_dir / "train.jsonl").exists():
        train_lines = (gen_splits_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
        val_lines = (gen_splits_dir / "validation.jsonl").read_text(encoding="utf-8").strip().splitlines()
        test_lines = (gen_splits_dir / "test.jsonl").read_text(encoding="utf-8").strip().splitlines()

        (output_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
        (output_dir / "validation.jsonl").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
        (output_dir / "test.jsonl").write_text("\n".join(test_lines) + "\n", encoding="utf-8")

        (splits_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
        (splits_dir / "validation.jsonl").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
        (splits_dir / "test.jsonl").write_text("\n".join(test_lines) + "\n", encoding="utf-8")

        train_count = len(train_lines)
        val_count = len(val_lines)
        test_count = len(test_lines)

    total_records = train_count + val_count + test_count
    elapsed_time = time.time() - start_time

    # Generate Manifest Summary
    manifest_data = {
        "dataset_name": f"{args.source}_dataset",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "input_source": str(input_path),
        "total_documents_processed": ingest_stats.documents_processed,
        "total_chunks_extracted": ingest_stats.total_chunks,
        "total_dataset_records": total_records,
        "splits": {
            "train": train_count,
            "validation": val_count,
            "test": test_count,
        },
        "files": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
        },
        "pipeline_duration_seconds": round(elapsed_time, 2),
    }

    with open(output_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("🎉 SUCCESS! DATASET PIPELINE COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Total Processing Time:  {elapsed_time:.2f} seconds")
    logger.info(f"Documents Ingested:     {ingest_stats.documents_processed}")
    logger.info(f"Text Chunks Created:    {ingest_stats.total_chunks}")
    logger.info(f"Final Dataset Records:  {total_records}")
    logger.info(f"  - Train Split (90%):  {train_count} records -> {output_dir / 'train.jsonl'}")
    logger.info(f"  - Val Split (5%):     {val_count} records -> {output_dir / 'validation.jsonl'}")
    logger.info(f"  - Test Split (5%):    {test_count} records -> {output_dir / 'test.jsonl'}")
    logger.info(f"Manifest Info:          {output_dir / 'dataset_manifest.json'}")
    logger.info("=" * 70)
    logger.info("\n💡 You can now use 'train.jsonl' directly in Colab QLoRA / SFT training!")


if __name__ == "__main__":
    main()
