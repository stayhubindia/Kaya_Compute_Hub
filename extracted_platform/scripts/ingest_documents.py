#!/usr/bin/env python3
"""
Command-Line Interface for Knowledge Document Ingestion (Phase 3.3).
Supports recursive ingestion of NPTEL, arXiv, and scientific PDF/HTML/JSON document corpora.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pipeline import KnowledgeIngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_documents")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3.3 — NPTEL & arXiv Knowledge Ingestion Engine"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to source directory or file containing documents to ingest.",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="unknown",
        help="Source identifier (e.g. nptel, arxiv, mit_ocw).",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="auto",
        choices=["auto", "pdf", "html", "json", "mixed"],
        help="Specific input format to filter or auto-detect.",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        "-o",
        type=str,
        required=True,
        help="Destination directory for structured knowledge datasets.",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume ingestion from last checkpoint (default: True).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume; start fresh run.",
    )
    parser.add_argument(
        "--max-documents",
        "-m",
        type=int,
        default=None,
        help="Maximum number of documents to process in this run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force reprocessing of previously completed documents.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="Generate detailed JSON and Markdown ingestion reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Discover files and print execution plan without extracting content.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    logger.info("=" * 60)
    logger.info("Phase 3.3 — Knowledge Ingestion Engine")
    logger.info(f"Input:      {input_path}")
    logger.info(f"Source:     {args.source}")
    logger.info(f"Output Dir: {output_dir}")
    logger.info(f"Resume:     {args.resume}")
    logger.info(f"Force:      {args.force}")
    logger.info(f"Max Docs:   {args.max_documents}")
    logger.info("=" * 60)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    pipeline = KnowledgeIngestionPipeline(
        output_dir=output_dir,
        source=args.source,
        resume=args.resume,
        force=args.force,
        seed=args.seed,
        max_documents=args.max_documents,
    )

    if args.dry_run:
        discovered = pipeline.discover_files(input_path)
        logger.info(f"[DRY-RUN] Discovered {len(discovered)} candidate files:")
        for idx, f in enumerate(discovered[:20]):
            logger.info(f"  [{idx+1}] {f}")
        if len(discovered) > 20:
            logger.info(f"  ... and {len(discovered) - 20} more files.")
        logger.info("[DRY-RUN] Completed without modifying output directory.")
        return 0

    stats = pipeline.run(input_path)

    logger.info("=" * 60)
    logger.info("INGESTION RUN COMPLETED")
    logger.info(f"Discovered: {stats.documents_discovered}")
    logger.info(f"Processed:  {stats.documents_processed}")
    logger.info(f"Successful: {stats.documents_successful}")
    logger.info(f"Partial:    {stats.documents_partial}")
    logger.info(f"Failed:     {stats.documents_failed}")
    logger.info(f"Duplicates: {stats.documents_duplicate}")
    logger.info(f"Total Chunks: {stats.total_chunks}")
    logger.info(f"Duration:   {stats.processing_duration_seconds:.2f}s")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
