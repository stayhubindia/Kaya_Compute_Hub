#!/usr/bin/env python3
"""
CLI Utility for Synthetic Dataset Generation Engine (Phase 2.3.3).
Generates deterministic synthetic conversational training datasets from declarative
TaskTemplate specifications, attaches full provenance, and optionally passes
generated batches through the end-to-end processing pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.generator import (
    GenerationRequest,
    GenerationResult,
    SampleSyntheticGenerator,
)
from src.dataset.pipeline import DatasetPipeline
from src.dataset.template_registry import TemplateRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen Synthetic Dataset Generation Engine (Phase 2.3.3)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="programming_python_debugging_intermediate",
        help="Template ID to generate examples for (from domain_templates.yaml).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of synthetic examples to generate (1-100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Explicit generation batch identifier (default: auto-generated).",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["beginner", "intermediate", "advanced", "expert"],
        help="Optional difficulty level override.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output JSONL file path (default: datasets/raw/synthetic/<batch_id>.jsonl).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="sample",
        choices=["sample"],
        help="Generator backend to use (default: sample).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to global dataset configuration YAML.",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="configs/domain_templates.yaml",
        help="Path to domain templates manifest YAML.",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Immediately process generated records through Phase 2.2 pipeline.",
    )
    parser.add_argument(
        "--pipeline-output-dir",
        type=str,
        default="datasets/processed/pilot",
        help="Output directory for processed splits and quality reports when --run-pipeline is set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.count < 1 or args.count > 50000:
        print(f"Error: --count must be between 1 and 50000 for safety (got {args.count}).", file=sys.stderr)
        return 1

    # 1. Load Template Registry
    templates_path = Path(args.templates)
    if not templates_path.is_file():
        print(f"Error: Templates file not found: {templates_path}", file=sys.stderr)
        return 1

    try:
        registry = TemplateRegistry.from_yaml(templates_path)
    except Exception as e:
        print(f"Error loading template registry: {e}", file=sys.stderr)
        return 1

    template = registry.lookup_template(args.template)
    if template is None:
        print(
            f"Error: Template '{args.template}' not found in registry. Available templates:\n"
            + ", ".join(t.id for t in registry.list_templates()[:10]) + " ...",
            file=sys.stderr,
        )
        return 1

    # 2. Instantiate Generator Backend
    if args.generator == "sample":
        generator = SampleSyntheticGenerator()
    else:
        print(f"Error: Unknown generator backend '{args.generator}'", file=sys.stderr)
        return 1

    # 3. Formulate Strongly Typed Request
    request = GenerationRequest(
        template_id=template.id,
        domain=template.domain,
        topic=template.topic,
        task_type=template.task_type,
        difficulty=args.difficulty or template.difficulty,
        number_of_examples=args.count,
        seed=args.seed,
        generation_batch_id=args.batch_id,
    )

    effective_batch_id = request.get_effective_batch_id()

    # Determine Output File Path
    if args.output:
        out_file = Path(args.output)
    else:
        out_file = Path("datasets/raw/synthetic") / f"{effective_batch_id}.jsonl"

    # Overwrite Guard Check
    if out_file.is_file() and not args.overwrite:
        print(
            f"Error: Output file '{out_file}' already exists. Pass --overwrite to replace.",
            file=sys.stderr,
        )
        return 1

    # 4. Execute Batch Generation
    result: GenerationResult = generator.generate_batch(request, template_registry=registry)

    # 5. Save Output
    if result.records:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            result.save_jsonl(out_file, overwrite=args.overwrite)
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            return 1

    # 6. Format CLI Report Output
    status_str = "SUCCESS" if result.is_successful else "FAILED"
    print("=" * 60)
    print("Qwen Synthetic Dataset Generation Engine (Phase 2.3.3)")
    print("=" * 60)
    print(f"Generator:        {result.generator_name} (v{result.generator_version})")
    print(f"Template:         {template.id}")
    print(f"Domain / Topic:   {template.domain} / {template.topic}")
    print(f"Task / Diff:      {template.task_type} / {request.difficulty or template.difficulty}")
    print(f"Batch ID:         {result.batch_id}")
    print(f"Seed:             {args.seed}")
    print(f"Requested:        {result.requested_count}")
    print(f"Generated:        {result.generated_count}")
    print(f"Failed:           {result.failed_count}")
    print(f"Output:           {out_file}")
    print(f"Status:           {status_str}")
    if result.errors:
        print(f"Errors:           {json.dumps(result.errors, indent=2)}")
    print("=" * 60)

    # 7. Optional Pipeline Integration Run
    if args.run_pipeline and result.records:
        print("\nExecuting Phase 2.2 Processing Pipeline on generated batch...")
        pipeline_out_dir = Path(args.pipeline_output_dir)
        pipeline_out_dir.mkdir(parents=True, exist_ok=True)

        pipeline = DatasetPipeline(config_path=Path(args.config))
        pipeline_result = pipeline.run(
            input_path=out_file,
            output_dir=pipeline_out_dir,
            save_outputs=True,
        )

        print("-" * 60)
        print("Pipeline Execution Summary:")
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
        print(f"- Pipeline Reports Dir:  {pipeline_out_dir}")
        print("-" * 60)

    return 0 if result.is_successful else 1


if __name__ == "__main__":
    sys.exit(main())
