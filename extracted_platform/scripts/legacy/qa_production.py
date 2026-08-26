#!/usr/bin/env python3
"""
Production Dataset Quality Assurance, Token Budget Analysis & Final Freeze CLI (Phase 3.3).

Usage:
  # 1. QA on candidate dataset
  python scripts/qa_production.py \
    --input datasets/production/processed/candidate_dataset.jsonl \
    --config configs/dataset.yaml \
    --output-dir datasets/production/reports

  # 2. QA with train/val/test split leakage analysis
  python scripts/qa_production.py \
    --train datasets/production/processed/train.jsonl \
    --validation datasets/production/processed/validation.jsonl \
    --test datasets/production/processed/test.jsonl \
    --output-dir datasets/production/reports

  # 3. Explicit Final Freeze Locking
  python scripts/qa_production.py \
    --input datasets/production/processed/candidate_dataset.jsonl \
    --manifest datasets/production/manifests/production_manifest.json \
    --freeze
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.dataset.loader import DatasetLoader
from src.dataset.production_qa import ProductionQAEngine, ReadinessStatus
from src.dataset.splitter import DatasetSplitter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3.3: Production Dataset QA, Token Budget Analysis & Freeze CLI."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to candidate_dataset.jsonl",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--validation",
        type=str,
        default=None,
        help="Path to validation.jsonl",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Path to test.jsonl",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML file (default: configs/dataset.yaml)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="datasets/production/manifests/production_manifest.json",
        help="Path to production_manifest.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional path or HF ID to override tokenizer path in config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/production/reports",
        help="Directory to save all QA and freeze reports",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=4096,
        help="Maximum sequence length for truncation risk analysis (default: 4096)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="List of epochs for training budget estimation (default: 1 2 3)",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Micro batch size for training step estimation",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help="Gradient accumulation steps for training step estimation",
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        help="Automatically generate 90/5/5 train/val/test splits if only --input is provided",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Execute final dataset cryptographic freeze locking",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save accepted/filtered dataset records",
    )
    parser.add_argument(
        "--min-score",
        "--min-quality-score",
        type=float,
        default=None,
        help="Minimum quality score threshold for record filtering",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force dataset freeze even if warning readiness gates are present",
    )

    args = parser.parse_args()

    # Determine input sources
    input_path = Path(args.input) if args.input else None
    train_path = Path(args.train) if args.train else None
    val_path = Path(args.validation) if args.validation else None
    test_path = Path(args.test) if args.test else None
    manifest_path = Path(args.manifest) if args.manifest else None
    out_dir = Path(args.output_dir)

    if not input_path and not (train_path and val_path and test_path):
        # Default fallback to candidate_dataset.jsonl
        default_input = Path("datasets/production/processed/candidate_dataset.jsonl")
        if default_input.is_file():
            input_path = default_input
        else:
            print("❌ Error: Must specify --input or (--train, --validation, --test).", file=sys.stderr)
            sys.exit(1)

    # Initialize QA Engine
    engine = ProductionQAEngine(config_path=args.config)
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            engine._tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            engine._tokenizer_loaded = True
            engine._tokenizer_status = f"LOADED ({args.tokenizer})"
        except Exception as e:
            engine._tokenizer_status = f"TOKEN_ANALYSIS_UNAVAILABLE ({e})"

    records = []
    if input_path and input_path.is_file():
        records = engine._load_records_from_source(input_path)
    elif train_path and train_path.is_file():
        records = engine._load_records_from_source(train_path)
        if val_path and val_path.is_file():
            records.extend(engine._load_records_from_source(val_path))
        if test_path and test_path.is_file():
            records.extend(engine._load_records_from_source(test_path))

    # Auto-split if requested and input provided
    if args.auto_split and input_path and not (train_path and val_path and test_path):
        splitter = DatasetSplitter(
            train_ratio=0.90,
            validation_ratio=0.05,
            test_ratio=0.05,
            random_seed=42,
            stratify_by_domain=True,
        )
        split_res = splitter.split(records)
        processed_dir = input_path.parent
        train_path = processed_dir / "train.jsonl"
        val_path = processed_dir / "validation.jsonl"
        test_path = processed_dir / "test.jsonl"

        with open(train_path, "w", encoding="utf-8") as f:
            for r in split_res.train:
                f.write(r.model_dump_json() + "\n")
        with open(val_path, "w", encoding="utf-8") as f:
            for r in split_res.validation:
                f.write(r.model_dump_json() + "\n")
        with open(test_path, "w", encoding="utf-8") as f:
            for r in split_res.test:
                f.write(r.model_dump_json() + "\n")
        print(f"📦 Auto-split created: Train ({len(split_res.train)}), Val ({len(split_res.validation)}), Test ({len(split_res.test)})")

    # Run QA
    print("=" * 70)
    print("🔍 RUNNING PRODUCTION DATASET QUALITY ASSURANCE & READINESS EVALUATION")
    print("=" * 70)

    report = engine.run_qa(
        dataset_records=records,
        train_records=train_path,
        val_records=val_path,
        test_records=test_path,
        manifest_path=manifest_path if (manifest_path and manifest_path.is_file()) else None,
    )

    # Save Reports
    saved_reports = engine.save_all_reports(report, out_dir)

    # Save filtered output if requested
    if args.output:
        out_target = Path(args.output)
        out_target.parent.mkdir(parents=True, exist_ok=True)
        min_threshold = args.min_score if args.min_score is not None else 0.80
        accepted_records = []
        rejected_records = []
        for r in records:
            score = getattr(r, "quality_score", None)
            if score is None and hasattr(r, "metadata") and isinstance(r.metadata, dict):
                score = r.metadata.get("quality_score")
            if score is None:
                score = 1.0
            if score >= min_threshold:
                accepted_records.append(r)
            else:
                rejected_records.append(r)

        with open(out_target, "w", encoding="utf-8") as f:
            for r in accepted_records:
                if hasattr(r, "model_dump_json"):
                    f.write(r.model_dump_json() + "\n")
                elif isinstance(r, dict):
                    f.write(json.dumps(r) + "\n")
                else:
                    f.write(str(r) + "\n")
        print(f"📦 Filtered Accepted records saved to: {out_target} ({len(accepted_records)} accepted, {len(rejected_records)} rejected)")

    # Print summary
    print(f"\nDataset Version: {report.dataset_version}")
    print(f"Evaluated Records: {report.record_count:,}")
    print(f"Quality Score Mean: {report.quality_qa.mean:.4f} (P95: {report.quality_qa.p95:.4f})")
    print(f"Provenance Completeness: {report.provenance_qa.provenance_completeness:.2%}")
    print(f"Duplicate Rate: {report.duplicate_qa.duplicate_rate:.2%}")
    if report.leakage_qa:
        print(f"Cross-Split Leaks: {report.leakage_qa.total_exact_leaks} exact, {report.leakage_qa.near_duplicate_leaks} near")
    print(f"Tokenizer Status: {report.token_qa.tokenizer_status}")
    print(f"Total Tokens: {report.token_qa.total_conversation_tokens:,}")

    print("\n--- Readiness Gates Summary ---")
    for g in report.gates:
        icon = "✅ PASS" if g.status == ReadinessStatus.PASS else (
            "⚠️ WARN" if g.status == ReadinessStatus.WARN else "❌ FAIL"
        )
        print(f"  [{icon}] {g.gate}: actual={g.actual} (threshold={g.threshold}) -> {g.message}")

    status_icon = "✅ PASS" if report.overall_readiness == ReadinessStatus.PASS else (
        "⚠️ WARN" if report.overall_readiness == ReadinessStatus.WARN else "❌ FAIL"
    )
    print(f"\n🎯 Overall Readiness: {status_icon}")
    print(f"📁 Reports saved to: {out_dir}/")

    # Handle Freeze
    if args.freeze:
        print("\n" + "=" * 70)
        print("❄️ EXECUTING PRODUCTION DATASET FREEZE PROTOCOL")
        print("=" * 70)
        if not manifest_path or not manifest_path.is_file():
            print(f"❌ Error: Manifest file not found at {manifest_path}. Cannot freeze without manifest.", file=sys.stderr)
            sys.exit(1)

        try:
            man, frozen_report = engine.freeze_dataset(
                manifest_path=manifest_path,
                dataset_records=input_path or records,
                train_records=train_path,
                val_records=val_path,
                test_records=test_path,
                reports_dir=out_dir,
                force=args.force,
            )
            print(f"✅ Dataset Successfully Frozen into state: {man.status}")
            print(f"   Candidate Dataset SHA-256: {frozen_report.freeze_qa.dataset_sha256}")
            if frozen_report.freeze_qa.train_sha256:
                print(f"   Train Split SHA-256:       {frozen_report.freeze_qa.train_sha256}")
            if frozen_report.freeze_qa.val_sha256:
                print(f"   Validation Split SHA-256:  {frozen_report.freeze_qa.val_sha256}")
            if frozen_report.freeze_qa.test_sha256:
                print(f"   Test Split SHA-256:        {frozen_report.freeze_qa.test_sha256}")
            print(f"   Updated Manifest:          {manifest_path}")
        except Exception as e:
            print(f"❌ Freeze Failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
