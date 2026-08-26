#!/usr/bin/env python3
"""
CLI Tool for Dataset-v2.0 Final QA, Audit, Verification, and Freeze (Phase 3.5).

Usage:
  python scripts/finalize_dataset.py --audit
  python scripts/finalize_dataset.py --report
  python scripts/finalize_dataset.py --verify
  python scripts/finalize_dataset.py --freeze
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.final_qa_auditor import (
    FinalQAAuditor,
    GateStatus,
    LifecycleState,
)


def verify_checksums(manifest_dir: Path, dataset_dir: Path) -> bool:
    """Verifies all hashes listed in checksums.sha256."""
    chk_file = manifest_dir / "checksums.sha256"
    if not chk_file.is_file():
        print(f"❌ Checksum file not found: {chk_file}")
        return False

    print(f"🔒 Verifying checksums from {chk_file}...")
    all_match = True

    with open(chk_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                expected_hash = parts[0]
                rel_path = " ".join(parts[1:])
                target_file = dataset_dir / rel_path

                if not target_file.is_file():
                    print(f"  ❌ Missing file: {rel_path}")
                    all_match = False
                    continue

                h = hashlib.sha256()
                with open(target_file, "rb") as tf:
                    while chunk := tf.read(65536):
                        h.update(chunk)
                actual_hash = h.hexdigest()

                if actual_hash == expected_hash:
                    print(f"  ✅ [MATCH] {rel_path} ({actual_hash[:12]}...)")
                else:
                    print(f"  ❌ [MISMATCH] {rel_path} (Expected {expected_hash[:12]}..., got {actual_hash[:12]}...)")
                    all_match = False

    return all_match


def update_checksums(dataset_dir: Path, output_dir: Path) -> dict:
    """Computes and writes updated SHA-256 checksums."""
    files_to_hash = {
        "splits/train.jsonl": dataset_dir / "splits" / "train.jsonl",
        "splits/validation.jsonl": dataset_dir / "splits" / "validation.jsonl",
        "splits/test.jsonl": dataset_dir / "splits" / "test.jsonl",
        "processed/accepted.jsonl": dataset_dir / "processed" / "accepted.jsonl",
        "processed/rejected.jsonl": dataset_dir / "processed" / "rejected.jsonl",
        "raw/candidates.jsonl": dataset_dir / "raw" / "candidates.jsonl",
        "combined_candidates.jsonl": dataset_dir / "combined_candidates.jsonl",
        "manifests/dataset_manifest.json": dataset_dir / "manifests" / "dataset_manifest.json",
    }
    checksums = {}
    for name, p in sorted(files_to_hash.items()):
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            checksums[name] = h.hexdigest()

    for dest_dir in [dataset_dir / "manifests", output_dir]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(dest_dir / "checksums.sha256", "w", encoding="utf-8") as f:
            for name, h in sorted(checksums.items()):
                f.write(f"{h}  {name}\n")
    return checksums


def main():
    parser = argparse.ArgumentParser(
        description="Dataset-v2.0 Final QA, Audit, Verification and Freeze Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audit", action="store_true", help="Run full 15-gate QA audit and print console summary")
    parser.add_argument("--report", action="store_true", help="Generate all JSON & Markdown reports in reports/final_qa/")
    parser.add_argument("--verify", action="store_true", help="Verify cryptographic SHA-256 checksums against dataset files")
    parser.add_argument("--freeze", action="store_true", help="Transition dataset lifecycle to FROZEN (requires all critical gates to PASS)")
    parser.add_argument("--version", type=str, default="v2.0", help="Dataset version identifier (e.g. v2.0)")
    parser.add_argument("--dataset-dir", type=str, default="data/instruction_dataset/v2.0", help="Path to dataset-v2.0 root directory")
    parser.add_argument("--corpus-dir", type=str, default="data/ingested/nptel_corpus", help="Path to ingested NPTEL corpus directory")
    parser.add_argument("--output-dir", type=str, default="reports/final_qa", help="Output directory for generated QA reports")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")

    args = parser.parse_args()

    # Default action if no flags provided
    if not (args.audit or args.report or args.verify or args.freeze):
        args.audit = True
        args.report = True

    dataset_path = Path(args.dataset_dir).resolve()
    corpus_path = Path(args.corpus_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    print("=" * 80)
    print("🚀 DATASET-v2.0 FINAL QUALITY ASSURANCE & FREEZE PIPELINE (PHASE 3.5)")
    print(f"📍 Dataset Target: {dataset_path}")
    print(f"📍 Output Target:  {output_path}")
    print("=" * 80)

    if args.verify:
        manifest_dir = dataset_path / "manifests"
        success = verify_checksums(manifest_dir, dataset_path)
        if success:
            print("\n✅ All dataset checksums cryptographically verified!")
            sys.exit(0)
        else:
            print("\n❌ Checksum verification failed!")
            sys.exit(1)

    auditor = FinalQAAuditor(
        dataset_dir=dataset_path,
        source_corpus_dir=corpus_path,
        seed=args.seed,
    )

    print("\n🔍 Running 15-Gate Independent QA Audit...")
    report = auditor.run_full_audit()

    print("\n" + "=" * 80)
    print("📊 15-DIMENSION QUALITY GATE SCORECARD")
    print("=" * 80)
    print(f"{'Gate':<6} | {'Dimension':<26} | {'Crit':<5} | {'Status':<6} | {'Score':<7} | {'Evidence'}")
    print("-" * 80)

    for g in report.gate_matrix:
        st_icon = "PASS" if g.status == GateStatus.PASS else ("WARN" if g.status == GateStatus.WARN else "FAIL")
        crit_str = "YES" if g.is_critical else "NO"
        print(f"{g.gate_id:<6} | {g.gate_name:<26} | {crit_str:<5} | {st_icon:<6} | {g.score:>6.2%} | {g.evidence[:45]}...")

    print("-" * 80)
    print(f"📈 Total Input Raw Candidates:    {report.count_reconciliation.raw_candidates:,}")
    print(f"❌ Rejected During Synthesis:     {report.count_reconciliation.rejected_candidates:,}")
    print(f"✨ Accepted Pre-Deduplication:    {report.count_reconciliation.accepted_before_dedup:,}")
    print(f"🧹 Duplicates Removed:            {report.count_reconciliation.total_duplicates_removed:,} ({report.count_reconciliation.exact_duplicates_removed} exact, {report.count_reconciliation.near_duplicates_removed} near)")
    print(f"🏆 Final Unique Dataset Records:  {report.count_reconciliation.final_unique_records:,}")
    print(f"   ├── Train Split (90%):         {report.count_reconciliation.train_records:,}")
    print(f"   ├── Validation Split (5%):     {report.count_reconciliation.validation_records:,}")
    print(f"   └── Test Split (5%):           {report.count_reconciliation.test_records:,}")
    print(f"⭐ Mean Scientific Quality:       {report.quality_audit.mean_score:.4f} (100% >= 0.90)")
    print(f"🎯 Zero Cross-Split Leakage:      {'YES (Zero Leaks)' if report.leakage_audit.is_leak_free else 'NO'}")
    print("=" * 80)

    if args.report or args.freeze or args.audit:
        print(f"\n📝 Writing QA Audit Reports to {output_path}...")
        written = auditor.write_all_reports(report, output_dir=output_path)
        for name, p in sorted(written.items()):
            print(f"  📄 Generated {name}")

    if args.freeze:
        if not report.all_critical_gates_passed:
            print("\n❌ FREEZE REJECTED: Mandatory critical quality gates failed.")
            for g in report.gate_matrix:
                if g.is_critical and g.status != GateStatus.PASS:
                    print(f"  ❌ Critical Gate Failure: {g.gate_id} ({g.gate_name}): {g.failure_reasons}")
            sys.exit(1)

        print("\n🔒 Transitioning Dataset Lifecycle State to FROZEN...")
        report.lifecycle_state = LifecycleState.FROZEN

        # Update dataset_manifest.json with FROZEN state
        ds_manifest_file = dataset_path / "manifests" / "dataset_manifest.json"
        if ds_manifest_file.is_file():
            with open(ds_manifest_file, "r", encoding="utf-8") as f:
                ds_data = json.load(f)
            ds_data["lifecycle_state"] = LifecycleState.FROZEN.value
            with open(ds_manifest_file, "w", encoding="utf-8") as f:
                json.dump(ds_data, f, indent=2)

        # Update final_qa_manifest.json with FROZEN state
        manifest_file = dataset_path / "manifests" / "final_qa_manifest.json"
        if manifest_file.is_file():
            with open(manifest_file, "r", encoding="utf-8") as f:
                mdata = json.load(f)
            mdata["lifecycle_state"] = LifecycleState.FROZEN.value
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(mdata, f, indent=2)

        # Refresh checksums.sha256 with frozen manifest hash
        update_checksums(dataset_path, output_path)

        print(f"✅ DATASET-v2.0 SUCCESSFULLY FROZEN! Manifest updated at {manifest_file}")

    print(f"\n🏁 QA Audit Complete! Dataset Lifecycle State: {report.lifecycle_state.value}")
    sys.exit(0 if report.all_critical_gates_passed else 1)


if __name__ == "__main__":
    main()
