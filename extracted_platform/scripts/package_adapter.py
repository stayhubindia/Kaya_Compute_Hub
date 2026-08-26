#!/usr/bin/env python3
"""
CLI Utility to Package Fine-Tuned QLoRA Adapter for Release (Phase 5.1).
Supports `--adapter-path`, `--output-dir`, `--config`, and `--dry-run`.

Usage:
    python scripts/package_adapter.py --dry-run
    python scripts/package_adapter.py --adapter-path checkpoints/best --output-dir releases/qwen3-4b-qlora-v1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.release.packager import ReleasePackager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5.1 — QLoRA Adapter Packaging & Release CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Path to trained adapter checkpoint directory (containing adapter_config.json and weights)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="releases",
        help="Target parent directory or release path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/release.yaml",
        help="Path to release configuration YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify configuration, paths, and metadata schemas without copying model weights",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packager = ReleasePackager(config_path=args.config)

    print("=" * 70)
    print("Phase 5.1: QLoRA Adapter Packaging System")
    print(f"Config: {args.config}")
    print(f"Dry Run: {args.dry_run}")
    print("=" * 70)

    if args.dry_run:
        dry_res = packager.execute_dry_run(target_dir=args.output_dir if args.output_dir != "releases" else None)
        print("\n[✓] DRY-RUN VALIDATION COMPLETE")
        print(f"  - Release ID: {dry_res['release_id']}")
        print(f"  - Base Model: {dry_res['base_model']}")
        print(f"  - Dataset Version: {dry_res['dataset_version']} (Status: {dry_res['dataset_provenance_status']})")
        print(f"  - Training Config Hash: {dry_res['training_config_hash'][:16]}...")
        print(f"  - Target Directory: {dry_res['expected_release_dir']}")
        print(f"  - Status: {dry_res['status']}")
        print(f"  - Message: {dry_res['message']}")
        return 0

    if not args.adapter_path:
        print("[✗] Error: --adapter-path is required when not in --dry-run mode.")
        return 1

    adapter_src = Path(args.adapter_path)
    if not adapter_src.exists():
        print(f"[✗] Error: Adapter source path does not exist: {adapter_src}")
        return 1

    print(f"[*] Packaging adapter from: {adapter_src}")
    success, manifest, errors = packager.package(
        adapter_source_dir=adapter_src,
        output_dir=args.output_dir,
        dry_run=False,
    )

    if success:
        print("\n[✓] PACKAGING SUCCESSFUL")
        print(f"  - Release ID: {manifest.release_id}")
        print(f"  - Status: {manifest.status.value}")
        print(f"  - Artifacts: {len(manifest.artifact_inventory)} files")
        print(f"  - Output Path: releases/{manifest.release_id}/")
        return 0
    else:
        print("\n[✗] PACKAGING FAILED")
        print(f"  - Release ID: {manifest.release_id}")
        print(f"  - Status: {manifest.status.value}")
        print(f"  - Reason: {manifest.status_reason}")
        for err in errors:
            print(f"    - {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
