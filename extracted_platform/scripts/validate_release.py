#!/usr/bin/env python3
"""
CLI Utility to Validate QLoRA Adapter Release Integrity & Compatibility (Phase 5.1).

Usage:
    python scripts/validate_release.py --release releases/qwen3-4b-qlora-v1.0
    python scripts/validate_release.py --release qwen3-4b-qlora-v1.0
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

from src.release.validator import ReleaseValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5.1 — QLoRA Release Audit & Verification CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--release",
        type=str,
        required=True,
        help="Path or name of release directory to audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rel_path = Path(args.release) if Path(args.release).is_dir() else Path("releases") / args.release

    print("=" * 70)
    print("Phase 5.1: QLoRA Release Validation Audit")
    print(f"Target Release: {rel_path}")
    print("=" * 70)

    if not rel_path.exists():
        print(f"[✗] Error: Target release directory not found at: {rel_path}")
        return 1

    validator = ReleaseValidator()
    report = validator.validate_release(rel_path)

    print("\n[✓] Audit Stage Breakdown:")
    for check_name, passed in sorted(report.checks_passed.items()):
        status_symbol = "[✓]" if passed else "[✗]"
        print(f"  {status_symbol} {check_name.capitalize()}: {'PASSED' if passed else 'FAILED'}")

    if report.is_valid:
        print("\n" + "=" * 70)
        print(f"RELEASE AUDIT STATUS: READY ({report.release_id})")
        print("All mandatory cryptographic, architectural, and documentation validations passed.")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"RELEASE AUDIT STATUS: INVALID ({report.release_id})")
        print("Errors detected:")
        for err in report.errors:
            print(f"  - {err}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
