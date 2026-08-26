#!/usr/bin/env python3
"""
CLI Utility to Generate/Refresh MODEL_CARD.md and README.md for a Release (Phase 5.1).

Usage:
    python scripts/create_model_card.py --release releases/qwen3-4b-qlora-v1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.release.manifest import ReleaseManifest
from src.release.model_card import ModelCardGenerator, ReadmeGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5.1 — Generate/Refresh MODEL_CARD.md and README.md",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--release",
        type=str,
        required=True,
        help="Path or name of release directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rel_path = Path(args.release) if Path(args.release).is_dir() else Path("releases") / args.release

    if not rel_path.exists():
        print(f"[✗] Release directory not found: {rel_path}")
        return 1

    man_path = rel_path / "manifest.json"
    if not man_path.exists():
        print(f"[✗] manifest.json not found in: {rel_path}")
        return 1

    manifest = ReleaseManifest.load(man_path)
    card_content = ModelCardGenerator.generate_model_card(manifest)
    readme_content = ReadmeGenerator.generate_readme(manifest)

    with open(rel_path / "MODEL_CARD.md", "w", encoding="utf-8") as f:
        f.write(card_content)

    with open(rel_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"[✓] Successfully generated MODEL_CARD.md and README.md in {rel_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
