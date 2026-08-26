#!/usr/bin/env python3
"""
Sync & Resumption Payload Generator for Google Colab.
Collects configs, sources, scripts, frozen datasets, and optionally restored checkpoint
binaries (checkpoint-1450) into a ZIP payload for Google Colab execution.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INCLUDE_PATTERNS_BASE = [
    "configs/*.yaml",
    "configs/**/*.yaml",
    "configs/*.md",
    "configs/**/*.md",
    "src/**/*.py",
    "scripts/**/*.py",
    "data/instruction_dataset/v2.0/splits/*.jsonl",
    "data/instruction_dataset/v2.0/manifests/*",
    "data/instruction_dataset/v2.0/*.sha256",
    "data/instruction_dataset/v3.0/splits/*.jsonl",
    "data/instruction_dataset/v3.0/manifests/*",
    "data/instruction_dataset/v3.0/*.sha256",
    "data/instruction_dataset/v3.0/*.jsonl",
    "data/instruction_dataset/v3.0/raw/*.jsonl",
    "data/fixtures/**/*",
]

CHECKPOINT_PATTERNS = [
    "outputs/training/dataset-v3.0/qlora-v3/production/checkpoints/**/*",
]

EXCLUDE_PATTERNS_BASE = [
    "**/__pycache__/**",
    "**/*.pyc",
    "scripts/sync_full_project.py",
    "**/.git/**",
    "models/**/*.safetensors",
    "**/*.bin",
    "**/*.pt",
    "**/*.zip",
]


def generate_sync_payload(include_checkpoints: bool = False, output_name: str = "workspace_sync.zip") -> Path:
    zip_path = PROJECT_ROOT / "data" / output_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    include_patterns = list(INCLUDE_PATTERNS_BASE)
    if include_checkpoints:
        include_patterns.extend(CHECKPOINT_PATTERNS)

    exclude_patterns = list(EXCLUDE_PATTERNS_BASE)
    if include_checkpoints:
        # Allow safetensors inside outputs/training checkpoint directories
        exclude_patterns = [p for p in exclude_patterns if "safetensors" not in p and "bin" not in p]

    files_packed = 0
    seen_paths = set()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for pattern in include_patterns:
            for file_path in PROJECT_ROOT.glob(pattern):
                if file_path.is_file():
                    rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()

                    if rel_path in seen_paths:
                        continue
                    if any(file_path.match(ex) for ex in exclude_patterns):
                        continue

                    zipf.write(file_path, arcname=rel_path)
                    seen_paths.add(rel_path)
                    files_packed += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"📦 Packed {files_packed} project files into ZIP payload: {zip_path.name} ({size_mb:.2f} MB)")
    return zip_path


def parse_args():
    parser = argparse.ArgumentParser(description="Pack workspace sync & checkpoint resumption zip payload for Google Colab.")
    parser.add_argument("--include-checkpoints", action="store_true", help="Include restored Step 1450 checkpoint weights in zip payload.")
    parser.add_argument("--output-name", type=str, default=None, help="Name of destination zip file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_name = args.output_name or ("workspace_sync_resume_1450.zip" if args.include_checkpoints else "workspace_sync.zip")
    generate_sync_payload(include_checkpoints=args.include_checkpoints, output_name=out_name)
