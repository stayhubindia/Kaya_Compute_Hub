#!/usr/bin/env python3
"""
Full Workspace Sync Payload Generator for Google Colab.
Collects all code, configs, benchmarks, datasets, reports, releases, instruction datasets, and ingested chunks.
Excludes huge 8GB intermediate scrapes (documents/sections) and .venv to keep zip size optimal (~300-350 MB).
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INCLUDE_PATTERNS = [
    "configs/**/*",
    "src/**/*",
    "scripts/**/*",
    "benchmarks/**/*",
    "datasets/**/*",
    "reports/**/*",
    "releases/**/*",
    "data/instruction_dataset/**/*",
    "data/fixtures/**/*",
    "*.md",
    "*.json",
    "*.ini",
    ".env",
    "train.py",
]

EXCLUDE_PATTERNS = [
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.git/**",
    "**/.venv/**",
    "**/.antigravity/**",
    "**/.pytest_cache/**",
    "Colab/**",
    "data/*.zip",
    "data/instruction_dataset/v3.0/raw/**",
]


def generate_full_sync_payload() -> Path:
    zip_path = PROJECT_ROOT / "Colab/full_workspace_v3.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    files_packed = 0
    seen_paths = set()
    
    print("📦 Packing full workspace into comprehensive ZIP payload...")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for pattern in INCLUDE_PATTERNS:
            for file_path in PROJECT_ROOT.glob(pattern):
                if file_path.is_file():
                    rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
                    
                    if rel_path in seen_paths:
                        continue
                    if any(file_path.match(ex) for ex in EXCLUDE_PATTERNS):
                        continue
                    
                    zipf.write(file_path, arcname=rel_path)
                    seen_paths.add(rel_path)
                    files_packed += 1
                    if files_packed % 50 == 0:
                        print(f"  ... packed {files_packed} files")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"✅ Successfully packed {files_packed} files into full ZIP payload: {zip_path.name} ({size_mb:.2f} MB)")
    
    # Also copy to workspace_sync.zip and dataset_v3_candidates_payload.zip for convenience
    alias_path = PROJECT_ROOT / "Colab/workspace_sync.zip"
    with open(zip_path, 'rb') as fsrc:
        with open(alias_path, 'wb') as fdst:
            fdst.write(fsrc.read())
            
    print(f"📋 Synced alias to {alias_path.name}")
    return zip_path


if __name__ == "__main__":
    generate_full_sync_payload()
