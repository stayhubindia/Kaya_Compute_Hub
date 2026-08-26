#!/usr/bin/env python3
"""
pack_platform_integration.py
==============================
Creates a comprehensive, platform-ready ZIP of the entire LLM training pipeline.

This script bundles:
  - Full source code (src/ modules)
  - All pipeline scripts (scripts/)
  - Configurations (configs/)
  - Tests (tests/)
  - Documentation (*.md)
  - Pipeline entry-points and manifests

Usage:
    python scripts/pack_platform_integration.py [--output <path>] [--include-datasets]

Output:
    platform_integration_<YYYYMMDD_HHMMSS>.zip
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


# ── Project root (one level above this script) ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── What to include in the ZIP ─────────────────────────────────────────────────
INCLUDE_DIRS = [
    "src",          # Core Python modules (ingestion, dataset, generation, training, …)
    "scripts",      # CLI entry-point scripts
    "configs",      # YAML config files + spec docs
    "tests",        # Unit & integration tests
]

INCLUDE_FILES = [
    "END_TO_END_PIPELINE_GUIDE.md",
    "MULTI_ACCOUNT_COLAB_GUIDE.md",
    "pytest.ini",
    "requirements.txt",
]

# Directories / patterns to always SKIP
EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".venv",
    ".git",
    ".antigravity",
    "node_modules",
    "Colab",          # Large raw Colab backup ZIPs
}

EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".egg-info",
    ".DS_Store", ".log",
}

EXCLUDE_FILES = {
    # Already very large sync payloads — not needed for integration
    "sync_full_project.py",
    "init_remote_data.py",
}


def should_skip(path: Path) -> bool:
    """Return True if this path should be excluded from the ZIP."""
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    if path.suffix in EXCLUDE_EXTENSIONS:
        return True
    if path.name in EXCLUDE_FILES:
        return True
    return False


def collect_files(include_datasets: bool = False) -> list[tuple[Path, str]]:
    """
    Walk the project and collect (absolute_path, zip_arcname) tuples.
    All paths inside the ZIP are relative to project root so they
    unpack cleanly into a single folder.
    """
    collected: list[tuple[Path, str]] = []

    # ── Source directories ─────────────────────────────────────────────────────
    for rel_dir in INCLUDE_DIRS:
        abs_dir = PROJECT_ROOT / rel_dir
        if not abs_dir.exists():
            print(f"  [WARN] Directory not found, skipping: {rel_dir}")
            continue
        for fpath in abs_dir.rglob("*"):
            if fpath.is_file() and not should_skip(fpath.relative_to(PROJECT_ROOT)):
                arcname = str(fpath.relative_to(PROJECT_ROOT))
                collected.append((fpath, arcname))

    # ── Top-level files ────────────────────────────────────────────────────────
    for fname in INCLUDE_FILES:
        fpath = PROJECT_ROOT / fname
        if fpath.exists():
            collected.append((fpath, fname))

    # ── Dataset splits (optional, can be large) ───────────────────────────────
    if include_datasets:
        for version_dir in (PROJECT_ROOT / "data" / "instruction_dataset").iterdir():
            if version_dir.is_dir():
                for fpath in version_dir.rglob("*.jsonl"):
                    arcname = str(fpath.relative_to(PROJECT_ROOT))
                    collected.append((fpath, arcname))
                # Include freeze manifest if present
                for fpath in version_dir.rglob("*.json"):
                    if "manifest" in fpath.name or "freeze" in fpath.name:
                        arcname = str(fpath.relative_to(PROJECT_ROOT))
                        collected.append((fpath, arcname))

    return collected


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(files: list[tuple[Path, str]], zip_path: Path, include_datasets: bool) -> dict:
    """Build a JSON manifest describing the package contents."""
    file_entries = []
    total_bytes = 0
    for fpath, arcname in files:
        size = fpath.stat().st_size
        total_bytes += size
        file_entries.append({
            "path": arcname,
            "size_bytes": size,
            "sha256": sha256_file(fpath),
        })

    return {
        "package_name": zip_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v4.0",
        "includes_datasets": include_datasets,
        "total_files": len(files),
        "total_size_bytes": total_bytes,
        "modules": {
            "ingestion":    "src/ingestion/   — PDF/HTML/TXT extraction & semantic chunking",
            "generation":   "src/generation/  — Synthetic QA & instruction pair generation",
            "dataset":      "src/dataset/     — Quality audit, dedup, leakage guard, freeze",
            "training":     "src/training/    — QLoRA SFT trainer, config, tokenizer",
            "evaluation":   "src/evaluation/  — Benchmark runner, metrics, regression tests",
            "distribution": "src/distribution/— HuggingFace upload & package auditing",
            "release":      "src/release/     — Adapter packaging, model card, provenance",
            "panel":        "src/panel/       — Dashboard / monitoring helpers",
        },
        "entry_points": {
            "ingest_documents":           "scripts/ingest_documents.py",
            "generate_instruction_dataset": "scripts/generate_instruction_dataset.py",
            "build_dataset_v2":           "scripts/build_dataset_v2.py",
            "process_documents_to_dataset": "scripts/process_documents_to_dataset.py",
            "train_production_v2":        "scripts/train_production_v2.py",
            "run_colab_job":              "scripts/run_colab_job.py",
            "colab_account_manager":      "scripts/colab_account_manager.py",
            "monitor_training":           "scripts/monitor_training.py",
            "chat_inference":             "scripts/chat_inference.py",
            "pack_sync_payload":          "scripts/pack_sync_payload.py",
            "evaluate_model":             "scripts/evaluate_model.py",
            "distribute_huggingface":     "scripts/distribute_huggingface.py",
        },
        "files": file_entries,
    }


def create_zip(output_path: Path, files: list[tuple[Path, str]], manifest: dict) -> None:
    """Write all files + embedded manifest into the ZIP."""
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        total = len(files)
        for idx, (fpath, arcname) in enumerate(files, 1):
            # Progress indicator
            bar_len = 40
            filled = int(bar_len * idx / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {idx}/{total}  {arcname[:55]:<55}", end="", flush=True)
            zf.write(fpath, arcname)

        # Embed the manifest as PIPELINE_MANIFEST.json at the root of the ZIP
        manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        zf.writestr("PIPELINE_MANIFEST.json", manifest_bytes)

    print()  # newline after progress bar


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the full LLM training pipeline for platform integration."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output ZIP file path. Defaults to project root.",
    )
    parser.add_argument(
        "--include-datasets",
        action="store_true",
        default=False,
        help="Include instruction dataset JSONL files (can be large).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"platform_integration_{timestamp}.zip"
    output_path = args.output if args.output else (PROJECT_ROOT / zip_name)

    print("\n" + "=" * 65)
    print("  🚀  LLM Pipeline — Platform Integration Packager")
    print("=" * 65)
    print(f"  Project root    : {PROJECT_ROOT}")
    print(f"  Output ZIP      : {output_path}")
    print(f"  Include datasets: {args.include_datasets}")
    print("=" * 65)

    print("\n[1/4] Collecting files …")
    t0 = time.time()
    files = collect_files(include_datasets=args.include_datasets)
    print(f"      → {len(files)} files collected in {time.time() - t0:.1f}s")

    print("\n[2/4] Building manifest …")
    manifest = build_manifest(files, output_path, args.include_datasets)
    print(f"      → Total uncompressed: {manifest['total_size_bytes'] / 1_048_576:.1f} MB")

    print("\n[3/4] Writing ZIP …")
    t1 = time.time()
    create_zip(output_path, files, manifest)
    zip_size = output_path.stat().st_size
    print(f"      → Done in {time.time() - t1:.1f}s  |  ZIP size: {zip_size / 1_048_576:.1f} MB")

    print("\n[4/4] Computing ZIP checksum …")
    zip_sha = sha256_file(output_path)
    print(f"      → SHA-256: {zip_sha}")

    # Write a sidecar checksum file
    checksum_path = output_path.with_suffix(".sha256")
    checksum_path.write_text(f"{zip_sha}  {output_path.name}\n")

    print("\n" + "=" * 65)
    print("  ✅  Package created successfully!")
    print(f"  📦  {output_path.name}")
    print(f"  🔐  {checksum_path.name}")
    print(f"  📋  Manifest embedded as PIPELINE_MANIFEST.json inside ZIP")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
