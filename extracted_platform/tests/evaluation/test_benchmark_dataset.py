"""
Unit tests for Benchmark Dataset Manager, Manifest, Statistics & CLI Tools (Phase 4.5).
"""

import json
from pathlib import Path
import pytest

from src.dataset.schema import Message, Role
from src.evaluation.benchmark_cases import BenchmarkCase, BenchmarkSuiteBuilder
from src.evaluation.benchmark_dataset import (
    BenchmarkDatasetManager,
    BenchmarkManifest,
    BenchmarkStatistics,
)


def test_manifest_serialization(tmp_path: Path):
    manifest = BenchmarkManifest(
        benchmark_version="benchmark-v1.0",
        case_count=10,
        lifecycle_status="FROZEN",
        benchmark_sha256="dummy_sha256",
        config_hash="dummy_cfg_hash",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    assert manifest_path.exists()

    loaded = BenchmarkManifest.load(manifest_path)
    assert loaded.benchmark_version == "benchmark-v1.0"
    assert loaded.case_count == 10
    assert loaded.lifecycle_status == "FROZEN"
    assert loaded.benchmark_sha256 == "dummy_sha256"


def test_statistics_percentiles_and_breakdowns():
    cases = BenchmarkSuiteBuilder.generate_benchmark_suite(target_count=15, seed=42)
    stats = BenchmarkDatasetManager.compute_statistics(cases)

    assert stats.total_cases == 15
    assert stats.prompt_tokens.mean > 0
    assert stats.prompt_tokens.p50 > 0
    assert stats.prompt_tokens.max >= stats.prompt_tokens.min
    assert len(stats.domains) > 0
    assert len(stats.difficulties) > 0
    assert stats.turn_types["single_turn"] + stats.turn_types["multi_turn"] == 15


def test_save_and_load_benchmark_bundle(tmp_path: Path):
    cases = BenchmarkSuiteBuilder.generate_benchmark_suite(target_count=10, seed=42)
    jsonl_path, manifest, stats = BenchmarkDatasetManager.save_benchmark(
        cases=cases,
        base_dir=tmp_path,
        config_hash="test_config_hash",
        generation_config={"temperature": 0.0, "seed": 42},
    )

    assert jsonl_path.exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "statistics.json").exists()
    assert (tmp_path / "README.md").exists()
    assert manifest.lifecycle_status == "FROZEN"
    assert manifest.case_count == 10
    assert len(manifest.benchmark_sha256) == 64

    # Load and verify
    loaded_cases, loaded_manifest, loaded_stats = BenchmarkDatasetManager.load_benchmark(tmp_path)
    assert len(loaded_cases) == 10
    assert loaded_manifest.benchmark_sha256 == manifest.benchmark_sha256


def test_tampered_benchmark_detection(tmp_path: Path):
    cases = BenchmarkSuiteBuilder.generate_benchmark_suite(target_count=5, seed=42)
    jsonl_path, manifest, _ = BenchmarkDatasetManager.save_benchmark(
        cases=cases,
        base_dir=tmp_path,
    )

    # Tamper with file
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write("\n")

    with pytest.raises(ValueError, match="Benchmark SHA-256 mismatch"):
        BenchmarkDatasetManager.load_benchmark(tmp_path)
