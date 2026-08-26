"""
Unit tests for ReleaseManifest and deterministic release ID construction (Phase 5.1).
"""

from pathlib import Path
import pytest

from src.release.manifest import (
    ReleaseManifest,
    ReleaseStatus,
    construct_release_id,
)


def test_deterministic_release_id():
    rel_id = construct_release_id(
        base_model_id="Qwen/Qwen3-4B-Base",
        adapter_version="v1.0",
        dataset_version="dataset-v1.0",
        training_config_hash="abc123hash",
    )
    assert rel_id == "qwen3-4b-qlora-v1.0"

    rel_id_2 = construct_release_id(
        base_model_id="Qwen/Qwen3-4B-Base",
        adapter_version="1.0",
        dataset_version="dataset-v1.0",
        training_config_hash="abc123hash",
    )
    assert rel_id_2 == "qwen3-4b-qlora-v1.0"


def test_release_manifest_lifecycle_and_atomic_save(tmp_path: Path):
    manifest = ReleaseManifest(
        release_id="qwen3-4b-qlora-v1.0",
        status=ReleaseStatus.PLANNED,
        dataset_version="dataset-v1.0",
        training_config_hash="dummy_hash",
    )
    assert manifest.status == ReleaseStatus.PLANNED
    assert manifest.baseline_experiment_id == "NOT_AVAILABLE"

    # Save and reload
    man_path = tmp_path / "manifest.json"
    manifest.save_atomic(man_path)
    assert man_path.exists()

    loaded = ReleaseManifest.load(man_path)
    assert loaded.release_id == "qwen3-4b-qlora-v1.0"
    assert loaded.status == ReleaseStatus.PLANNED
