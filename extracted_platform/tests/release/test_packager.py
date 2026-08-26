"""
Unit tests for ReleasePackager and dry-run packaging workflow (Phase 5.1).
"""

import json
from pathlib import Path
import pytest

from src.release.manifest import ReleaseStatus
from src.release.packager import ReleasePackager


def test_packager_dry_run():
    packager = ReleasePackager(config_path="configs/release.yaml")
    dry_res = packager.execute_dry_run()

    assert dry_res["dry_run"] is True
    assert dry_res["release_id"] == "qwen3-4b-qlora-v1.0"
    assert dry_res["dataset_version"] == "dataset-v1.0"
    assert dry_res["dataset_provenance_status"] == "VERIFIED"
    assert dry_res["status"] == "VALIDATED"


def test_packager_clean_failure_on_missing_adapter(tmp_path: Path):
    packager = ReleasePackager(config_path="configs/release.yaml")
    success, manifest, errors = packager.package(
        adapter_source_dir=tmp_path / "non_existent_adapter",
        output_dir=tmp_path / "releases",
        dry_run=False,
    )
    assert success is False
    assert manifest.status == ReleaseStatus.INVALID
    assert len(errors) > 0


def test_packager_successful_packaging_from_simulated_checkpoint(tmp_path: Path):
    # Simulate a trained checkpoint directory
    ckpt_dir = tmp_path / "checkpoint-100"
    ckpt_dir.mkdir()

    adapter_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "bias": "none",
        "base_model_name_or_path": "Qwen/Qwen3-4B-Base",
    }
    with open(ckpt_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_cfg, f)
    with open(ckpt_dir / "adapter_model.safetensors", "wb") as f:
        f.write(b"simulated_trained_adapter_weights")

    out_releases = tmp_path / "releases"
    packager = ReleasePackager(config_path="configs/release.yaml")

    success, manifest, errors = packager.package(
        adapter_source_dir=ckpt_dir,
        output_dir=out_releases,
        dry_run=False,
    )

    assert success is True
    assert manifest.status == ReleaseStatus.READY
    assert len(errors) == 0

    release_dir = out_releases / manifest.release_id
    assert (release_dir / "manifest.json").exists()
    assert (release_dir / "checksums.sha256").exists()
    assert (release_dir / "MODEL_CARD.md").exists()
    assert (release_dir / "reproducibility.json").exists()
    assert (release_dir / "adapter" / "adapter_model.safetensors").exists()
