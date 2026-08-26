"""
Unit tests for ReleaseValidator (Phase 5.1).
"""

import json
from pathlib import Path
import pytest

from src.release.integrity import ReleaseIntegrityManager
from src.release.manifest import ReleaseManifest, ReleaseStatus
from src.release.model_card import ModelCardGenerator, ReadmeGenerator
from src.release.provenance import DatasetProvenance, HardwareProvenance, TrainingProvenance
from src.release.reproducibility import ReproducibilityManager
from src.release.validator import ReleaseValidator


def test_validator_detects_non_existent_release(tmp_path: Path):
    validator = ReleaseValidator()
    report = validator.validate_release(tmp_path / "does_not_exist")
    assert report.is_valid is False
    assert len(report.errors) > 0


def test_validator_valid_release_bundle(tmp_path: Path):
    rel_dir = tmp_path / "qwen3-4b-qlora-v1.0"
    rel_dir.mkdir()

    # 1. Adapter dir
    adapter_dir = rel_dir / "adapter"
    adapter_dir.mkdir()
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
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_cfg, f)
    with open(adapter_dir / "adapter_model.safetensors", "wb") as f:
        f.write(b"mock_weights")

    # 2. Compatibility
    comp_data = {
        "target_base_model": "Qwen/Qwen3-4B-Base",
        "target_architectures": ["Qwen3ForCausalLM"],
    }
    with open(rel_dir / "compatibility.json", "w") as f:
        json.dump(comp_data, f)

    # 3. Provenance
    prov_data = {
        "dataset_provenance": {
            "dataset_version": "dataset-v1.0",
            "provenance_status": "VERIFIED",
        },
        "training_provenance": {
            "config_hash": "dummy_train_hash",
        },
        "hardware_provenance": {},
    }
    with open(rel_dir / "provenance.json", "w") as f:
        json.dump(prov_data, f)

    # 4. Reproducibility
    repro_data = {
        "random_seed": 42,
        "training_config_hash": "dummy_train_hash",
    }
    with open(rel_dir / "reproducibility.json", "w") as f:
        json.dump(repro_data, f)

    # 5. Manifest
    manifest = ReleaseManifest(
        release_id="qwen3-4b-qlora-v1.0",
        release_version="v1.0",
        status=ReleaseStatus.READY,
        dataset_version="dataset-v1.0",
        training_config_hash="dummy_train_hash",
    )
    manifest.save_atomic(rel_dir / "manifest.json")

    # 6. Model Card & README
    card = ModelCardGenerator.generate_model_card(manifest)
    with open(rel_dir / "MODEL_CARD.md", "w") as f:
        f.write(card)
    readme = ReadmeGenerator.generate_readme(manifest)
    with open(rel_dir / "README.md", "w") as f:
        f.write(readme)

    # 7. Checksums
    ReleaseIntegrityManager.generate_checksums_file(rel_dir)

    # Run audit
    validator = ReleaseValidator()
    report = validator.validate_release(rel_dir)

    assert report.is_valid is True
    assert report.status == "READY"
    assert report.manifest_valid is True
    assert report.adapter_valid is True
    assert report.integrity_valid is True
    assert report.model_card_valid is True
