"""
Unit tests for ProvenanceCollector (dataset, training, and hardware) (Phase 5.1).
"""

from pathlib import Path
import pytest

from src.release.provenance import ProvenanceCollector


def test_collect_dataset_provenance():
    prov = ProvenanceCollector.collect_dataset_provenance(
        manifest_path="datasets/production/manifests/production_manifest.json",
        dataset_dir="datasets/production/processed",
    )
    assert prov.dataset_version == "dataset-v1.0"
    assert prov.lifecycle_status == "FROZEN"
    assert prov.provenance_status == "VERIFIED"
    assert "train.jsonl" in prov.split_hashes


def test_collect_training_provenance():
    prov = ProvenanceCollector.collect_training_provenance("configs/training.yaml")
    assert prov.base_model_name == "Qwen/Qwen3-4B-Base"
    assert prov.optimizer == "paged_adamw_8bit"
    assert prov.learning_rate == 2.0e-4
    assert prov.effective_batch_size == 8
    assert len(prov.config_hash) == 64


def test_collect_hardware_provenance_pre_training():
    prov = ProvenanceCollector.collect_hardware_provenance(telemetry=None)
    assert prov.status == "NOT_AVAILABLE"
    assert prov.gpu_name == "NOT_AVAILABLE"
    assert prov.gpu_count == 0
