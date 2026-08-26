"""
Tests for Training Dataset Loader & Split Isolation (Phase 4.1).
"""

import json
import pytest
from pathlib import Path

from src.dataset.production import DatasetFreezeState, ProductionManifest
from src.dataset.schema import DatasetRecord, Message, Role, SourceType, TaskType, DifficultyLevel
from src.training.config import DatasetConfig
from src.training.dataset import (
    DatasetIntegrityError,
    SplitIsolationError,
    TrainingDatasetLoader,
)
from src.training.utils import compute_file_sha256


@pytest.fixture
def sample_valid_record():
    return DatasetRecord(
        messages=[
            Message(role=Role.USER, content="How do indexes work in SQL?"),
            Message(role=Role.ASSISTANT, content="Indexes speed up data retrieval by creating lookup trees like B-Trees."),
        ],
        metadata={
            "domain": "software_engineering",
            "topic": "sql_indexing",
            "task_type": TaskType.QUESTION_ANSWERING,
            "difficulty": DifficultyLevel.BEGINNER,
            "provenance": {
                "source_name": "unit_test",
                "source_type": SourceType.SYNTHETIC,
                "created_at": "2026-08-12T00:00:00Z",
                "pipeline_version": "2.2",
            },
        },
    )


def test_production_frozen_dataset_loading():
    """Verify loading the actual frozen production dataset."""
    cfg = DatasetConfig(
        version="dataset-v1.0",
        manifest_path="datasets/production/manifests/production_manifest.json",
        train_file="datasets/production/processed/train.jsonl",
        validation_file="datasets/production/processed/validation.jsonl",
        test_file="datasets/production/processed/test.jsonl",
        require_frozen=True,
        validate_sha256=True,
    )
    loader = TrainingDatasetLoader(cfg)
    manifest = loader.load_manifest()
    assert manifest.status == DatasetFreezeState.FROZEN.value

    train_ds, val_ds, test_ds = loader.load_splits()
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    assert len(train_ds) + len(val_ds) + len(test_ds) == 59


def test_reject_non_frozen_dataset(tmp_path):
    """Ensure training is rejected if dataset is not FROZEN."""
    manifest_path = tmp_path / "manifest.json"
    manifest_data = {
        "dataset_version": "dataset-test",
        "status": "VALIDATING",  # Not FROZEN
        "target_count": 100,
        "candidate_target": 100,
        "seed": 42,
        "domain_targets": {"software_engineering": 1.0},
        "difficulty_targets": {"beginner": 1.0},
        "batch_size": 10,
        "batch_count": 10,
        "created_at": "2026-08-12T00:00:00Z",
        "checksums": {},
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)

    cfg = DatasetConfig(
        version="dataset-test",
        manifest_path=str(manifest_path),
        require_frozen=True,
    )
    loader = TrainingDatasetLoader(cfg)
    with pytest.raises(DatasetIntegrityError, match="Only FROZEN datasets are accepted"):
        loader.load_manifest()


def test_reject_tampered_checksum(tmp_path, sample_valid_record):
    """Ensure tampered files with mismatched checksums are rejected."""
    train_file = tmp_path / "train.jsonl"
    with open(train_file, "w") as f:
        f.write(sample_valid_record.model_dump_json() + "\n")

    manifest_path = tmp_path / "manifest.json"
    manifest_data = {
        "dataset_version": "dataset-test",
        "status": "FROZEN",
        "target_count": 1,
        "candidate_target": 1,
        "seed": 42,
        "domain_targets": {"software_engineering": 1.0},
        "difficulty_targets": {"beginner": 1.0},
        "batch_size": 1,
        "batch_count": 1,
        "created_at": "2026-08-12T00:00:00Z",
        "checksums": {"train.jsonl": "0000000000000000000000000000000000000000000000000000000000000000"},
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)

    cfg = DatasetConfig(
        version="dataset-test",
        manifest_path=str(manifest_path),
        train_file=str(train_file),
    )
    loader = TrainingDatasetLoader(cfg)
    loader.load_manifest()

    with pytest.raises(DatasetIntegrityError, match="Manifest checksum mismatch"):
        loader.verify_file_checksum(train_file)


def test_cross_split_leakage_detection(sample_valid_record):
    """Ensure identical records across train and validation splits trigger SplitIsolationError."""
    loader = TrainingDatasetLoader(DatasetConfig())
    train_recs = [sample_valid_record]
    val_recs = [sample_valid_record]  # Exact duplicate in val
    test_recs = []

    with pytest.raises(SplitIsolationError, match="Contamination detected"):
        loader.audit_split_isolation(train_recs, val_recs, test_recs)
