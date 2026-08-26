"""
Unit tests for Evaluation Dataset Loader & Split Isolation (Phase 4.4).
"""

import json
import pytest
from pathlib import Path
from src.dataset.schema import DatasetRecord, Message, RecordMetadata, Role
from src.evaluation.config import EvaluationDatasetConfig
from src.evaluation.dataset import EvaluationDatasetError, EvaluationDatasetLoader, EvaluationExample


def test_frozen_dataset_loading():
    cfg = EvaluationDatasetConfig()
    loader = EvaluationDatasetLoader(cfg)
    manifest = loader.load_manifest()
    assert manifest.dataset_version == "dataset-v1.0"
    status_str = manifest.status.value if hasattr(manifest.status, "value") else str(manifest.status)
    assert status_str == "FROZEN"


def test_split_isolation_and_examples():
    cfg = EvaluationDatasetConfig()
    loader = EvaluationDatasetLoader(cfg)
    isolated_count = loader.verify_split_isolation()
    assert isolated_count == 7

    examples = loader.load_examples()
    assert len(examples) == 7
    for ex in examples:
        assert isinstance(ex, EvaluationExample)
        assert len(ex.prompt_messages) >= 1
        assert len(ex.reference_completion) > 0
        assert ex.domain != ""
        assert ex.difficulty in ("beginner", "intermediate", "advanced", "expert")


def test_domain_and_difficulty_filtering():
    cfg = EvaluationDatasetConfig()
    loader = EvaluationDatasetLoader(cfg)
    all_examples = loader.load_examples()

    sample_dom = all_examples[0].domain
    filtered_dom = loader.load_examples(domain_filter=sample_dom)
    assert all(e.domain == sample_dom for e in filtered_dom)

    sample_diff = all_examples[0].difficulty
    filtered_diff = loader.load_examples(difficulty_filter=sample_diff)
    assert all(e.difficulty == sample_diff for e in filtered_diff)


def test_contamination_detection(tmp_path: Path):
    # Create contaminated train and test files
    rec_dict = {
        "messages": [
            {"role": "user", "content": "Hello test prompt"},
            {"role": "assistant", "content": "Hello test response"},
        ],
        "metadata": {
            "domain": "programming",
            "topic": "testing",
            "task_type": "code_generation",
            "difficulty": "beginner",
            "quality_score": 1.0,
            "source": "unit_test",
            "source_type": "synthetic",
        },
    }

    train_file = tmp_path / "train.jsonl"
    test_file = tmp_path / "test.jsonl"
    val_file = tmp_path / "val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec_dict) + "\n")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec_dict) + "\n")  # Contaminated duplicate
    with open(val_file, "w", encoding="utf-8") as f:
        pass

    cfg = EvaluationDatasetConfig(
        train_file=str(train_file),
        test_file=str(test_file),
        validation_file=str(val_file),
        require_frozen=False,
        validate_sha256=False,
    )
    loader = EvaluationDatasetLoader(cfg)

    with pytest.raises(EvaluationDatasetError, match="Contamination detected"):
        loader.verify_split_isolation()
