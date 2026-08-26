"""
Test Suite: Schema Validation Audit (Phase 3.5).
Verifies that 100% of dataset records in all splits conform to the canonical DatasetRecord schema.
"""

import json
from pathlib import Path
import pytest

from src.dataset.schema import DatasetRecord, Role


@pytest.fixture
def dataset_root():
    return Path("data/instruction_dataset/v2.0").resolve()


def test_splits_schema_conformance(dataset_root):
    """Validates every single record across train, val, test, and accepted splits."""
    for split_name in ["train", "validation", "test"]:
        file_path = dataset_root / "splits" / f"{split_name}.jsonl"
        assert file_path.is_file()

        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                raw_dict = json.loads(line)
                rec = DatasetRecord.model_validate(raw_dict)

                assert len(rec.messages) >= 2, f"Split {split_name} record {idx} has < 2 messages"
                assert rec.messages[0].role in (Role.USER, Role.SYSTEM)
                assert rec.messages[-1].role == Role.ASSISTANT

                # Check contents
                for m in rec.messages:
                    assert isinstance(m.content, str) and m.content.strip(), f"Empty message in {split_name}:{idx}"

                # Metadata validation
                assert rec.metadata.domain
                assert rec.metadata.task_type
                assert rec.metadata.difficulty
                assert rec.metadata.source_type
                assert rec.metadata.provenance is not None
                assert rec.metadata.provenance.source_id is not None
