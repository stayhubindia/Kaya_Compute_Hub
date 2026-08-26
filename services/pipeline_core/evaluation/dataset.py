"""
Evaluation Dataset Loader & Split Manager (Phase 4.4).
Enforces FROZEN dataset verification, SHA-256 checksum integrity,
split isolation validation, and extracts evaluation prompts and target references.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field

from src.dataset.production import DatasetFreezeState, ProductionManifest
from src.dataset.schema import DatasetRecord, Message, Role
from src.evaluation.config import EvaluationDatasetConfig
from src.training.utils import compute_file_sha256


class EvaluationDatasetError(ValueError):
    """Raised when evaluation dataset validation fails."""
    pass


class EvaluationExample(BaseModel):
    """Structured evaluation instance with extracted prompt and target completion."""
    record_id: str
    domain: str
    topic: str
    task_type: str
    difficulty: str
    quality_score: float = 1.0
    messages: List[Message]
    prompt_messages: List[Message]
    reference_completion: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dataset_record(cls, record: DatasetRecord) -> EvaluationExample:
        """Extract prompt messages and ground truth assistant completion from DatasetRecord."""
        messages = record.messages
        if not messages:
            raise EvaluationDatasetError("Record has no messages.")

        # Identify final assistant turn as reference completion
        last_msg = messages[-1]
        role_val = last_msg.role.value if hasattr(last_msg.role, "value") else str(last_msg.role)
        if role_val != "assistant":
            raise EvaluationDatasetError(
                f"Record last message must be ASSISTANT, found '{role_val}'."
            )

        prompt_messages = messages[:-1]
        if not prompt_messages:
            raise EvaluationDatasetError(
                "Record has no prompt messages before assistant completion."
            )

        diff_str = record.metadata.difficulty.value if hasattr(record.metadata.difficulty, 'value') else str(record.metadata.difficulty)
        task_str = record.metadata.task_type.value if hasattr(record.metadata.task_type, 'value') else str(record.metadata.task_type)
        rec_id = record.metadata.source_id or record.canonical_content_hash()[:16]

        return cls(
            record_id=rec_id,
            domain=record.metadata.domain,
            topic=record.metadata.topic,
            task_type=task_str,
            difficulty=diff_str,
            quality_score=record.metadata.quality_score or 1.0,
            messages=messages,
            prompt_messages=prompt_messages,
            reference_completion=last_msg.content,
            metadata=record.metadata.model_dump(),
        )


class EvaluationDatasetLoader:
    """
    Dedicated loader for evaluation datasets.
    Guarantees that test records are purely sourced from frozen test splits
    and maintains strict isolation from training and validation records.
    """

    def __init__(self, config: EvaluationDatasetConfig):
        self.config = config
        self._manifest: Optional[ProductionManifest] = None

    def load_manifest(self) -> ProductionManifest:
        """Load and validate the production manifest."""
        manifest_path = Path(self.config.manifest_path)
        if not manifest_path.exists():
            alt = Path("datasets/production/manifests/production_manifest.json")
            if alt.exists():
                manifest_path = alt
            else:
                raise EvaluationDatasetError(f"Manifest file not found: {manifest_path}")

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            manifest = ProductionManifest(**data)
        except Exception as e:
            raise EvaluationDatasetError(f"Failed to parse manifest: {e}") from e

        status_str = manifest.status.value if hasattr(manifest.status, "value") else str(manifest.status)
        if self.config.require_frozen and status_str != DatasetFreezeState.FROZEN.value:
            raise EvaluationDatasetError(
                f"Dataset status is '{status_str}', but FROZEN is required."
            )
        self._manifest = manifest
        return manifest

    def verify_split_checksums(self) -> Dict[str, bool]:
        """Verify that split file SHA-256 hashes match the manifest or exist on disk."""
        manifest = self._manifest or self.load_manifest()
        results: Dict[str, bool] = {}

        split_files = {
            "train": Path(self.config.train_file),
            "validation": Path(self.config.validation_file),
            "test": Path(self.config.test_file),
        }

        for split_name, split_path in split_files.items():
            if not split_path.exists():
                raise EvaluationDatasetError(f"Split file missing: {split_path}")

            actual_sha = compute_file_sha256(split_path)
            expected_sha = manifest.checksums.get(split_path.name) or manifest.checksums.get(split_name)

            if expected_sha and actual_sha.lower() != expected_sha.lower():
                raise EvaluationDatasetError(
                    f"SHA-256 mismatch for split '{split_name}': expected {expected_sha}, got {actual_sha}"
                )
            results[split_name] = True

        return results

    def verify_split_isolation(self) -> int:
        """
        Verify that test records have zero hash collisions with train and validation records.
        Returns total number of verified isolated test records.
        """
        def _get_hashes(path: Path) -> Set[str]:
            hashes = set()
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        rec = DatasetRecord(**data)
                        hashes.add(rec.canonical_content_hash())
            return hashes

        train_path = Path(self.config.train_file)
        val_path = Path(self.config.validation_file)
        test_path = Path(self.config.test_file)

        train_hashes = _get_hashes(train_path) if train_path.exists() else set()
        val_hashes = _get_hashes(val_path) if val_path.exists() else set()
        test_hashes = _get_hashes(test_path) if test_path.exists() else set()

        train_test_overlap = train_hashes.intersection(test_hashes)
        if train_test_overlap:
            raise EvaluationDatasetError(
                f"Contamination detected! {len(train_test_overlap)} records overlap between train and test splits."
            )

        val_test_overlap = val_hashes.intersection(test_hashes)
        if val_test_overlap:
            raise EvaluationDatasetError(
                f"Contamination detected! {len(val_test_overlap)} records overlap between validation and test splits."
            )

        return len(test_hashes)

    def load_examples(
        self,
        domain_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
        task_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[EvaluationExample]:
        """
        Load validated evaluation examples from the target split file.
        """
        if self.config.require_frozen:
            self.load_manifest()
        if self.config.validate_sha256:
            self.verify_split_checksums()
        self.verify_split_isolation()

        split_file_map = {
            "test": Path(self.config.test_file),
            "validation": Path(self.config.validation_file),
            "train": Path(self.config.train_file),
        }
        target_path = split_file_map.get(self.config.split, Path(self.config.test_file))

        if not target_path.exists():
            raise EvaluationDatasetError(f"Target evaluation file not found: {target_path}")

        examples: List[EvaluationExample] = []
        max_count = limit or self.config.max_examples

        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                record = DatasetRecord(**data)
                example = EvaluationExample.from_dataset_record(record)

                # Apply filters
                if domain_filter and example.domain != domain_filter:
                    continue
                if difficulty_filter and example.difficulty != difficulty_filter:
                    continue
                if task_filter and example.task_type != task_filter:
                    continue

                examples.append(example)
                if max_count is not None and len(examples) >= max_count:
                    break

        return examples
