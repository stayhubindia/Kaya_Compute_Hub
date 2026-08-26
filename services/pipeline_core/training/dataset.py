"""
Training Dataset Loader & Split Manager (Phase 4.1).
Provides strict validation of frozen dataset status, SHA-256 verification,
zero cross-split leakage enforcement, and PyTorch Dataset integration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from torch.utils.data import Dataset

from src.dataset.production import DatasetFreezeState, ProductionManifest
from src.dataset.schema import DatasetRecord, RecordValidationError
from src.training.config import DatasetConfig, TrainingConfig
from src.training.utils import compute_file_sha256


class DatasetIntegrityError(ValueError):
    """Raised when dataset fails manifest, checksum, or freeze state validation."""
    pass


class SplitIsolationError(ValueError):
    """Raised when cross-split leakage or contamination is detected."""
    pass


class QwenTrainingDataset(Dataset):
    """PyTorch Dataset wrapper for validated canonical DatasetRecord objects."""

    def __init__(self, records: List[DatasetRecord], split_name: str = "train"):
        self.records = records
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> DatasetRecord:
        return self.records[idx]

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.records)


class TrainingDatasetLoader:
    """
    Dedicated loader for QLoRA fine-tuning datasets.
    Enforces strict immutability, FROZEN state validation, SHA-256 integrity,
    and zero cross-split leakage.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self._manifest: Optional[ProductionManifest] = None

    def load_manifest(self) -> ProductionManifest:
        """Load and validate the production manifest."""
        manifest_path = Path(self.config.manifest_path)
        if not manifest_path.exists():
            # Also check alternative path in parent/processed directories
            alt_path = Path("datasets/production/manifests/production_manifest.json")
            if alt_path.exists():
                manifest_path = alt_path
            else:
                raise DatasetIntegrityError(f"Production manifest not found at: {manifest_path}")

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            manifest = ProductionManifest(**data)
        except Exception as e:
            raise DatasetIntegrityError(f"Failed to parse production manifest: {e}") from e

        if self.config.version and manifest.dataset_version != self.config.version:
            raise DatasetIntegrityError(
                f"Manifest dataset_version '{manifest.dataset_version}' does not match requested '{self.config.version}'"
            )

        status_str = manifest.status.value if hasattr(manifest.status, "value") else str(manifest.status)
        if self.config.require_frozen and status_str != DatasetFreezeState.FROZEN.value:
            raise DatasetIntegrityError(
                f"Training rejected: Dataset '{manifest.dataset_version}' is in '{status_str}' state. "
                "Only FROZEN datasets are accepted for training."
            )

        self._manifest = manifest
        return manifest

    def verify_file_checksum(self, file_path: Union[str, Path], expected_sha256: Optional[str] = None) -> str:
        """Verify the SHA-256 checksum of a file against expected value or manifest."""
        p = Path(file_path)
        if not p.exists():
            raise DatasetIntegrityError(f"Dataset file missing at: {p}")

        actual_sha = compute_file_sha256(p)

        if expected_sha256:
            if actual_sha.lower() != expected_sha256.lower():
                raise DatasetIntegrityError(
                    f"Checksum mismatch for '{p.name}': expected {expected_sha256}, got {actual_sha}"
                )
        elif self._manifest and p.name in self._manifest.checksums:
            expected = self._manifest.checksums[p.name]
            if actual_sha.lower() != expected.lower():
                raise DatasetIntegrityError(
                    f"Manifest checksum mismatch for '{p.name}': expected {expected}, got {actual_sha}"
                )

        return actual_sha

    def load_records_from_file(self, file_path: Union[str, Path]) -> List[DatasetRecord]:
        """Load and strictly validate DatasetRecord instances from a JSONL file."""
        p = Path(file_path)
        if not p.exists():
            raise DatasetIntegrityError(f"Cannot load records: file does not exist at {p}")

        records: List[DatasetRecord] = []
        with open(p, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    raw_dict = json.loads(clean_line)
                    record = DatasetRecord(**raw_dict)
                    if not record.metadata.provenance or not record.metadata.provenance.source or not record.metadata.provenance.source_type:
                        raise DatasetIntegrityError(
                            f"Record at line {line_idx} in '{p.name}' missing complete provenance metadata."
                        )
                    records.append(record)
                except Exception as e:
                    raise DatasetIntegrityError(
                        f"Schema validation failed at line {line_idx} in '{p.name}': {e}"
                    ) from e

        return records

    def audit_split_isolation(
        self,
        train_records: List[DatasetRecord],
        val_records: List[DatasetRecord],
        test_records: List[DatasetRecord],
    ) -> None:
        """Verify strict zero cross-split leakage across train, validation, and test sets."""
        def _get_hashes(recs: List[DatasetRecord]) -> Set[str]:
            hashes = set()
            for r in recs:
                # Use concatenated message content hash
                content = "".join(f"{m.role}:{m.content}" for m in r.messages)
                h = hashlib.sha256(content.encode("utf-8")).hexdigest()
                hashes.add(h)
            return hashes

        train_hashes = _get_hashes(train_records)
        val_hashes = _get_hashes(val_records)
        test_hashes = _get_hashes(test_records)

        train_val_overlap = train_hashes.intersection(val_hashes)
        if train_val_overlap:
            raise SplitIsolationError(
                f"Contamination detected: {len(train_val_overlap)} record overlaps between TRAIN and VALIDATION splits."
            )

        train_test_overlap = train_hashes.intersection(test_hashes)
        if train_test_overlap:
            raise SplitIsolationError(
                f"Contamination detected: {len(train_test_overlap)} record overlaps between TRAIN and TEST splits."
            )

        val_test_overlap = val_hashes.intersection(test_hashes)
        if val_test_overlap:
            raise SplitIsolationError(
                f"Contamination detected: {len(val_test_overlap)} record overlaps between VALIDATION and TEST splits."
            )

    def load_splits(self) -> Tuple[QwenTrainingDataset, QwenTrainingDataset, QwenTrainingDataset]:
        """
        Load all three dataset splits with full manifest validation, SHA-256 verification,
        and zero cross-split leakage checks.
        """
        # 1. Validate manifest
        manifest = self.load_manifest()

        # 2. Check and load files
        train_path = Path(self.config.train_file)
        val_path = Path(self.config.validation_file)
        test_path = Path(self.config.test_file)

        # Fallback resolution if running from relative subdirectories
        if not train_path.exists():
            train_path = Path("datasets/production/processed/train.jsonl")
            val_path = Path("datasets/production/processed/validation.jsonl")
            test_path = Path("datasets/production/processed/test.jsonl")

        if self.config.validate_sha256:
            self.verify_file_checksum(train_path)
            self.verify_file_checksum(val_path)
            self.verify_file_checksum(test_path)

        # 3. Load and validate records
        train_records = self.load_records_from_file(train_path)
        val_records = self.load_records_from_file(val_path)
        test_records = self.load_records_from_file(test_path)

        if not train_records:
            raise DatasetIntegrityError("Train split is empty.")

        # 4. Audit split isolation
        self.audit_split_isolation(train_records, val_records, test_records)

        return (
            QwenTrainingDataset(train_records, split_name="train"),
            QwenTrainingDataset(val_records, split_name="validation"),
            QwenTrainingDataset(test_records, split_name="test"),
        )
