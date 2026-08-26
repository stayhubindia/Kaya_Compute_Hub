"""
Unit tests for ReproducibilityManager and metadata (Phase 5.1).
"""

from pathlib import Path
import pytest

from src.release.provenance import DatasetProvenance, HardwareProvenance, TrainingProvenance
from src.release.reproducibility import ReproducibilityManager


def test_build_reproducibility_record(tmp_path: Path):
    d_prov = DatasetProvenance(
        dataset_version="dataset-v1.0",
        manifest_sha256="dummy_sha",
        provenance_status="VERIFIED",
    )
    t_prov = TrainingProvenance(
        config_hash="test_train_hash",
        seed=42,
        optimizer="paged_adamw_8bit",
    )
    h_prov = HardwareProvenance()

    record = ReproducibilityManager.build_record(d_prov, t_prov, h_prov)
    assert record.random_seed == 42
    assert record.dataset_version == "dataset-v1.0"
    assert record.training_config_hash == "test_train_hash"
    assert "python" in record.environment_versions

    out_file = tmp_path / "reproducibility.json"
    record.save(out_file)
    assert out_file.exists()
