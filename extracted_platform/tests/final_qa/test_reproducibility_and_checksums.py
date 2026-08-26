"""
Test Suite: Reproducibility and Cryptographic Integrity Audit (Phase 3.5).
"""

from pathlib import Path
import pytest

from scripts.finalize_dataset import verify_checksums
from src.dataset.final_qa_auditor import FinalQAAuditor, GateStatus


@pytest.fixture
def dataset_root():
    return Path("data/instruction_dataset/v2.0").resolve()


def test_reproducibility_metadata(dataset_root):
    """Verifies deterministic seed, hashes, and platform metadata."""
    auditor = FinalQAAuditor(dataset_dir=dataset_root)
    report = auditor.run_full_audit()

    repro = report.reproducibility
    assert repro.seed == 42
    assert repro.generator_version == "2.0.0"
    assert repro.config_hash
    assert repro.python_version

    g14 = next(g for g in report.gate_matrix if g.gate_id == "G14")
    assert g14.status == GateStatus.PASS


def test_cryptographic_checksum_verification(dataset_root):
    """Verifies that all files in checksums.sha256 match their actual disk contents."""
    manifest_dir = dataset_root / "manifests"
    assert (manifest_dir / "checksums.sha256").is_file()

    success = verify_checksums(manifest_dir, dataset_root)
    assert success is True

    auditor = FinalQAAuditor(dataset_dir=dataset_root)
    report = auditor.run_full_audit()

    g15 = next(g for g in report.gate_matrix if g.gate_id == "G15")
    assert g15.status == GateStatus.PASS
