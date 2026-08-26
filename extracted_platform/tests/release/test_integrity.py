"""
Unit tests for ReleaseIntegrityManager and checksum verification (Phase 5.1).
"""

from pathlib import Path
import pytest

from src.release.integrity import ReleaseIntegrityManager


def test_checksums_generation_and_verification(tmp_path: Path):
    rel_dir = tmp_path / "test_release"
    rel_dir.mkdir()

    (rel_dir / "file1.txt").write_text("hello world")
    (rel_dir / "file2.json").write_text('{"key": "value"}')

    sub_dir = rel_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "file3.bin").write_bytes(b"\x00\x01\x02")

    # Generate checksums
    chk_file = ReleaseIntegrityManager.generate_checksums_file(rel_dir)
    assert chk_file.exists()

    # Verify integrity
    res = ReleaseIntegrityManager.verify_release_integrity(rel_dir)
    assert res.is_valid is True
    assert len(res.verified_files) == 3
    assert len(res.missing_files) == 0
    assert len(res.mismatched_files) == 0
    assert len(res.unexpected_files) == 0


def test_integrity_detects_tampered_file(tmp_path: Path):
    rel_dir = tmp_path / "tampered_release"
    rel_dir.mkdir()
    (rel_dir / "weights.bin").write_bytes(b"original")

    ReleaseIntegrityManager.generate_checksums_file(rel_dir)

    # Tamper with file
    (rel_dir / "weights.bin").write_bytes(b"tampered_data")

    res = ReleaseIntegrityManager.verify_release_integrity(rel_dir)
    assert res.is_valid is False
    assert len(res.mismatched_files) == 1
    assert res.mismatched_files[0]["file"] == "weights.bin"


def test_integrity_detects_unexpected_file(tmp_path: Path):
    rel_dir = tmp_path / "unexpected_release"
    rel_dir.mkdir()
    (rel_dir / "valid.txt").write_text("valid")

    ReleaseIntegrityManager.generate_checksums_file(rel_dir)

    # Add unexpected file
    (rel_dir / "stray.tmp").write_text("untracked")

    res = ReleaseIntegrityManager.verify_release_integrity(rel_dir)
    assert res.is_valid is False
    assert "stray.tmp" in res.unexpected_files
