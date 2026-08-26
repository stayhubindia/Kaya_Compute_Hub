"""
Unit tests for Hugging Face Distribution, Secret Scanner, and Preflight Auditor (Phase 5.5).
"""

import json
from pathlib import Path
import pytest

from src.distribution.secret_scanner import ReleaseSecretScanner
from src.distribution.package_auditor import DistributionPackageAuditor
from src.distribution.hf_distributor import HFDistributor


def test_secret_scanner_clean():
    """Verify that the official release directory contains zero secrets."""
    scanner = ReleaseSecretScanner()
    res = scanner.scan_directory("releases/qwen3-4b-qlora-v1.0")
    assert res.clean is True
    assert len(res.findings) == 0


def test_secret_scanner_detects_token(tmp_path: Path):
    """Verify that secret scanner identifies credential patterns without exposing values."""
    scanner = ReleaseSecretScanner()
    test_file = tmp_path / "leaked_credentials.txt"
    test_file.write_text("HF_TOKEN = 'hf_0123456789abcdef0123456789abcdef012'\n")

    findings = scanner.scan_file(test_file)
    assert len(findings) == 1
    assert findings[0].secret_category == "HuggingFace Token"
    assert findings[0].line_number == 1
    # Ensure raw secret string is not embedded in the finding representation
    assert "hf_0123456789" not in findings[0].model_dump_json()


def test_distribution_package_auditor():
    """Verify that DistributionPackageAuditor validates all 15 release artifacts."""
    auditor = DistributionPackageAuditor(release_dir="releases/qwen3-4b-qlora-v1.0")
    res = auditor.audit_package()

    assert res.passed is True
    assert res.checksum_verified is True
    assert res.secrets_clean is True
    assert res.doc_valid is True
    assert res.total_files == 15
    assert len(res.errors) == 0


def test_hf_distributor_dry_run():
    """Verify that dry-run manifest is constructed without remote mutations."""
    distributor = HFDistributor(release_dir="releases/qwen3-4b-qlora-v1.0")
    dry_res = distributor.generate_dry_run_manifest()

    assert dry_res["preflight_passed"] is True
    assert dry_res["status"] == "HUGGING FACE UPLOAD READY — EXPLICIT AUTHORIZATION REQUIRED"
    assert dry_res["total_files_to_upload"] == 15
    assert "durgeshkidsyt/qwen3-4b-qlora-v1.0" in dry_res["target_repository"] or "unauthenticated" in dry_res["target_repository"]


def test_hf_distributor_upload_requires_confirmation():
    """Verify that upload fails cleanly when explicit confirmation is not provided."""
    distributor = HFDistributor(release_dir="releases/qwen3-4b-qlora-v1.0")
    success, res = distributor.upload_release(confirm_upload=False)

    assert success is False
    assert res["status"] == "UPLOAD_ABORTED_NO_CONFIRMATION"
