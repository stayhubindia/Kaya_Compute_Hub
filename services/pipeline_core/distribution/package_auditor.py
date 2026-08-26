"""
Distribution Package Auditor Subsystem (Phase 5.5).
Validates local release packages, audits documentation and license disclosure,
and ensures complete cleanliness and integrity before distribution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field

from src.distribution.secret_scanner import ReleaseSecretScanner, SecretScanResult
from src.release.integrity import ReleaseIntegrityManager
from src.training.utils import compute_file_sha256

logger = logging.getLogger(__name__)

MANDATORY_RELEASE_FILES = [
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
    "adapter/tokenizer.json",
    "adapter/tokenizer_config.json",
    "adapter/chat_template.jinja",
    "adapter/checkpoint_metadata.json",
    "adapter/trainer_state.json",
    "adapter/README.md",
    "checksums.sha256",
    "compatibility.json",
    "manifest.json",
    "MODEL_CARD.md",
    "provenance.json",
    "README.md",
    "reproducibility.json",
]

PROHIBITED_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".tmp",
    ".swp",
    ".DS_Store",
    ".env",
    "credentials",
]


class PackagePreflightResult(BaseModel):
    """Overall outcome of package preflight audit."""
    release_id: str
    release_dir: str
    passed: bool = False
    total_files: int = 0
    total_size_bytes: int = 0
    checksum_verified: bool = False
    secrets_clean: bool = False
    doc_valid: bool = False
    license_status: str = "LICENSE INFORMATION REQUIRES VERIFICATION"
    artifact_inventory: List[Dict[str, Any]] = Field(default_factory=list)
    secret_findings: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DistributionPackageAuditor:
    """Audits local release candidate packages for Hugging Face distribution readiness."""

    def __init__(self, release_dir: Union[str, Path] = "releases/qwen3-4b-qlora-v1.0"):
        self.release_dir = Path(release_dir)
        self.secret_scanner = ReleaseSecretScanner()

    def audit_package(self) -> PackagePreflightResult:
        result = PackagePreflightResult(
            release_id=self.release_dir.name,
            release_dir=str(self.release_dir),
        )

        if not self.release_dir.exists() or not self.release_dir.is_dir():
            result.errors.append(f"Release directory not found: {self.release_dir}")
            return result

        # 1. Check Mandatory Files
        missing_files = []
        for mf in MANDATORY_RELEASE_FILES:
            file_path = self.release_dir / mf
            if not file_path.exists():
                missing_files.append(mf)

        if missing_files:
            result.errors.extend([f"Missing mandatory release file: {m}" for m in missing_files])

        # 2. Check Prohibited Files
        unauthorized_files = []
        total_size = 0
        inventory: List[Dict[str, Any]] = []

        for p in sorted(self.release_dir.rglob("*")):
            if p.is_file():
                rel_posix = p.relative_to(self.release_dir).as_posix()
                size = p.stat().st_size
                total_size += size
                sha = compute_file_sha256(p)
                inventory.append({
                    "path": rel_posix,
                    "size_bytes": size,
                    "sha256": sha,
                })

                for prob in PROHIBITED_PATTERNS:
                    if prob in rel_posix:
                        unauthorized_files.append(rel_posix)

        result.total_files = len(inventory)
        result.total_size_bytes = total_size
        result.artifact_inventory = inventory

        if unauthorized_files:
            result.errors.extend([f"Prohibited file detected: {u}" for u in unauthorized_files])

        # 3. Cryptographic Checksums Audit
        integrity_check = ReleaseIntegrityManager.verify_release_integrity(self.release_dir)
        if integrity_check.is_valid:
            result.checksum_verified = True
        else:
            result.checksum_verified = False
            for m in integrity_check.missing_files:
                result.errors.append(f"Checksum catalog missing file: {m}")
            for m in integrity_check.mismatched_files:
                result.errors.append(f"Checksum mismatch for: {m.get('file')}")
            for u in integrity_check.unexpected_files:
                result.errors.append(f"Untracked file in checksum catalog: {u}")

        # 4. Secret Scan
        scan_res: SecretScanResult = self.secret_scanner.scan_directory(self.release_dir)
        result.secrets_clean = scan_res.clean
        if not scan_res.clean:
            result.secret_findings = [f.to_dict() for f in scan_res.findings]
            result.errors.extend(
                [f"Secret detected in {f.file_path} (Category: {f.secret_category}, Line: {f.line_number})"
                 for f in scan_res.findings]
            )

        # 5. Documentation & License Audit
        readme_path = self.release_dir / "README.md"
        card_path = self.release_dir / "MODEL_CARD.md"
        doc_errors = []

        if readme_path.exists():
            readme_text = readme_path.read_text(encoding="utf-8")
            if "Qwen/Qwen3-4B-Base" not in readme_text:
                doc_errors.append("README.md missing Base Model reference (Qwen/Qwen3-4B-Base)")
            if "39" not in readme_text:
                doc_errors.append("README.md missing 39 training records disclosure")
            if "500" not in readme_text:
                doc_errors.append("README.md missing 500 benchmark cases count")
        else:
            doc_errors.append("README.md missing")

        if card_path.exists():
            card_text = card_path.read_text(encoding="utf-8")
            if "39" not in card_text:
                doc_errors.append("MODEL_CARD.md missing 39 training records disclosure")
            if "benchmark-v1.0" not in card_text:
                doc_errors.append("MODEL_CARD.md missing benchmark-v1.0 reference")
        else:
            doc_errors.append("MODEL_CARD.md missing")

        if doc_errors:
            result.doc_valid = False
            result.errors.extend(doc_errors)
        else:
            result.doc_valid = True

        result.license_status = "Qwen Research License / Apache 2.0 (Requires verification against upstream terms)"

        # Final preflight determination
        result.passed = len(result.errors) == 0 and result.checksum_verified and result.secrets_clean and result.doc_valid
        return result
