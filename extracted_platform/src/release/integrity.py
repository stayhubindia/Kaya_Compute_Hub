"""
Cryptographic Integrity & Checksum Management Engine (Phase 5.1).
Computes, records, and verifies SHA-256 checksums across all release artifacts,
detecting missing, corrupted, altered, or unexpected files.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from src.training.utils import compute_file_sha256

logger = logging.getLogger(__name__)


class IntegrityCheckResult(BaseModel):
    """Detailed audit report of release directory integrity."""
    is_valid: bool = False
    total_files_checked: int = 0
    verified_files: List[str] = Field(default_factory=list)
    missing_files: List[str] = Field(default_factory=list)
    mismatched_files: List[Dict[str, str]] = Field(default_factory=list)
    unexpected_files: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ReleaseIntegrityManager:
    """Manages generation, storage, and verification of cryptographic checksums for releases."""

    @staticmethod
    def hash_file(file_path: Union[str, Path]) -> str:
        """Compute SHA-256 hash for a given file."""
        return compute_file_sha256(Path(file_path))

    @staticmethod
    def generate_checksums_file(
        release_dir: Union[str, Path],
        checksums_filename: str = "checksums.sha256",
        exclude_checksums_file: bool = True,
    ) -> Path:
        """
        Scan all files in release_dir, compute SHA-256, and write sorted checksums.sha256.
        Format: `<sha256>  <relative_path>`
        """
        r_dir = Path(release_dir)
        if not r_dir.exists() or not r_dir.is_dir():
            raise FileNotFoundError(f"Release directory not found: {r_dir}")

        checksums_path = r_dir / checksums_filename
        lines: List[Tuple[str, str]] = []

        for p in sorted(r_dir.rglob("*")):
            if p.is_file():
                rel_path = p.relative_to(r_dir).as_posix()
                if exclude_checksums_file and rel_path == checksums_filename:
                    continue
                file_hash = compute_file_sha256(p)
                lines.append((file_hash, rel_path))

        # Sort alphabetically by relative path
        lines.sort(key=lambda x: x[1])

        with open(checksums_path, "w", encoding="utf-8") as f:
            for file_hash, rel_path in lines:
                f.write(f"{file_hash}  {rel_path}\n")

        return checksums_path

    @staticmethod
    def verify_release_integrity(
        release_dir: Union[str, Path],
        checksums_filename: str = "checksums.sha256",
    ) -> IntegrityCheckResult:
        """
        Verify every file listed in checksums.sha256 against on-disk state
        and detect any unexpected files not listed in the checksum inventory.
        """
        r_dir = Path(release_dir)
        result = IntegrityCheckResult()

        if not r_dir.exists():
            result.errors.append(f"Release directory does not exist: {r_dir}")
            return result

        chk_path = r_dir / checksums_filename
        if not chk_path.exists():
            result.errors.append(f"Checksums file not found: {chk_path}")
            return result

        expected_checksums: Dict[str, str] = {}
        with open(chk_path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    result.errors.append(f"Malformed checksum line {line_no}: '{line}'")
                    continue
                expected_hash, rel_path = parts[0], parts[1].strip()
                expected_checksums[rel_path] = expected_hash

        result.total_files_checked = len(expected_checksums)

        # 1. Check all expected files
        for rel_path, exp_hash in expected_checksums.items():
            f_path = r_dir / rel_path
            if not f_path.exists():
                result.missing_files.append(rel_path)
            else:
                act_hash = compute_file_sha256(f_path)
                if act_hash != exp_hash:
                    result.mismatched_files.append({
                        "file": rel_path,
                        "expected_sha256": exp_hash,
                        "actual_sha256": act_hash,
                    })
                else:
                    result.verified_files.append(rel_path)

        # 2. Check for unexpected files in release directory
        for p in r_dir.rglob("*"):
            if p.is_file():
                rel_p = p.relative_to(r_dir).as_posix()
                if rel_p != checksums_filename and rel_p not in expected_checksums:
                    result.unexpected_files.append(rel_p)

        # Determine overall validity
        if (
            len(result.missing_files) == 0
            and len(result.mismatched_files) == 0
            and len(result.unexpected_files) == 0
            and len(result.errors) == 0
        ):
            result.is_valid = True
        else:
            result.is_valid = False

        return result
