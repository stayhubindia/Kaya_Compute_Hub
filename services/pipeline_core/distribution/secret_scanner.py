"""
Release Secret Scanner Subsystem (Phase 5.5).
Scans directories and release artifacts for credentials, tokens, and private keys.
Strictly redacts and never exposes matched secret values in logs, reports, or CLI outputs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Sensitive patterns (name, regex)
SECRET_PATTERNS = [
    ("HuggingFace Token", re.compile(r"hf_[a-zA-Z0-9]{34,}")),
    ("Generic HuggingFace Key", re.compile(r"HFAK[a-zA-Z0-9]{25,}")),
    ("Bearer Authorization Token", re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}")),
    ("Private Key Header", re.compile(r"-----BEGIN\s+([A-Z0-9_-]+\s+)?PRIVATE\s+KEY-----")),
    ("Generic Secret Key Assignment", re.compile(r"(?i)(?:api_key|apikey|secret_key|access_token|auth_token)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]")),
    ("OpenAI API Key", re.compile(r"sk-[a-zA-Z0-9]{32,}")),
    ("AWS Access Key ID", re.compile(r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}")),
    ("Generic Password Assignment", re.compile(r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{8,}['\"]")),
]


class SecretFinding(BaseModel):
    """Safe representation of an identified secret detection."""
    file_path: str
    secret_category: str
    line_number: int
    severity: str = "HIGH"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "secret_category": self.secret_category,
            "line_number": self.line_number,
            "severity": self.severity,
        }


class SecretScanResult(BaseModel):
    """Aggregate outcome of secret audit across target release files."""
    scanned_files_count: int = 0
    clean: bool = True
    findings: List[SecretFinding] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class ReleaseSecretScanner:
    """Audits files and directories to ensure zero credential leakage prior to distribution."""

    def __init__(self, ignored_patterns: Optional[Set[str]] = None):
        self.ignored_patterns = ignored_patterns or {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "venv",
        }

    def scan_file(self, file_path: Union[str, Path]) -> List[SecretFinding]:
        """Scan an individual file for sensitive credential patterns."""
        p = Path(file_path)
        if not p.is_file():
            return []

        # Skip large binary files like safetensors from regex parsing
        if p.suffix in (".safetensors", ".bin", ".pt", ".pth", ".onnx", ".parquet"):
            return []

        findings: List[SecretFinding] = []
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, start=1):
                    for cat_name, pattern in SECRET_PATTERNS:
                        if pattern.search(line):
                            findings.append(
                                SecretFinding(
                                    file_path=str(p),
                                    secret_category=cat_name,
                                    line_number=line_idx,
                                )
                            )
        except Exception as e:
            logger.warning(f"Could not read file for secret scan {p}: {e}")

        return findings

    def scan_directory(self, target_dir: Union[str, Path]) -> SecretScanResult:
        """Recursively scan a directory for secrets."""
        target_path = Path(target_dir)
        result = SecretScanResult()

        if not target_path.exists() or not target_path.is_dir():
            result.errors.append(f"Target directory does not exist: {target_path}")
            result.clean = False
            return result

        for item in sorted(target_path.rglob("*")):
            # Check ignored directories
            if any(part in self.ignored_patterns for part in item.parts):
                continue

            if item.is_file():
                result.scanned_files_count += 1
                item_findings = self.scan_file(item)
                if item_findings:
                    result.findings.extend(item_findings)

        result.clean = len(result.findings) == 0
        return result
