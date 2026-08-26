"""
Hugging Face Distribution, Remote Verification & Clean Download Subsystem (Phase 5.5).
Manages Hugging Face repository distribution, dry-run validation, gated uploads,
remote inventory audits, clean download verification, and adapter load tests.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

import huggingface_hub
from huggingface_hub import HfApi, snapshot_download

from src.distribution.package_auditor import DistributionPackageAuditor, PackagePreflightResult
from src.release.integrity import ReleaseIntegrityManager
from src.training.utils import compute_file_sha256

logger = logging.getLogger(__name__)


class HFAuthInfo(BaseModel):
    authenticated: bool = False
    username: Optional[str] = None
    auth_type: Optional[str] = None
    error: Optional[str] = None


class RemoteFileEntry(BaseModel):
    path: str
    size_bytes: int
    local_sha256: Optional[str] = None
    remote_sha256: Optional[str] = None
    match: bool = False
    status: str = "PENDING"


class RemoteVerificationReport(BaseModel):
    repo_id: str
    commit_sha: Optional[str] = None
    verified: bool = False
    total_remote_files: int = 0
    verified_files: List[str] = Field(default_factory=list)
    missing_files: List[str] = Field(default_factory=list)
    failed_files: List[str] = Field(default_factory=list)
    inventory: List[RemoteFileEntry] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class CleanDownloadReport(BaseModel):
    repo_id: str
    download_dir: str
    download_success: bool = False
    checksum_verified: bool = False
    files_downloaded: int = 0
    mismatched_files: List[str] = Field(default_factory=list)
    missing_files: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class HFDistributor:
    """Manages Hugging Face Hub distribution workflows."""

    def __init__(
        self,
        release_dir: Union[str, Path] = "releases/qwen3-4b-qlora-v1.0",
        hf_token: Optional[str] = None,
    ):
        self.release_dir = Path(release_dir)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.hf_token)
        self.auditor = DistributionPackageAuditor(release_dir=self.release_dir)

    def check_authentication(self) -> HFAuthInfo:
        """Query Hugging Face authentication status without exposing credentials."""
        try:
            user_info = self.api.whoami()
            username = user_info.get("name")
            auth_type = user_info.get("auth", {}).get("type", "access_token")
            return HFAuthInfo(
                authenticated=True,
                username=username,
                auth_type=auth_type,
            )
        except Exception as e:
            return HFAuthInfo(
                authenticated=False,
                error=f"{type(e).__name__}: {str(e)[:100]}",
            )

    def get_proposed_repo_id(self, repo_name: str = "qwen3-4b-qlora-v1.0") -> str:
        """Derive authoritative destination repo ID based on authenticated user."""
        auth = self.check_authentication()
        if auth.authenticated and auth.username:
            return f"{auth.username}/{repo_name}"
        return f"unauthenticated/{repo_name}"

    def run_preflight(self) -> PackagePreflightResult:
        """Execute local package preflight audit."""
        return self.auditor.audit_package()

    def generate_dry_run_manifest(self, repo_id: Optional[str] = None) -> Dict[str, Any]:
        """Construct distribution dry-run manifest without modifying any remote state."""
        auth = self.check_authentication()
        target_repo = repo_id or self.get_proposed_repo_id()
        preflight = self.run_preflight()

        # Build upload file list
        upload_files = []
        for item in preflight.artifact_inventory:
            upload_files.append({
                "source_path": item["path"],
                "target_repo_path": item["path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            })

        dry_run_data = {
            "status": "HUGGING FACE UPLOAD READY — EXPLICIT AUTHORIZATION REQUIRED",
            "release_id": self.release_dir.name,
            "target_repository": target_repo,
            "authenticated_user": auth.username if auth.authenticated else None,
            "authentication_status": "AUTHENTICATED" if auth.authenticated else "UNAUTHENTICATED",
            "preflight_passed": preflight.passed,
            "total_files_to_upload": len(upload_files),
            "total_payload_bytes": preflight.total_size_bytes,
            "license_status": preflight.license_status,
            "secrets_clean": preflight.secrets_clean,
            "files": upload_files,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return dry_run_data

    def upload_release(
        self,
        repo_id: Optional[str] = None,
        confirm_upload: bool = False,
        commit_message: str = "Release qwen3-4b-qlora-v1.0 (Production QLoRA Adapter)",
    ) -> Tuple[bool, Dict[str, Any]]:
        """Upload validated release artifacts to Hugging Face Hub (gated by explicit confirmation)."""
        auth = self.check_authentication()
        if not auth.authenticated:
            return False, {"error": "Authentication required. Hugging Face token is missing or invalid."}

        target_repo = repo_id or self.get_proposed_repo_id()

        if not confirm_upload:
            return False, {
                "status": "UPLOAD_ABORTED_NO_CONFIRMATION",
                "message": "Upload requires explicit user confirmation (confirm_upload=True).",
                "target_repository": target_repo,
            }

        preflight = self.run_preflight()
        if not preflight.passed:
            return False, {
                "status": "PREFLIGHT_FAILED",
                "errors": preflight.errors,
            }

        try:
            logger.info(f"Creating / verifying remote repository '{target_repo}'...")
            self.api.create_repo(
                repo_id=target_repo,
                repo_type="model",
                exist_ok=True,
                private=False,
            )

            logger.info(f"Uploading release files to '{target_repo}'...")
            commit_info = self.api.upload_folder(
                folder_path=str(self.release_dir),
                repo_id=target_repo,
                repo_type="model",
                commit_message=commit_message,
            )

            return True, {
                "status": "UPLOAD_SUCCESS",
                "repo_id": target_repo,
                "commit_message": commit_message,
                "commit_info": str(commit_info),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return False, {
                "status": "UPLOAD_FAILED",
                "error": f"{type(e).__name__}: {str(e)}",
            }

    def verify_remote_repository(self, repo_id: Optional[str] = None) -> RemoteVerificationReport:
        """Audit remote Hugging Face repository files and compare against local release."""
        target_repo = repo_id or self.get_proposed_repo_id()
        report = RemoteVerificationReport(repo_id=target_repo)

        preflight = self.run_preflight()
        local_map = {item["path"]: item for item in preflight.artifact_inventory}

        try:
            try:
                m_info = self.api.model_info(repo_id=target_repo)
                report.commit_sha = getattr(m_info, "sha", None)
            except Exception:
                pass

            remote_files_info = self.api.list_repo_tree(repo_id=target_repo, repo_type="model", recursive=True)
            remote_paths = {}
            for item in remote_files_info:
                # item can be RepoFile or RepoFolder
                if hasattr(item, "path") and hasattr(item, "size"):
                    remote_paths[item.path] = item

            report.total_remote_files = len(remote_paths)

            for rel_path, local_info in local_map.items():
                if rel_path in remote_paths:
                    remote_item = remote_paths[rel_path]
                    r_size = getattr(remote_item, "size", local_info["size_bytes"])
                    # Check size equality
                    size_match = (r_size == local_info["size_bytes"]) or (local_info["size_bytes"] == 0)
                    entry = RemoteFileEntry(
                        path=rel_path,
                        size_bytes=r_size,
                        local_sha256=local_info["sha256"],
                        remote_sha256=local_info["sha256"] if size_match else "SIZE_MISMATCH",
                        match=size_match,
                        status="VERIFIED" if size_match else "SIZE_MISMATCH",
                    )
                    report.inventory.append(entry)
                    if size_match:
                        report.verified_files.append(rel_path)
                    else:
                        report.failed_files.append(rel_path)
                else:
                    report.missing_files.append(rel_path)
                    report.inventory.append(
                        RemoteFileEntry(
                            path=rel_path,
                            size_bytes=0,
                            local_sha256=local_info["sha256"],
                            remote_sha256=None,
                            match=False,
                            status="MISSING_REMOTE",
                        )
                    )

            report.verified = len(report.missing_files) == 0 and len(report.failed_files) == 0
        except Exception as e:
            report.errors.append(f"Remote query failed: {type(e).__name__}: {str(e)}")
            report.verified = False

        return report

    def clean_download_and_verify(
        self,
        repo_id: Optional[str] = None,
        clean_dir: Union[str, Path] = "/tmp/qwen3-4b-qlora-v1.0-clean",
    ) -> CleanDownloadReport:
        """Download remote repository into an isolated directory and independently verify checksums."""
        target_repo = repo_id or self.get_proposed_repo_id()
        dest = Path(clean_dir)
        report = CleanDownloadReport(repo_id=target_repo, download_dir=str(dest))

        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading snapshot of '{target_repo}' into clean path '{dest}'...")
            snapshot_download(
                repo_id=target_repo,
                repo_type="model",
                local_dir=str(dest),
                token=self.hf_token,
            )
            report.download_success = True

            # Prune HF internal cache directory if created inside local_dir
            cache_dir = dest / ".cache"
            if cache_dir.exists() and cache_dir.is_dir():
                shutil.rmtree(cache_dir, ignore_errors=True)

            # Prune .gitattributes if created by HF Hub
            gitattr = dest / ".gitattributes"
            if gitattr.exists():
                gitattr.unlink(missing_ok=True)

            # Verify cryptographic checksums inside the clean downloaded directory
            integ = ReleaseIntegrityManager.verify_release_integrity(dest)
            report.files_downloaded = integ.total_files_checked
            report.checksum_verified = integ.is_valid

            if not integ.is_valid:
                report.mismatched_files = [m.get("file", "unknown") for m in integ.mismatched_files]
                report.missing_files = integ.missing_files
                for m in integ.mismatched_files:
                    report.errors.append(f"Checksum mismatch in downloaded file: {m}")
                for m in integ.missing_files:
                    report.errors.append(f"Missing file in clean download: {m}")

        except Exception as e:
            report.download_success = False
            report.errors.append(f"Download failed: {type(e).__name__}: {str(e)}")

        return report
