"""Storage Provider & Folder Contract Manager for Kaya Compute Hub.

Implements deterministic directory layouts, atomic write operations (tempfile -> fsync -> rename),
resumable streaming checksum validation (SHA-256), and local/Drive storage abstractions.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union


class StorageProvider(Protocol):
    """Protocol interface for storage backends (Local VM, Google Drive, Object Store)."""

    def put(self, local_path: Union[str, Path], remote_key: str, *, checksum: Optional[str] = None) -> str:
        ...

    def get(self, remote_key: str, local_path: Union[str, Path], *, checksum: Optional[str] = None) -> Path:
        ...

    def list(self, prefix: str) -> List[str]:
        ...

    def verify(self, remote_key: str, checksum: str) -> bool:
        ...


class FolderContractManager:
    """Manages the standard dataset collection directory tree on VM disk or Drive."""

    CONTRACT_DIRECTORIES = [
        "00-source",
        "10-raw/pdf",
        "10-raw/html",
        "10-raw/json",
        "10-raw/archives",
        "10-raw/images",
        "20-extracted/pages",
        "20-extracted/ocr",
        "30-normalized",
        "40-chunks",
        "50-generated",
        "60-qa",
        "70-training-ready",
        "80-training/runs",
        "90-evaluation/reports",
        "90-evaluation/predictions",
        "manifests",
        "reports",
        "logs",
    ]

    def __init__(self, storage_root: Union[str, Path]):
        self.storage_root = Path(storage_root).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def initialize_collection(self, collection_slug: str) -> Dict[str, Path]:
        """Creates standard collection folder structure and returns directory paths."""
        col_dir = self.storage_root / collection_slug
        dirs: Dict[str, Path] = {"root": col_dir}

        for sub_path in self.CONTRACT_DIRECTORIES:
            d = col_dir / sub_path
            d.mkdir(parents=True, exist_ok=True)
            key = sub_path.replace("/", "_").replace("-", "_")
            dirs[key] = d

        return dirs

    def get_collection_path(self, collection_slug: str, sub_path: str = "") -> Path:
        """Resolves path under collection slug ensuring path stays inside root."""
        col_dir = (self.storage_root / collection_slug).resolve()
        target = (col_dir / sub_path).resolve()
        try:
            target.relative_to(self.storage_root)
        except ValueError:
            raise PermissionError(f"Path traversal blocked: {sub_path} escapes root {self.storage_root}")
        return target


def compute_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Computes streaming SHA-256 digest of a file without holding full file in RAM."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_write_bytes(dest_path: Union[str, Path], content: bytes) -> str:
    """Atomically writes bytes to dest_path using temporary file + fsync + rename."""
    path = Path(dest_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_file = tempfile.mkstemp(dir=path.parent, prefix=".tmp_write_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, path)
    except Exception:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise

    return compute_sha256(path)


def atomic_write_text(dest_path: Union[str, Path], text: str, encoding: str = "utf-8") -> str:
    """Atomically writes text content to dest_path."""
    return atomic_write_bytes(dest_path, text.encode(encoding))


def atomic_write_json(dest_path: Union[str, Path], data: Any, indent: int = 2) -> str:
    """Atomically writes serialized JSON object to dest_path."""
    text = json.dumps(data, indent=indent, ensure_ascii=False)
    return atomic_write_text(dest_path, text)


class LocalVMStorageProvider:
    """VM local disk implementation of StorageProvider protocol."""

    def __init__(self, root_dir: Union[str, Path]):
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, local_path: Union[str, Path], remote_key: str, *, checksum: Optional[str] = None) -> str:
        src = Path(local_path).resolve()
        dest = (self.root / remote_key).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        if checksum:
            src_hash = compute_sha256(src)
            if src_hash != checksum:
                raise ValueError(f"Checksum mismatch for {src}: expected {checksum}, got {src_hash}")

        shutil.copy2(src, dest)
        return compute_sha256(dest)

    def get(self, remote_key: str, local_path: Union[str, Path], *, checksum: Optional[str] = None) -> Path:
        src = (self.root / remote_key).resolve()
        dest = Path(local_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not src.is_file():
            raise FileNotFoundError(f"Remote key {remote_key} not found in {self.root}")

        if checksum:
            src_hash = compute_sha256(src)
            if src_hash != checksum:
                raise ValueError(f"Checksum mismatch for remote key {remote_key}: expected {checksum}, got {src_hash}")

        shutil.copy2(src, dest)
        return dest

    def list(self, prefix: str) -> List[str]:
        target_dir = (self.root / prefix).resolve()
        if not target_dir.exists():
            return []
        keys = []
        for p in target_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(self.root)
                keys.append(str(rel))
        return sorted(keys)

    def verify(self, remote_key: str, checksum: str) -> bool:
        target = (self.root / remote_key).resolve()
        if not target.is_file():
            return False
        return compute_sha256(target) == checksum
