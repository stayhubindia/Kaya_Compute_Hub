"""Safe Archive Extraction Module

Protects against Path Traversal (Zip Slip / Tar Slip), Symlink Escape Attacks,
Decompression Bombs, Absolute Path Injection, and Unsafe Archive Members.
"""

import os
import tarfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Union


class ArchiveSecurityError(Exception):
    """Raised when an archive member violates security boundaries."""
    pass


MAX_DECOMPRESSION_RATIO = 100.0  # Max ratio of uncompressed to compressed size
MAX_SINGLE_FILE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
MAX_TOTAL_FILES = 20000


def safe_extract_zip(
    zip_path: Union[str, Path],
    dest_dir: Union[str, Path],
    max_total_files: int = MAX_TOTAL_FILES,
    max_file_size: int = MAX_SINGLE_FILE_BYTES,
) -> List[Path]:
    """Safely extract a ZIP archive while validating security constraints."""
    zip_path = Path(zip_path).resolve()
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: List[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()

        if len(members) > max_total_files:
            raise ArchiveSecurityError(
                f"Archive exceeds maximum allowed file count ({len(members)} > {max_total_files})"
            )

        for member in members:
            # Check for path traversal / absolute paths
            target_path = (dest_dir / member.filename).resolve()

            try:
                target_path.relative_to(dest_dir)
            except ValueError:
                raise ArchiveSecurityError(
                    f"Path traversal detected in zip member: {member.filename}"
                )

            if member.file_size > max_file_size:
                raise ArchiveSecurityError(
                    f"Zip member {member.filename} exceeds maximum size limit ({member.file_size} bytes)"
                )

            # Check decompression ratio
            if member.compress_size > 0:
                ratio = member.file_size / float(member.compress_size)
                if ratio > MAX_DECOMPRESSION_RATIO:
                    raise ArchiveSecurityError(
                        f"Decompression bomb detected in zip member {member.filename} (ratio: {ratio:.1f})"
                    )

            # Extract safely
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, open(target_path, "wb") as target:
                    total_bytes = 0
                    while chunk := source.read(65536):
                        total_bytes += len(chunk)
                        if total_bytes > max_file_size:
                            raise ArchiveSecurityError(
                                f"Extracted file {member.filename} exceeded maximum size limit during stream."
                            )
                        target.write(chunk)
                extracted_files.append(target_path)

    return extracted_files


def safe_extract_tar(
    tar_path: Union[str, Path],
    dest_dir: Union[str, Path],
    max_total_files: int = MAX_TOTAL_FILES,
    max_file_size: int = MAX_SINGLE_FILE_BYTES,
) -> List[Path]:
    """Safely extract a TAR / TAR.GZ archive while validating security constraints."""
    tar_path = Path(tar_path).resolve()
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: List[Path] = []

    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()

        if len(members) > max_total_files:
            raise ArchiveSecurityError(
                f"Tar archive exceeds maximum allowed file count ({len(members)} > {max_total_files})"
            )

        for member in members:
            # Reject symlinks and hardlinks
            if member.issym() or member.islnk():
                raise ArchiveSecurityError(
                    f"Symlink/Hardlink member rejected for security: {member.name}"
                )

            target_path = (dest_dir / member.name).resolve()

            try:
                target_path.relative_to(dest_dir)
            except ValueError:
                raise ArchiveSecurityError(
                    f"Path traversal detected in tar member: {member.name}"
                )

            if member.size > max_file_size:
                raise ArchiveSecurityError(
                    f"Tar member {member.name} exceeds maximum size limit ({member.size} bytes)"
                )

            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                source = tf.extractfile(member)
                if source is not None:
                    with open(target_path, "wb") as target:
                        total_bytes = 0
                        while chunk := source.read(65536):
                            total_bytes += len(chunk)
                            if total_bytes > max_file_size:
                                raise ArchiveSecurityError(
                                    f"Extracted file {member.name} exceeded max size limit."
                                )
                            target.write(chunk)
                    extracted_files.append(target_path)

    return extracted_files
