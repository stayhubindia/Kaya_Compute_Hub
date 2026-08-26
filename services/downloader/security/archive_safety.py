import os
import zipfile
import tarfile
from typing import List, Tuple
from django.conf import settings

class ArchiveSafetyError(ValueError):
    """Raised when an archive member violates safety rules."""
    pass

# Configurable Safety Boundaries
DEFAULT_MAX_EXTRACTED_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
DEFAULT_MAX_ARCHIVE_FILES = 10000
DEFAULT_MAX_NESTING_DEPTH = 5

def is_safe_path(target_dir: str, path: str) -> bool:
    """
    Ensures that target extraction path stays strictly inside target_dir.
    Prevents path traversal attacks (e.g. '../../etc/passwd').
    """
    resolved_target = os.path.realpath(target_dir)
    resolved_destination = os.path.realpath(os.path.join(target_dir, path))
    return resolved_destination.startswith(resolved_target + os.sep) or resolved_destination == resolved_target

def validate_zip_safety(zip_file: zipfile.ZipFile, target_dir: str) -> Tuple[int, int]:
    """
    Validates ZIP file members before extraction.
    Returns (total_files, total_uncompressed_bytes).
    Raises ArchiveSafetyError on safety violation.
    """
    max_bytes = getattr(settings, 'DOWNLOAD_MAX_EXTRACTED_SIZE_BYTES', DEFAULT_MAX_EXTRACTED_BYTES)
    max_files = getattr(settings, 'DOWNLOAD_MAX_ARCHIVE_FILES', DEFAULT_MAX_ARCHIVE_FILES)

    total_files = 0
    total_uncompressed_bytes = 0

    for member in zip_file.infolist():
        # Check path traversal and absolute paths
        filename = member.filename
        if filename.startswith('/') or filename.startswith('\\') or '..' in filename.split('/'):
            raise ArchiveSafetyError(f"Malicious Zip member path traversal detected: '{filename}'")

        if not is_safe_path(target_dir, filename):
            raise ArchiveSafetyError(f"Extracted path '{filename}' escapes target directory.")

        # Check total file count
        total_files += 1
        if total_files > max_files:
            raise ArchiveSafetyError(f"Archive file count limit exceeded ({total_files} > {max_files}).")

        # Check total size
        total_uncompressed_bytes += member.file_size
        if total_uncompressed_bytes > max_bytes:
            raise ArchiveSafetyError(f"Archive uncompressed size limit exceeded ({total_uncompressed_bytes} bytes > {max_bytes} bytes).")

    return total_files, total_uncompressed_bytes

def validate_tar_safety(tar_file: tarfile.TarFile, target_dir: str) -> Tuple[int, int]:
    """
    Validates TAR file members before extraction.
    Returns (total_files, total_uncompressed_bytes).
    Raises ArchiveSafetyError on safety violation.
    """
    max_bytes = getattr(settings, 'DOWNLOAD_MAX_EXTRACTED_SIZE_BYTES', DEFAULT_MAX_EXTRACTED_BYTES)
    max_files = getattr(settings, 'DOWNLOAD_MAX_ARCHIVE_FILES', DEFAULT_MAX_ARCHIVE_FILES)

    total_files = 0
    total_uncompressed_bytes = 0

    for member in tar_file.getmembers():
        # Reject Symlinks and Hardlinks
        if member.issym() or member.islnk():
            raise ArchiveSafetyError(f"Symlink or hardlink member detected in Tar archive: '{member.name}'")

        # Check path traversal
        name = member.name
        if name.startswith('/') or name.startswith('\\') or '..' in name.split('/'):
            raise ArchiveSafetyError(f"Malicious Tar member path traversal detected: '{name}'")

        if not is_safe_path(target_dir, name):
            raise ArchiveSafetyError(f"Extracted path '{name}' escapes target directory.")

        total_files += 1
        if total_files > max_files:
            raise ArchiveSafetyError(f"Archive file count limit exceeded ({total_files} > {max_files}).")

        total_uncompressed_bytes += member.size
        if total_uncompressed_bytes > max_bytes:
            raise ArchiveSafetyError(f"Archive uncompressed size limit exceeded ({total_uncompressed_bytes} bytes > {max_bytes} bytes).")

    return total_files, total_uncompressed_bytes

def safe_extract_archive(archive_path: str, extract_dir: str) -> Tuple[int, int]:
    """
    Safely extracts a ZIP, TAR, or GZIP archive into an isolated extract_dir.
    Validates all members prior to extracting any files.
    """
    os.makedirs(extract_dir, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            total_files, total_bytes = validate_zip_safety(zf, extract_dir)
            zf.extractall(extract_dir)
            return total_files, total_bytes

    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as tf:
            total_files, total_bytes = validate_tar_safety(tf, extract_dir)
            tf.extractall(extract_dir)
            return total_files, total_bytes

    else:
        raise ArchiveSafetyError(f"Unsupported or invalid archive format for file '{archive_path}'.")
