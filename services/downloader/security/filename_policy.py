import os
import re
import uuid

SAFE_EXTENSIONS_ALLOWLIST = {
    '.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.7z',
    '.csv', '.json', '.txt', '.pdf', '.parquet', '.h5', '.hdf5',
    '.png', '.jpg', '.jpeg', '.bin', '.dat', '.pt', '.pth', '.safetensors'
}

def sanitize_filename(original_name: str) -> str:
    """
    Sanitizes user or server-provided filenames to prevent path traversal or filesystem corruption.
    """
    if not original_name:
        return "download.bin"

    # Remove any directory path components
    basename = os.path.basename(original_name.replace('\\', '/'))
    
    # Strip null bytes and non-printable characters
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', basename)
    
    # Remove risky characters, keeping alphanumeric, dots, dashes, underscores
    cleaned = re.sub(r'[^a-zA-Z0-9._-]', '_', cleaned)

    # Remove leading dots or dashes
    cleaned = cleaned.lstrip('.-')

    if not cleaned or cleaned == '.' or cleaned == '..':
        return "download.bin"

    return cleaned[:255]

def generate_safe_internal_filename(download_id: str, original_filename: str = "") -> str:
    """
    Generates a guaranteed safe internal filename combining a UUID identifier and a safe extension.
    Example: '550e8400-e29b-41d4-a716-446655440000.zip'
    """
    sanitized = sanitize_filename(original_filename)
    _, ext = os.path.splitext(sanitized)
    ext_lower = ext.lower()
    
    if ext_lower not in SAFE_EXTENSIONS_ALLOWLIST and not re.match(r'^\.[a-zA-Z0-9]{1,10}$', ext_lower):
        ext_lower = ".bin"

    return f"{download_id}{ext_lower}"
