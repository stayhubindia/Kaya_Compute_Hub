import os
import shutil
from typing import Tuple
from django.conf import settings
from services.downloader.security import generate_safe_internal_filename

def get_storage_root() -> str:
    storage_root = getattr(settings, 'DOWNLOAD_STORAGE_ROOT', None)
    if not storage_root:
        storage_root = os.path.join(settings.BASE_DIR, 'storage', 'datasets')
    os.makedirs(storage_root, exist_ok=True)
    return storage_root

def get_quarantine_root() -> str:
    quarantine_root = os.path.join(settings.BASE_DIR, 'storage', 'quarantine')
    os.makedirs(quarantine_root, exist_ok=True)
    return quarantine_root

def store_verified_dataset(temp_filepath: str, download_id: str, original_filename: str = "") -> Tuple[str, int]:
    """
    Atomically moves verified download from temporary location to storage/datasets/<download_id>/<safe_filename>.
    Returns (storage_uri, file_size_bytes).
    """
    if not os.path.exists(temp_filepath):
        raise FileNotFoundError(f"Temporary file '{temp_filepath}' not found.")

    storage_root = get_storage_root()
    dataset_dir = os.path.join(storage_root, str(download_id))
    os.makedirs(dataset_dir, exist_ok=True)

    safe_filename = generate_safe_internal_filename(str(download_id), original_filename)
    final_destination = os.path.join(dataset_dir, safe_filename)

    # Atomic move
    shutil.move(temp_filepath, final_destination)

    file_size = os.path.getsize(final_destination)
    storage_uri = f"storage://datasets/{download_id}/{safe_filename}"

    return storage_uri, file_size

def quarantine_failed_download(temp_filepath: str, download_id: str) -> str:
    """
    Moves failed/invalid download to quarantine location storage/quarantine/<download_id>.failed.
    """
    if not temp_filepath or not os.path.exists(temp_filepath):
        return ""

    quarantine_root = get_quarantine_root()
    quarantine_destination = os.path.join(quarantine_root, f"{download_id}.failed")

    try:
        shutil.move(temp_filepath, quarantine_destination)
        return quarantine_destination
    except Exception:
        return ""
