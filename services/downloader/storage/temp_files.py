import os
import shutil
from django.conf import settings
from services.downloader.security import generate_safe_internal_filename

def get_temp_root() -> str:
    temp_root = getattr(settings, 'DOWNLOAD_TEMP_ROOT', None)
    if not temp_root:
        temp_root = os.path.join(settings.BASE_DIR, 'storage', 'temp', 'downloads')
    os.makedirs(temp_root, exist_ok=True)
    return temp_root

def get_temp_download_path(download_id: str, original_filename: str = "") -> str:
    temp_dir = get_temp_root()
    safe_name = generate_safe_internal_filename(download_id, original_filename)
    return os.path.join(temp_dir, f"{safe_name}.part")

def cleanup_temp_file(filepath: str):
    if filepath and os.path.exists(filepath):
        try:
            if os.path.isdir(filepath):
                shutil.rmtree(filepath, ignore_errors=True)
            else:
                os.remove(filepath)
        except OSError:
            pass
