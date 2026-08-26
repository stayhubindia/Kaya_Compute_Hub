import os
import time
from django.utils import timezone
from celery import shared_task

from apps.downloads.models import Download, DownloadStatus
from apps.audit.services import log_audit_event
from services.downloader.providers import get_provider_for_url
from services.downloader.security import validate_url_security, SSRFError, safe_extract_archive, ArchiveSafetyError
from services.downloader.storage import (
    get_temp_download_path,
    cleanup_temp_file,
    verify_file_checksum,
    store_verified_dataset,
    quarantine_failed_download
)

@shared_task(bind=True, max_retries=3, default_retry_delay=5)
def process_download_job(self, download_id: str):
    """
    Celery task to execute background download, checksum validation, safe archive extraction,
    and persistent storage placement.
    """
    try:
        download = Download.objects.get(id=download_id)
    except Download.DoesNotExist:
        return {"status": "error", "message": f"Download {download_id} not found."}

    if download.status == DownloadStatus.CANCELLED:
        return {"status": "cancelled", "message": "Download was cancelled before execution."}

    # 1. Resolving Provider & Pre-flight SSRF Validation
    download.status = DownloadStatus.RESOLVING
    download.started_at = timezone.now() if not download.started_at else download.started_at
    download.save(update_fields=['status', 'started_at', 'updated_at'])

    try:
        validate_url_security(download.source_url)
        provider = get_provider_for_url(download.source_url)
    except SSRFError as e:
        download.status = DownloadStatus.FAILED
        download.error_code = "SSRF_BLOCKED"
        download.error_message = str(e)
        download.save(update_fields=['status', 'error_code', 'error_message', 'updated_at'])
        log_audit_event(action="download.url_rejected", resource_type="download", resource_id=str(download.id), metadata={"reason": str(e)})
        return {"status": "failed", "reason": str(e)}

    # 2. Prepare Temp Download Path
    temp_path = get_temp_download_path(str(download.id), download.original_filename)
    download.temporary_path = temp_path
    download.status = DownloadStatus.DOWNLOADING
    download.save(update_fields=['status', 'temporary_path', 'updated_at'])

    log_audit_event(action="download.started", resource_type="download", resource_id=str(download.id))

    last_db_update = time.time()

    def progress_cb(downloaded_bytes: int, total_bytes: int, speed: float):
        nonlocal last_db_update
        now = time.time()
        # Throttle DB updates to once per 500ms
        if now - last_db_update > 0.5 or (total_bytes and downloaded_bytes >= total_bytes):
            last_db_update = now
            pct = round((downloaded_bytes / total_bytes * 100), 2) if total_bytes > 0 else 0.0
            Download.objects.filter(id=download.id).update(
                downloaded_size_bytes=downloaded_bytes,
                expected_size_bytes=total_bytes or download.expected_size_bytes,
                progress_percent=pct,
                current_speed_bytes=round(speed, 2),
                updated_at=timezone.now()
            )

    # 3. Download Stream
    try:
        provider.download(
            url=download.source_url,
            destination_path=temp_path,
            progress_callback=progress_cb
        )
    except Exception as e:
        # Check if cancelled while downloading
        download.refresh_from_db()
        if download.status == DownloadStatus.CANCELLED:
            cleanup_temp_file(temp_path)
            return {"status": "cancelled"}

        if self.request.retries < self.max_retries:
            download.retry_count += 1
            download.save(update_fields=['retry_count', 'updated_at'])
            raise self.retry(exc=e)

        download.status = DownloadStatus.FAILED
        download.error_code = "PROVIDER_ERROR"
        download.error_message = str(e)
        download.save(update_fields=['status', 'error_code', 'error_message', 'updated_at'])
        log_audit_event(action="download.provider_error", resource_type="download", resource_id=str(download.id), metadata={"error": str(e)})
        cleanup_temp_file(temp_path)
        return {"status": "failed", "reason": str(e)}

    # 4. Checksum Verification
    download.status = DownloadStatus.VALIDATING
    download.save(update_fields=['status', 'updated_at'])

    is_valid, actual_checksum = verify_file_checksum(
        filepath=temp_path,
        expected_checksum=download.expected_checksum,
        algorithm=download.checksum_algorithm
    )

    download.actual_checksum = actual_checksum
    if not is_valid:
        download.status = DownloadStatus.FAILED
        download.error_code = "CHECKSUM_MISMATCH"
        download.error_message = f"Expected checksum '{download.expected_checksum}' does not match actual checksum '{actual_checksum}'."
        download.save(update_fields=['status', 'actual_checksum', 'error_code', 'error_message', 'updated_at'])
        log_audit_event(action="download.checksum_mismatch", resource_type="download", resource_id=str(download.id))
        quarantine_failed_download(temp_path, str(download.id))
        return {"status": "failed", "reason": "Checksum mismatch"}

    # 5. Archive Extraction (if requested)
    if download.extract:
        download.status = DownloadStatus.EXTRACTING
        download.save(update_fields=['status', 'updated_at'])

        extract_dir = os.path.join(os.path.dirname(temp_path), f"extracted_{download.id}")
        try:
            safe_extract_archive(temp_path, extract_dir)
        except ArchiveSafetyError as e:
            download.status = DownloadStatus.FAILED
            download.error_code = "ARCHIVE_EXTRACTION_REJECTED"
            download.error_message = str(e)
            download.save(update_fields=['status', 'error_code', 'error_message', 'updated_at'])
            log_audit_event(action="download.archive_extraction_rejected", resource_type="download", resource_id=str(download.id), metadata={"reason": str(e)})
            quarantine_failed_download(temp_path, str(download.id))
            return {"status": "failed", "reason": str(e)}

    # 6. Store Verified Dataset
    storage_uri, file_size = store_verified_dataset(temp_path, str(download.id), download.original_filename)

    download.status = DownloadStatus.COMPLETED
    download.storage_uri = storage_uri
    download.downloaded_size_bytes = file_size
    download.progress_percent = 100.0
    download.completed_at = timezone.now()
    download.save(update_fields=['status', 'storage_uri', 'downloaded_size_bytes', 'progress_percent', 'completed_at', 'updated_at'])

    log_audit_event(action="download.completed", resource_type="download", resource_id=str(download.id), metadata={"storage_uri": storage_uri, "bytes": file_size})

    return {"status": "completed", "download_id": str(download.id), "storage_uri": storage_uri}
