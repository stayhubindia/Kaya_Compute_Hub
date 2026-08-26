from django.conf import settings
from django.utils import timezone
from datetime import timedelta
from django.db.models import Sum
from apps.downloads.models import Download, DownloadStatus

class QuotaExceededError(ValueError):
    """Raised when user or global download quota is exceeded."""
    pass

DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
DEFAULT_DAILY_QUOTA_BYTES = 50 * 1024 * 1024 * 1024    # 50 GB
DEFAULT_MAX_CONCURRENT = 5

def check_user_download_quota(user, expected_size_bytes: int = 0):
    """
    Checks concurrent download count, daily bytes, and file size limits for a user.
    Raises QuotaExceededError if limits are breached.
    """
    max_file_size = getattr(settings, 'DOWNLOAD_MAX_FILE_SIZE_BYTES', DEFAULT_MAX_FILE_SIZE_BYTES)
    daily_quota = getattr(settings, 'DOWNLOAD_DAILY_QUOTA_BYTES', DEFAULT_DAILY_QUOTA_BYTES)
    max_concurrent = getattr(settings, 'DOWNLOAD_MAX_CONCURRENT_PER_USER', DEFAULT_MAX_CONCURRENT)

    # 1. Max File Size Check
    if expected_size_bytes and expected_size_bytes > max_file_size:
        raise QuotaExceededError(
            f"Requested file size ({expected_size_bytes} bytes) exceeds maximum allowed size ({max_file_size} bytes)."
        )

    # 2. Max Concurrent Downloads Check
    active_statuses = [
        DownloadStatus.PENDING,
        DownloadStatus.QUEUED,
        DownloadStatus.RESOLVING,
        DownloadStatus.DOWNLOADING,
        DownloadStatus.VALIDATING,
        DownloadStatus.EXTRACTING,
    ]
    active_count = Download.objects.filter(created_by=user, status__in=active_statuses).count()
    if active_count >= max_concurrent:
        raise QuotaExceededError(
            f"Maximum concurrent download limit reached ({active_count}/{max_concurrent}). Please wait for active downloads to complete."
        )

    # 3. Daily Byte Limit Check
    since_time = timezone.now() - timedelta(days=1)
    daily_bytes = Download.objects.filter(
        created_by=user,
        created_at__gte=since_time,
        status__in=[DownloadStatus.COMPLETED, DownloadStatus.DOWNLOADING]
    ).aggregate(total=Sum('downloaded_size_bytes'))['total'] or 0

    if daily_bytes + expected_size_bytes > daily_quota:
        raise QuotaExceededError(
            f"Daily download quota exceeded ({daily_bytes} bytes used + {expected_size_bytes} requested > {daily_quota} limit)."
        )

    return True
