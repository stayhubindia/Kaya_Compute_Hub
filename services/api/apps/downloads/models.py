import uuid
from django.db import models
from django.conf import settings

class DownloadStatus(models.TextChoices):
    PENDING = 'pending', 'Pending'
    QUEUED = 'queued', 'Queued'
    RESOLVING = 'resolving', 'Resolving'
    DOWNLOADING = 'downloading', 'Downloading'
    VALIDATING = 'validating', 'Validating'
    EXTRACTING = 'extracting', 'Extracting'
    COMPLETED = 'completed', 'Completed'
    FAILED = 'failed', 'Failed'
    CANCELLED = 'cancelled', 'Cancelled'
    PAUSED = 'paused', 'Paused'

class Download(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='downloads',
        db_index=True
    )
    source_url = models.TextField()
    provider = models.CharField(max_length=50, default='generic_http', db_index=True)
    original_filename = models.CharField(max_length=255, blank=True)
    storage_uri = models.CharField(max_length=500, blank=True, null=True)
    temporary_path = models.CharField(max_length=500, blank=True, null=True)
    content_type = models.CharField(max_length=100, blank=True, null=True)
    
    expected_size_bytes = models.BigIntegerField(blank=True, null=True)
    downloaded_size_bytes = models.BigIntegerField(default=0)
    checksum_algorithm = models.CharField(max_length=20, default='sha256')
    expected_checksum = models.CharField(max_length=255, blank=True, null=True)
    actual_checksum = models.CharField(max_length=255, blank=True, null=True)
    
    extract = models.BooleanField(default=False)
    status = models.CharField(
        max_length=30,
        choices=DownloadStatus.choices,
        default=DownloadStatus.PENDING,
        db_index=True
    )
    progress_percent = models.FloatField(default=0.0)
    current_speed_bytes = models.FloatField(default=0.0)
    retry_count = models.IntegerField(default=0)
    error_code = models.CharField(max_length=100, blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)

    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'downloads_download'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['created_by', 'created_at']),
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['provider', 'created_at']),
        ]

    def __str__(self):
        return f"Download {self.id} ({self.status}) - {self.source_url[:30]}"
