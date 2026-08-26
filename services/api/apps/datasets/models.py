import uuid
from django.db import models
from django.conf import settings

class DatasetStatusChoices(models.TextChoices):
    PENDING = 'pending', 'Pending'
    AVAILABLE = 'available', 'Available'
    FAILED = 'failed', 'Failed'
    DELETED = 'deleted', 'Deleted'

class Dataset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    source_url = models.URLField(max_length=1024, blank=True)
    storage_uri = models.CharField(max_length=1024)
    format = models.CharField(max_length=50, blank=True)
    size_bytes = models.BigIntegerField(default=0)
    checksum = models.CharField(max_length=128, blank=True)
    status = models.CharField(
        max_length=50,
        choices=DatasetStatusChoices.choices,
        default=DatasetStatusChoices.PENDING,
        db_index=True
    )
    parent_dataset = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='derived_datasets'
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='datasets'
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'datasets_dataset'
        ordering = ['-created_at']

    def __str__(self):
        return f"Dataset {self.name} ({self.format}) - [{self.status}]"
