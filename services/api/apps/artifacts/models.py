import uuid
from django.db import models
from django.conf import settings
from apps.jobs.models import Job

class ArtifactTypeChoices(models.TextChoices):
    MODEL = 'model', 'Model'
    CHECKPOINT = 'checkpoint', 'Checkpoint'
    REPORT = 'report', 'Report'
    NOTEBOOK = 'notebook', 'Notebook'
    ARCHIVE = 'archive', 'Archive'
    LOG = 'log', 'Log'

class Artifact(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    artifact_type = models.CharField(
        max_length=50,
        choices=ArtifactTypeChoices.choices,
        db_index=True
    )
    storage_uri = models.CharField(max_length=1024)
    size_bytes = models.BigIntegerField(default=0)
    checksum = models.CharField(max_length=128, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    job = models.ForeignKey(
        Job,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='artifacts'
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='artifacts'
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'artifacts_artifact'
        ordering = ['-created_at']

    def __str__(self):
        return f"Artifact {self.name} ({self.artifact_type})"
