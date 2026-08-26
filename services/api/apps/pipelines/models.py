import uuid
from django.db import models
from django.conf import settings
from apps.datasets.models import Dataset

class ProcessingRunStatus(models.TextChoices):
    DRAFT = 'draft', 'Draft'
    QUEUED = 'queued', 'Queued'
    VALIDATING = 'validating', 'Validating'
    RUNNING = 'running', 'Running'
    CHECKPOINTING = 'checkpointing', 'Checkpointing'
    SUCCEEDED = 'succeeded', 'Succeeded'
    FAILED = 'failed', 'Failed'
    CANCELLED = 'cancelled', 'Cancelled'
    PAUSED = 'paused', 'Paused'
    RETRYING = 'retrying', 'Retrying'

class PipelineDefinition(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='pipelines'
    )
    version = models.CharField(max_length=50, default='1.0.0')
    enabled = models.BooleanField(default=True)
    stages = models.JSONField(default=list, help_text="List of configured pipeline stages")
    resource_policy = models.JSONField(default=dict, help_text="Resource limits for container execution")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'pipelines_pipelinedefinition'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} (v{self.version})"

class ProcessingRun(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pipeline = models.ForeignKey(
        PipelineDefinition,
        on_delete=models.CASCADE,
        related_name='runs'
    )
    source_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='source_processing_runs'
    )
    output_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='output_processing_runs'
    )
    status = models.CharField(
        max_length=50,
        choices=ProcessingRunStatus.choices,
        default=ProcessingRunStatus.DRAFT,
        db_index=True
    )
    current_stage = models.CharField(max_length=100, blank=True)
    progress_percent = models.FloatField(default=0.0)
    input_manifest_uri = models.CharField(max_length=1024, blank=True)
    output_manifest_uri = models.CharField(max_length=1024, blank=True)
    error_code = models.CharField(max_length=100, blank=True)
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='processing_runs'
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'pipelines_processingrun'
        ordering = ['-created_at']

    def __str__(self):
        return f"ProcessingRun {self.id} [{self.status}]"

class ProcessingStageEvent(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    processing_run = models.ForeignKey(
        ProcessingRun,
        on_delete=models.CASCADE,
        related_name='stage_events'
    )
    stage_name = models.CharField(max_length=100)
    status = models.CharField(max_length=50)
    input_uri = models.CharField(max_length=1024, blank=True)
    output_uri = models.CharField(max_length=1024, blank=True)
    metrics = models.JSONField(default=dict)
    log_uri = models.CharField(max_length=1024, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'pipelines_processingstageevent'
        ordering = ['created_at']

    def __str__(self):
        return f"StageEvent {self.stage_name} - {self.status}"

class DatasetManifest(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.OneToOneField(
        Dataset,
        on_delete=models.CASCADE,
        related_name='manifest'
    )
    schema_version = models.CharField(max_length=50, default='1.0')
    file_count = models.IntegerField(default=0)
    total_bytes = models.BigIntegerField(default=0)
    checksum = models.CharField(max_length=128, blank=True)
    format = models.CharField(max_length=50, blank=True)
    columns = models.JSONField(default=list)
    statistics = models.JSONField(default=dict)
    provenance = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'pipelines_datasetmanifest'

    def __str__(self):
        return f"DatasetManifest for {self.dataset.name}"
