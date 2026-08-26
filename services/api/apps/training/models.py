import uuid
from django.db import models
from django.conf import settings
from apps.datasets.models import Dataset
from apps.pipelines.models import ProcessingRun

class TrainingRunStatus(models.TextChoices):
    DRAFT = 'draft', 'Draft'
    QUEUED = 'queued', 'Queued'
    SCHEDULED = 'scheduled', 'Scheduled'
    PREPARING = 'preparing', 'Preparing'
    RUNNING = 'running', 'Running'
    CHECKPOINTING = 'checkpointing', 'Checkpointing'
    PAUSED = 'paused', 'Paused'
    CANCELLING = 'cancelling', 'Cancelling'
    SUCCEEDED = 'succeeded', 'Succeeded'
    FAILED = 'failed', 'Failed'
    CANCELLED = 'cancelled', 'Cancelled'
    RETRYING = 'retrying', 'Retrying'

class TrainingRun(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='training_runs'
    )
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='training_runs'
    )
    processing_run = models.ForeignKey(
        ProcessingRun,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='training_runs'
    )
    backend = models.CharField(max_length=50, default='demo')
    container_image = models.CharField(max_length=255, default='kaya/ml-trainer:pytorch-2.2')
    container_digest = models.CharField(max_length=255, blank=True)
    configuration = models.JSONField(default=dict, help_text="Hyperparameters and model settings")
    resource_policy = models.JSONField(default=dict, help_text="CPU/GPU resource boundaries")
    status = models.CharField(
        max_length=50,
        choices=TrainingRunStatus.choices,
        default=TrainingRunStatus.DRAFT,
        db_index=True
    )
    current_epoch = models.IntegerField(default=0)
    current_step = models.IntegerField(default=0)
    progress_percent = models.FloatField(default=0.0)
    best_metric = models.FloatField(null=True, blank=True)
    best_metric_name = models.CharField(max_length=100, blank=True)
    checkpoint_uri = models.CharField(max_length=1024, blank=True)
    output_model_uri = models.CharField(max_length=1024, blank=True)
    error_code = models.CharField(max_length=100, blank=True)
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'training_trainingrun'
        ordering = ['-created_at']

    def __str__(self):
        return f"TrainingRun {self.name} [{self.status}]"

class TrainingMetric(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    training_run = models.ForeignKey(
        TrainingRun,
        on_delete=models.CASCADE,
        related_name='metrics'
    )
    step = models.IntegerField()
    epoch = models.IntegerField()
    name = models.CharField(max_length=100)
    value = models.FloatField()
    split = models.CharField(max_length=50, default='train')
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'training_trainingmetric'
        ordering = ['step']

    def __str__(self):
        return f"Metric {self.name}={self.value} (epoch {self.epoch}, step {self.step})"

class TrainingCheckpoint(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    training_run = models.ForeignKey(
        TrainingRun,
        on_delete=models.CASCADE,
        related_name='checkpoints'
    )
    step = models.IntegerField()
    epoch = models.IntegerField()
    storage_uri = models.CharField(max_length=1024)
    checksum = models.CharField(max_length=128)
    size_bytes = models.BigIntegerField(default=0)
    metrics = models.JSONField(default=dict)
    status = models.CharField(max_length=50, default='valid')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'training_trainingcheckpoint'
        ordering = ['-created_at']

    def __str__(self):
        return f"Checkpoint epoch {self.epoch} step {self.step} ({self.status})"
