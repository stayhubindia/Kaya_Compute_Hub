import uuid
from django.db import models
from django.conf import settings

class JobTypeChoices(models.TextChoices):
    DOWNLOAD = 'download', 'Download'
    EXTRACTION = 'extraction', 'Extraction'
    PREPROCESSING = 'preprocessing', 'Preprocessing'
    INGESTION = 'ingest_documents', 'Ingest Documents'
    GENERATION = 'generate_candidates', 'Generate Candidates'
    QUALITY_AUDIT = 'run_quality_audit', 'Run Quality Audit'
    FREEZE_DATASET = 'freeze_dataset', 'Freeze Dataset'
    TRAINING_QLORA = 'train_qlora', 'Train QLoRA'
    EVALUATION = 'evaluate_model', 'Evaluate Model'
    SYNC_DRIVE = 'sync_to_drive', 'Sync to Drive'
    CUSTOM_SCRIPT = 'custom_script', 'Custom Script'
    # Backward-compatible public API values retained for existing clients.
    NOTEBOOK = 'notebook', 'Notebook'
    TRAINING = 'training', 'Training'

class JobStatusChoices(models.TextChoices):
    DRAFT = 'draft', 'Draft'
    QUEUED = 'queued', 'Queued'
    LEASED = 'leased', 'Leased'
    RUNNING = 'running', 'Running'
    SUCCEEDED = 'succeeded', 'Succeeded'
    FAILED = 'failed', 'Failed'
    CANCELLED = 'cancelled', 'Cancelled'
    RETRYING = 'retrying', 'Retrying'

class Job(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='jobs',
        db_index=True
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    job_type = models.CharField(
        max_length=50,
        choices=JobTypeChoices.choices,
        db_index=True
    )
    status = models.CharField(
        max_length=50,
        choices=JobStatusChoices.choices,
        default=JobStatusChoices.DRAFT,
        db_index=True
    )
    priority = models.IntegerField(default=0)
    payload = models.JSONField(default=dict, blank=True)
    idempotency_key = models.CharField(max_length=255, null=True, blank=True)
    selected_google_account = models.ForeignKey(
        'integrations.ConnectedAccount',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='jobs'
    )
    
    # Progress & Execution Tracking
    progress_percentage = models.IntegerField(default=0)
    current_stage = models.CharField(max_length=100, default='queued', blank=True)
    progress_message = models.CharField(max_length=255, default='', blank=True)
    assigned_worker = models.ForeignKey(
        'workers.Worker',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='assigned_jobs'
    )
    retry_count = models.IntegerField(default=0)
    max_retries = models.IntegerField(default=3)

    error_code = models.CharField(max_length=100, null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'jobs_job'
        ordering = ['-created_at']
        constraints = [
            models.UniqueConstraint(
                fields=['created_by', 'idempotency_key'],
                name='unique_user_idempotency_key',
                condition=models.Q(idempotency_key__isnull=False)
            )
        ]
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['job_type']),
            models.Index(fields=['created_by']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"Job {self.name} ({self.job_type}) - [{self.status}] ({self.progress_percentage}%)"
