import uuid
from django.db import models
from django.conf import settings
from apps.training.models import TrainingRun
from apps.artifacts.models import Artifact

class ModelStatusChoices(models.TextChoices):
    REGISTERED = 'registered', 'Registered'
    VALIDATING = 'validating', 'Validating'
    APPROVED = 'approved', 'Approved'
    ARCHIVED = 'archived', 'Archived'
    REJECTED = 'rejected', 'Rejected'

class ModelVersion(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    version = models.CharField(max_length=50)
    training_run = models.ForeignKey(
        TrainingRun,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='registered_models'
    )
    artifact = models.ForeignKey(
        Artifact,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='model_versions'
    )
    framework = models.CharField(max_length=100)
    framework_version = models.CharField(max_length=50, blank=True)
    model_format = models.CharField(max_length=50)
    checksum = models.CharField(max_length=128)
    metadata = models.JSONField(default=dict, help_text="Hyperparameters, dataset lineage, metrics summary")
    status = models.CharField(
        max_length=50,
        choices=ModelStatusChoices.choices,
        default=ModelStatusChoices.REGISTERED,
        db_index=True
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='model_versions'
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = 'models_registry_modelversion'
        unique_together = ('name', 'version')
        ordering = ['-created_at']

    def __str__(self):
        return f"ModelVersion {self.name}:{self.version} [{self.status}]"
