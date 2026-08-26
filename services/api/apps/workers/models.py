import uuid
from django.db import models

class WorkerStatusChoices(models.TextChoices):
    OFFLINE = 'offline', 'Offline'
    IDLE = 'idle', 'Idle'
    BUSY = 'busy', 'Busy'
    DRAINING = 'draining', 'Draining'
    UNHEALTHY = 'unhealthy', 'Unhealthy'

class Worker(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, unique=True)
    hostname = models.CharField(max_length=255)
    status = models.CharField(
        max_length=50,
        choices=WorkerStatusChoices.choices,
        default=WorkerStatusChoices.OFFLINE,
        db_index=True
    )
    capabilities = models.JSONField(default=dict, blank=True)
    cpu_count = models.IntegerField(default=0)
    memory_bytes = models.BigIntegerField(default=0)
    gpu_count = models.IntegerField(default=0)
    gpu_model = models.CharField(max_length=255, blank=True)
    gpu_memory_bytes = models.BigIntegerField(default=0)
    cuda_version = models.CharField(max_length=50, blank=True)
    available_gpu_slots = models.IntegerField(default=0)
    allocated_gpu_slots = models.IntegerField(default=0)
    last_heartbeat_at = models.DateTimeField(null=True, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'workers_worker'
        ordering = ['-last_heartbeat_at']

    def __str__(self):
        return f"Worker {self.name} ({self.hostname}) - [{self.status}]"
