import uuid
from django.db import models
from apps.jobs.models import Job

class LogLevelChoices(models.TextChoices):
    DEBUG = 'debug', 'Debug'
    INFO = 'info', 'Info'
    WARNING = 'warning', 'Warning'
    ERROR = 'error', 'Error'

class JobLog(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='logs')
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    level = models.CharField(max_length=16, choices=LogLevelChoices.choices, default=LogLevelChoices.INFO, db_index=True)
    message = models.TextField()
    module = models.CharField(max_length=64, default='general')

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"[{self.level.upper()}] {self.job_id}: {self.message[:40]}"
