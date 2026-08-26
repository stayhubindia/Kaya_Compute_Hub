import uuid
from django.db import models
from django.conf import settings

class SystemEvent(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    event_type = models.CharField(max_length=64, db_index=True)
    job_id = models.UUIDField(null=True, blank=True, db_index=True)
    worker_id = models.UUIDField(null=True, blank=True, db_index=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    payload = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.event_type} - {self.id} ({self.created_at})"
