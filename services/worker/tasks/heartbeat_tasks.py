import os
import socket
from datetime import timedelta
from django.utils import timezone
from celery import shared_task

from apps.workers.models import Worker, WorkerStatusChoices
from apps.audit.services import log_audit_event

@shared_task
def worker_heartbeat_task(worker_name: str = None, status: str = "idle", capabilities: dict = None):
    """
    Periodic task to record worker node heartbeats and hardware capabilities.
    """
    if not worker_name:
        worker_name = f"worker-{socket.gethostname()}"

    hostname = socket.gethostname()
    now = timezone.now()

    worker, created = Worker.objects.get_or_create(
        name=worker_name,
        defaults={
            'hostname': hostname,
            'status': status,
            'capabilities': capabilities or {'docker': True, 'demo_executors': True},
            'cpu_count': os.cpu_count() or 2,
            'memory_bytes': 8 * 1024 * 1024 * 1024,
            'gpu_count': 0,
            'last_heartbeat_at': now
        }
    )

    worker.last_heartbeat_at = now
    if not created and worker.status != WorkerStatusChoices.BUSY:
        worker.status = status
    worker.save()

    return {"worker": worker.name, "status": worker.status, "last_heartbeat_at": worker.last_heartbeat_at.isoformat()}

@shared_task
def mark_stale_workers_task(stale_seconds: int = 60):
    """
    Scans worker fleet and marks nodes as offline/unhealthy if last heartbeat exceeds threshold.
    """
    cutoff = timezone.now() - timedelta(seconds=stale_seconds)
    stale_workers = Worker.objects.filter(
        last_heartbeat_at__lt=cutoff
    ).exclude(status__in=[WorkerStatusChoices.OFFLINE, WorkerStatusChoices.UNHEALTHY])

    count = 0
    for worker in stale_workers:
        previous_status = worker.status
        worker.status = WorkerStatusChoices.OFFLINE
        worker.save(update_fields=['status', 'updated_at'])
        count += 1

        log_audit_event(
            action="worker.marked_offline",
            resource_type="worker",
            resource_id=str(worker.id),
            metadata={"previous_status": previous_status, "stale_cutoff_seconds": stale_seconds}
        )

    return {"stale_workers_marked_offline": count}
