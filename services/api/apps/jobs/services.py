from django.db import transaction
from django.utils import timezone
from rest_framework.exceptions import ValidationError
from apps.jobs.models import Job, JobStatusChoices
from apps.workers.models import Worker, WorkerStatusChoices
from apps.audit.services import log_audit_event

ALLOWED_TRANSITIONS = {
    JobStatusChoices.DRAFT: [JobStatusChoices.QUEUED],
    JobStatusChoices.QUEUED: [JobStatusChoices.LEASED, JobStatusChoices.CANCELLED, JobStatusChoices.FAILED],
    JobStatusChoices.LEASED: [JobStatusChoices.RUNNING, JobStatusChoices.CANCELLED, JobStatusChoices.FAILED],
    JobStatusChoices.RUNNING: [
        JobStatusChoices.SUCCEEDED,
        JobStatusChoices.FAILED,
        JobStatusChoices.CANCELLED,
    ],
    JobStatusChoices.FAILED: [JobStatusChoices.RETRYING],
    JobStatusChoices.RETRYING: [JobStatusChoices.QUEUED, JobStatusChoices.LEASED, JobStatusChoices.FAILED],
}

def transition_job_status(
    job: Job,
    new_status: str,
    actor=None,
    error_code: str = None,
    error_message: str = None,
    request=None
) -> Job:
    """
    State machine transition validator and executor for Job lifecycle.
    """
    current_status = job.status
    allowed = ALLOWED_TRANSITIONS.get(current_status, [])

    if new_status not in allowed:
        raise ValidationError({
            "status": f"Invalid status transition from '{current_status}' to '{new_status}'. Allowed target statuses: {allowed}"
        })

    job.status = new_status

    if error_code:
        job.error_code = error_code
    if error_message:
        job.error_message = error_message

    now = timezone.now()
    if new_status == JobStatusChoices.RUNNING:
        job.started_at = now
        job.current_stage = 'running'
    elif new_status == JobStatusChoices.SUCCEEDED:
        job.finished_at = now
        job.progress_percentage = 100
        job.current_stage = 'completed'
    elif new_status in [JobStatusChoices.FAILED, JobStatusChoices.CANCELLED]:
        job.finished_at = now
        job.current_stage = new_status

    job.save()

    # Log state change audit event
    log_audit_event(
        action=f"job.transition.{new_status}",
        resource_type="job",
        resource_id=str(job.id),
        actor=actor,
        metadata={
            "previous_status": current_status,
            "new_status": new_status,
            "error_code": error_code,
            "error_message": error_message,
        },
        request=request
    )

    return job

def update_job_progress(job: Job, percentage: int, stage: str = None, message: str = None) -> Job:
    """
    Safely update job progress percentage and status message.
    """
    if percentage < 0:
        percentage = 0
    if percentage > 100:
        percentage = 100

    job.progress_percentage = percentage
    if stage:
        job.current_stage = stage
    if message is not None:
        job.progress_message = message

    job.save(update_fields=['progress_percentage', 'current_stage', 'progress_message', 'updated_at'])
    return job

def claim_job_atomically(job_id: str, worker_name: str = "default-worker") -> tuple[Job, bool]:
    """
    Atomically claims a queued/retrying job using row-level locking (select_for_update).
    Returns (job, claimed_successfully).
    """
    with transaction.atomic():
        try:
            job = Job.objects.select_for_update().get(id=job_id)
        except Job.DoesNotExist:
            return None, False

        # Only queued or retrying jobs can be leased
        if job.status not in [JobStatusChoices.QUEUED, JobStatusChoices.RETRYING]:
            return job, False

        # Register or get worker instance
        worker, _ = Worker.objects.get_or_create(
            name=worker_name,
            defaults={
                'hostname': 'localhost',
                'status': WorkerStatusChoices.BUSY,
                'last_heartbeat_at': timezone.now()
            }
        )
        worker.status = WorkerStatusChoices.BUSY
        worker.last_heartbeat_at = timezone.now()
        worker.save()

        job.assigned_worker = worker
        job.status = JobStatusChoices.LEASED
        job.current_stage = 'leased'
        job.save()

        log_audit_event(
            action="job.leased",
            resource_type="job",
            resource_id=str(job.id),
            metadata={"worker": worker.name}
        )
        return job, True
