import logging
from celery import shared_task

from apps.jobs.models import Job, JobStatusChoices
from apps.jobs.services import transition_job_status, update_job_progress, claim_job_atomically
from apps.audit.services import log_audit_event
from services.worker.executors.colab_executor import ColabExecutionError, run_colab_job

logger = logging.getLogger(__name__)

@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    time_limit=21600,
    soft_time_limit=21300,
)
def execute_job(self, job_id: str, worker_name: str = "celery-worker-01"):
    """
    Celery task entrypoint for async job execution.
    Handles atomic leasing, progress updates, state machine transitions, and safe demo execution.
    """
    logger.info(f"Received execute_job task for job_id={job_id}")

    # 1. Atomically claim job
    job, claimed = claim_job_atomically(job_id, worker_name=worker_name)
    if not claimed or not job:
        logger.warning(f"Job {job_id} could not be claimed (already claimed, cancelled, or missing).")
        return {"status": "skipped", "reason": "job_already_claimed_or_missing"}

    # 2. Check if cancelled before running
    if job.status == JobStatusChoices.CANCELLED:
        logger.info(f"Job {job_id} was cancelled before execution.")
        return {"status": "cancelled"}

    try:
        # 3. Transition to running
        transition_job_status(job, JobStatusChoices.RUNNING)

        def progress_callback(pct: int, stage: str, msg: str):
            # Check if job was cancelled during execution
            current_job = Job.objects.get(id=job.id)
            if current_job.status == JobStatusChoices.CANCELLED:
                raise InterruptedError("Job execution cancelled by user request.")
            update_job_progress(current_job, pct, stage, msg)

        # 4. The VM is control-plane only.  Every accepted Job executes inside
        # the chosen Colab kernel; there is deliberately no local executor
        # fallback for compute work.
        if (job.payload or {}).get("execution_target") != "colab":
            raise RuntimeError("VM execution is disabled. Submit this job to an active Colab session.")
        output_result = run_colab_job(job, progress_callback)

        job.refresh_from_db()
        job.payload = {**(job.payload or {}), "execution_result": output_result}
        job.save(update_fields=["payload", "updated_at"])

        # 5. Transition to succeeded
        transition_job_status(job, JobStatusChoices.SUCCEEDED)
        log_audit_event(
            action="job.completed",
            resource_type="job",
            resource_id=str(job.id),
            metadata={"result": output_result}
        )
        return {"status": "succeeded", "result": output_result}

    except InterruptedError as e:
        logger.info(f"Job {job.id} cancelled during execution.")
        log_audit_event(action="job.cancelled_in_flight", resource_type="job", resource_id=str(job.id))
        return {"status": "cancelled"}

    except NotImplementedError as e:
        # Permanent failure (unsupported job type) -> do not retry
        logger.error(f"Job {job.id} permanent error: {e}")
        transition_job_status(job, JobStatusChoices.FAILED, error_code="NOT_IMPLEMENTED", error_message=str(e))
        return {"status": "failed", "error": str(e)}

    except ColabExecutionError as e:
        # Invalid session/account/Drive state cannot be fixed by a blind retry.
        logger.error(f"Job {job.id} Colab dispatch error: {e}")
        transition_job_status(job, JobStatusChoices.FAILED, error_code="COLAB_EXECUTION_ERROR", error_message=str(e))
        return {"status": "failed", "error": str(e)}

    except Exception as exc:
        logger.exception(f"Job {job.id} encountered exception: {exc}")
        job.retry_count += 1
        job.save()

        # Bounded retries
        if job.retry_count < job.max_retries:
            transition_job_status(job, JobStatusChoices.FAILED, error_code="TRANSIENT_ERROR", error_message=str(exc))
            transition_job_status(job, JobStatusChoices.RETRYING)
            transition_job_status(job, JobStatusChoices.QUEUED)
            raise self.retry(exc=exc, countdown=5 * (2 ** job.retry_count))
        else:
            transition_job_status(job, JobStatusChoices.FAILED, error_code="MAX_RETRIES_EXCEEDED", error_message=str(exc))
            return {"status": "failed", "error": "Max retries exceeded"}
