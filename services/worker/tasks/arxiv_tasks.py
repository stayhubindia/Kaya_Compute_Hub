"""
Celery task: ArXiv Batch Download Job.
Runs the headless ArXivBatchEngine and writes progress to Download model.
"""
import threading
import logging
from pathlib import Path

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)

ARXIV_OUTPUT_BASE = Path("/tmp/kaya_arxiv_downloads")


@shared_task(bind=True, max_retries=0, name="tasks.arxiv_batch_download")
def arxiv_batch_download_task(self, job_id: str, category: str, month: str,
                               workers: int = 4, delay: float = 1.0,
                               output_dir: str = ""):
    """
    Celery task to run a full ArXiv batch download (discover + download).

    Args:
        job_id:     Kaya Job UUID (used to update status in DB).
        category:   ArXiv category string e.g. "cs.AI", "astro-ph", "quant-ph".
        month:      Month string e.g. "2025-01", "2024-06".
        workers:    Parallel download threads (1-6).
        delay:      Base inter-paper polite delay in seconds.
        output_dir: Override output directory (uses /tmp/kaya_arxiv_downloads/<category>/<month> by default).
    """
    from apps.jobs.models import Job, JobStatusChoices

    job = None
    try:
        job = Job.objects.get(id=job_id)
        job.status = JobStatusChoices.RUNNING
        job.started_at = timezone.now()
        job.save(update_fields=["status", "started_at", "updated_at"])
    except Exception as exc:
        logger.warning(f"[ARXIV TASK] Could not fetch job {job_id}: {exc}")

    # Resolve output path
    if not output_dir:
        out_path = ARXIV_OUTPUT_BASE / category.replace("/", "_") / month
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Stop event tied to job cancellation check
    stop_event = threading.Event()

    def on_progress(stats: dict):
        """Called after each paper — update job metadata."""
        if job:
            try:
                job.metadata = {
                    **(job.metadata or {}),
                    "arxiv_stats": stats,
                    "output_dir": str(out_path),
                }
                job.save(update_fields=["metadata", "updated_at"])
            except Exception:
                pass
        # Check if job was cancelled externally
        try:
            job.refresh_from_db(fields=["status"])
            if job.status == JobStatusChoices.CANCELLED:
                stop_event.set()
        except Exception:
            pass

    from services.downloader.providers.arxiv_batch import ArXivBatchEngine

    engine = ArXivBatchEngine(
        output_dir=out_path,
        workers=workers,
        delay=delay,
        on_progress=on_progress,
        stop_event=stop_event,
    )

    final_stats = engine.run(category=category, month=month)
    final_stats["output_dir"] = str(out_path)
    final_stats["category"]   = category
    final_stats["month"]      = month

    # Finish job
    if job:
        try:
            job.refresh_from_db(fields=["status"])
            if job.status not in [JobStatusChoices.CANCELLED]:
                job.status = JobStatusChoices.COMPLETED
            job.finished_at = timezone.now()
            job.metadata = {**(job.metadata or {}), "arxiv_stats": final_stats}
            job.save(update_fields=["status", "finished_at", "metadata", "updated_at"])
        except Exception as exc:
            logger.error(f"[ARXIV TASK] Final save error: {exc}")

    logger.info(f"[ARXIV TASK] Done: {final_stats}")
    return final_stats
