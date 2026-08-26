"""
Celery & Thread-safe task: ArXiv Batch Download Job.
Runs the headless ArXivBatchEngine and updates Kaya Job status/metadata.
"""
import threading
import logging
from pathlib import Path

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)

COLAB_GDRIVE_BASE = Path("/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv")
FALLBACK_LOCAL_BASE = Path("/tmp/kaya_arxiv_downloads")


def resolve_arxiv_output_path(month_str: str, custom_dir: str = "") -> Path:
    """
    Resolves year-wise output directory matching Colab Drive format:
    /content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv/<year>/[pdf|html]
    """
    import re
    m = re.search(r"\b(19\d\d|20\d\d)\b", month_str)
    year = m.group(1) if m else str(timezone.now().year)

    if custom_dir:
        base_path = Path(custom_dir)
    else:
        base_path = COLAB_GDRIVE_BASE

    try:
        out_path = base_path / year
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning(f"[ARXIV OUTPUT] Cannot write to {base_path}: {exc}. Falling back to {FALLBACK_LOCAL_BASE}")
        out_path = FALLBACK_LOCAL_BASE / year
        out_path.mkdir(parents=True, exist_ok=True)

    return out_path


@shared_task(bind=True, max_retries=0, name="tasks.arxiv_batch_download")
def arxiv_batch_download_task(self_or_job_id, job_id: str = None, category: str = "", month: str = "",
                               workers: int = 4, delay: float = 1.0, output_dir: str = ""):
    """
    Task to run a full ArXiv batch download (discover + download).
    Supports both Celery task execution (.delay()) and direct Thread execution.
    """
    # Disambiguate arguments between Celery bound task and direct call
    if isinstance(self_or_job_id, str):
        # Direct function call: self_or_job_id is job_id
        actual_job_id = self_or_job_id
        actual_category = job_id or ""
        actual_month = category or ""
        actual_workers = month if isinstance(month, int) else workers
        actual_delay = workers if isinstance(workers, (int, float)) else delay
    else:
        actual_job_id = job_id
        actual_category = category
        actual_month = month
        actual_workers = workers
        actual_delay = delay

    from apps.jobs.models import Job, JobStatusChoices

    job = None
    if actual_job_id:
        try:
            job = Job.objects.get(id=actual_job_id)
            job.status = JobStatusChoices.RUNNING
            job.started_at = timezone.now()
            job.save(update_fields=["status", "started_at", "updated_at"])
        except Exception as exc:
            logger.warning(f"[ARXIV TASK] Could not fetch job {actual_job_id}: {exc}")

    # Resolve year-wise output path (/content/drive/MyDrive/Colab Notebooks/Datasets/Arxiv/<year>)
    out_path = resolve_arxiv_output_path(actual_month, output_dir)

    # Stop event tied to job cancellation check
    stop_event = threading.Event()

    def on_progress(stats: dict):
        """Called after each paper — update job payload & progress fields."""
        if job:
            try:
                proc = stats.get("processed", 0)
                tot = stats.get("total", 1) or 1
                pct = min(100, int((proc / tot) * 100))
                job.progress_percentage = pct
                job.progress_message = f"Downloaded {stats.get('html', 0)} HTML, {stats.get('pdf', 0)} PDF papers"
                job.payload = {
                    **(job.payload or {}),
                    "arxiv_stats": stats,
                    "output_dir": str(out_path),
                }
                job.save(update_fields=["payload", "progress_percentage", "progress_message", "updated_at"])
            except Exception:
                pass
        # Check if job was cancelled externally
        if job:
            try:
                job.refresh_from_db(fields=["status"])
                if job.status == JobStatusChoices.CANCELLED:
                    stop_event.set()
            except Exception:
                pass

    from services.downloader.providers.arxiv_batch import ArXivBatchEngine

    engine = ArXivBatchEngine(
        output_dir=out_path,
        workers=actual_workers,
        delay=actual_delay,
        on_progress=on_progress,
        stop_event=stop_event,
    )

    final_stats = engine.run(category=actual_category, month=actual_month)
    final_stats["output_dir"] = str(out_path)
    final_stats["category"]   = actual_category
    final_stats["month"]      = actual_month

    # Finish job
    if job:
        try:
            job.refresh_from_db(fields=["status"])
            if job.status not in [JobStatusChoices.CANCELLED]:
                job.status = JobStatusChoices.SUCCEEDED
            job.finished_at = timezone.now()
            job.progress_percentage = 100
            job.progress_message = f"Batch download complete: {final_stats.get('processed', 0)} papers"
            job.payload = {**(job.payload or {}), "arxiv_stats": final_stats}
            job.save(update_fields=["status", "finished_at", "progress_percentage", "progress_message", "payload", "updated_at"])
        except Exception as exc:
            logger.error(f"[ARXIV TASK] Final save error: {exc}")

    logger.info(f"[ARXIV TASK] Done: {final_stats}")
    return final_stats
