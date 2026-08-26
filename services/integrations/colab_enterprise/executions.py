class ExternalRunStatus:
    REQUESTED = "requested"
    AUTHORIZING = "authorizing"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    CHOICES = [
        (REQUESTED, "Requested"),
        (AUTHORIZING, "Authorizing"),
        (SUBMITTED, "Submitted"),
        (RUNNING, "Running"),
        (COMPLETED, "Completed"),
        (FAILED, "Failed"),
        (CANCELLED, "Cancelled"),
        (TIMED_OUT, "Timed Out"),
    ]

def map_colab_state_to_run_status(gcp_state: str) -> str:
    """Map GCP Vertex AI Notebook Execution State to ExternalRunStatus."""
    state_map = {
        "JOB_STATE_QUEUED": ExternalRunStatus.SUBMITTED,
        "JOB_STATE_PENDING": ExternalRunStatus.SUBMITTED,
        "JOB_STATE_RUNNING": ExternalRunStatus.RUNNING,
        "JOB_STATE_SUCCEEDED": ExternalRunStatus.COMPLETED,
        "JOB_STATE_FAILED": ExternalRunStatus.FAILED,
        "JOB_STATE_CANCELLED": ExternalRunStatus.CANCELLED,
        "JOB_STATE_EXPIRED": ExternalRunStatus.TIMED_OUT,
    }
    return state_map.get(gcp_state, ExternalRunStatus.RUNNING)
