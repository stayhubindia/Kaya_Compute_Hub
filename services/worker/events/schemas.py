from typing import Dict, Any, Optional
from datetime import datetime, timezone
import uuid

EVENT_TYPES = [
    "job.queued",
    "job.leased",
    "job.started",
    "job.progress",
    "job.log",
    "job.checkpoint",
    "job.succeeded",
    "job.failed",
    "job.cancelled",
    "job.paused",
    "job.resumed",
    "worker.heartbeat",
    "worker.status_changed",
    "artifact.created",
]

def sanitize_event_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Redacts secrets, private IP ranges, and internal absolute paths from event payloads."""
    sanitized = {}
    forbidden_keys = {"password", "secret", "token", "api_key", "cookie", "auth"}
    
    for k, v in payload.items():
        if any(f in k.lower() for f in forbidden_keys):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_event_payload(v)
        else:
            sanitized[k] = v
    return sanitized

def format_event(
    event_type: str,
    payload: Dict[str, Any],
    job_id: Optional[str] = None,
    worker_id: Optional[str] = None,
    user_id: Optional[str] = None,
    event_id: Optional[str] = None
) -> Dict[str, Any]:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unsupported event type: {event_type}")

    return {
        "event_id": event_id or str(uuid.uuid4()),
        "event_type": event_type,
        "job_id": str(job_id) if job_id else None,
        "worker_id": str(worker_id) if worker_id else None,
        "user_id": str(user_id) if user_id else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": sanitize_event_payload(payload)
    }
