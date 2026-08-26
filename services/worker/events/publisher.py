import logging
from typing import Dict, Any, Optional
from services.worker.events.schemas import format_event

logger = logging.getLogger(__name__)

class EventPublisher:
    @staticmethod
    def publish_event(
        event_type: str,
        payload: Dict[str, Any],
        job_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        formatted = format_event(
            event_type=event_type,
            payload=payload,
            job_id=job_id,
            worker_id=worker_id,
            user_id=user_id
        )

        try:
            from apps.events.models import SystemEvent
            SystemEvent.objects.create(
                id=formatted["event_id"],
                event_type=formatted["event_type"],
                job_id=formatted["job_id"],
                worker_id=formatted["worker_id"],
                user_id=formatted["user_id"],
                payload=formatted["payload"]
            )
        except Exception as e:
            logger.warning(f"Failed to persist SystemEvent record: {e}")

        return formatted
