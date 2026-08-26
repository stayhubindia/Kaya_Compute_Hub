import json
import time
from django.http import StreamingHttpResponse, JsonResponse
from django.views import View
from apps.events.models import SystemEvent

class SSEEventStreamView(View):
    def get(self, request):
        if not request.user.is_authenticated or not request.user.is_active:
            return JsonResponse({"detail": "Authentication credentials were not provided."}, status=401)

        user = request.user
        job_id = request.GET.get("job_id")
        worker_id = request.GET.get("worker_id")
        last_event_id = request.GET.get("last_event_id")

        def event_generator():
            init_data = json.dumps({"status": "connected", "user": user.email})
            yield f"event: connection\ndata: {init_data}\n\n"

            last_created_at = None
            if last_event_id:
                try:
                    last_evt = SystemEvent.objects.get(id=last_event_id)
                    last_created_at = last_evt.created_at
                except SystemEvent.DoesNotExist:
                    pass

            query = SystemEvent.objects.all()
            if job_id:
                query = query.filter(job_id=job_id)
            if worker_id:
                query = query.filter(worker_id=worker_id)
            if last_created_at:
                query = query.filter(created_at__gt=last_created_at)

            for evt in query.order_by("created_at")[:100]:
                payload_str = json.dumps({
                    "event_id": str(evt.id),
                    "event_type": evt.event_type,
                    "job_id": str(evt.job_id) if evt.job_id else None,
                    "worker_id": str(evt.worker_id) if evt.worker_id else None,
                    "timestamp": evt.created_at.isoformat(),
                    "payload": evt.payload
                })
                yield f"id: {evt.id}\nevent: {evt.event_type}\ndata: {payload_str}\n\n"

            for _ in range(5):
                time.sleep(2)
                yield ": keepalive\n\n"

        response = StreamingHttpResponse(
            event_generator(),
            content_type="text/event-stream"
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response
