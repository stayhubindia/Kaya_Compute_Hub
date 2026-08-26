import html
import re
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from apps.jobs.models import Job
from apps.logs.models import JobLog
from apps.accounts.permissions import IsAuthenticatedAdmin

SECRET_PATTERNS = [
    (r'(?i)(password|secret|token|key|api_key|auth)\s*[:=]\s*["\']?([^"\'\s]+)["\']?', r'\1: [REDACTED]'),
    (r'Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*', r'Bearer [REDACTED]'),
]

def sanitize_log_message(message: str) -> str:
    clean = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', message)
    for pattern, repl in SECRET_PATTERNS:
        clean = re.sub(pattern, repl, clean)
    return html.escape(clean)

class JobLogsView(APIView):
    permission_classes = [IsAuthenticatedAdmin]

    def get(self, request, job_id):
        job = get_object_or_404(Job, id=job_id)
        queryset = JobLog.objects.filter(job=job)

        level = request.query_params.get("level")
        if level:
            queryset = queryset.filter(level=level.lower())

        since = request.query_params.get("since")
        if since:
            queryset = queryset.filter(timestamp__gt=since)

        try:
            page_size = min(int(request.query_params.get("page_size", 100)), 1000)
        except ValueError:
            page_size = 100

        logs = queryset.order_by("timestamp")[:page_size]

        serialized = [
            {
                "id": str(log.id),
                "job_id": str(log.job_id),
                "timestamp": log.timestamp.isoformat(),
                "level": log.level,
                "module": log.module,
                "message": sanitize_log_message(log.message)
            }
            for log in logs
        ]

        return Response({
            "job_id": str(job.id),
            "count": len(serialized),
            "logs": serialized
        })
