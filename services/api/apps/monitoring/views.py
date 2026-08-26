from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions, status
from django.shortcuts import get_object_or_404
from django.utils import timezone
from apps.workers.models import Worker, WorkerStatusChoices
from apps.jobs.models import Job, JobStatusChoices

class WorkerListView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        now = timezone.now()
        workers = Worker.objects.all().order_by("name")
        serialized = []

        for worker in workers:
            is_stale = (now - worker.last_heartbeat_at).total_seconds() > 60 if worker.last_heartbeat_at else True
            active_jobs_count = Job.objects.filter(assigned_worker=worker, status=JobStatusChoices.RUNNING).count()
            
            # Map state
            effective_status = worker.status
            if is_stale and worker.status != WorkerStatusChoices.OFFLINE:
                effective_status = "unhealthy"

            serialized.append({
                "id": str(worker.id),
                "name": worker.name,
                "hostname_label": worker.hostname.split(".")[0],  # Sanitized hostname label
                "status": effective_status,
                "last_heartbeat_at": worker.last_heartbeat_at.isoformat() if worker.last_heartbeat_at else None,
                "is_stale": is_stale,
                "cpu_count": worker.cpu_count,
                "memory_bytes": worker.memory_bytes,
                "gpu_count": worker.gpu_count,
                "gpu_model": worker.gpu_model or "N/A",
                "available_gpu_slots": worker.available_gpu_slots,
                "allocated_gpu_slots": worker.allocated_gpu_slots,
                "active_jobs_count": active_jobs_count,
                "capabilities": worker.capabilities
            })

        return Response({"workers": serialized})

class WorkerMetricsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, worker_id):
        worker = get_object_or_404(Worker, id=worker_id)
        now = timezone.now()
        is_stale = (now - worker.last_heartbeat_at).total_seconds() > 60 if worker.last_heartbeat_at else True

        active_jobs = Job.objects.filter(assigned_worker=worker, status=JobStatusChoices.RUNNING)

        return Response({
            "worker_id": str(worker.id),
            "name": worker.name,
            "status": worker.status,
            "is_stale": is_stale,
            "cpu_count": worker.cpu_count,
            "memory_bytes": worker.memory_bytes,
            "gpu_count": worker.gpu_count,
            "gpu_model": worker.gpu_model or "N/A",
            "available_gpu_slots": worker.available_gpu_slots,
            "allocated_gpu_slots": worker.allocated_gpu_slots,
            "active_job_ids": [str(j.id) for j in active_jobs]
        })
