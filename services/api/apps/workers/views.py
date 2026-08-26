from django.utils import timezone
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.workers.models import Worker, WorkerStatusChoices
from apps.workers.serializers import WorkerSerializer, WorkerHeartbeatSerializer

class WorkerViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Worker.objects.all()
    serializer_class = WorkerSerializer
    permission_classes = [permissions.IsAuthenticated]
    filterset_fields = ['status', 'hostname']
    search_fields = ['name', 'hostname']
    ordering_fields = ['last_heartbeat_at', 'created_at']

    @action(detail=True, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def heartbeat(self, request, pk=None):
        worker = self.get_object()
        serializer = WorkerHeartbeatSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        worker.last_heartbeat_at = timezone.now()
        new_status = serializer.validated_data.get('status')
        if new_status:
            worker.status = new_status
        elif worker.status == WorkerStatusChoices.OFFLINE:
            worker.status = WorkerStatusChoices.IDLE

        capabilities = serializer.validated_data.get('capabilities')
        if capabilities is not None:
            worker.capabilities = capabilities

        worker.save()
        return Response(WorkerSerializer(worker).data, status=status.HTTP_200_OK)

    @action(detail=True, methods=['get'])
    def metrics(self, request, pk=None):
        worker = self.get_object()
        from apps.jobs.models import Job, JobStatusChoices
        active_jobs = Job.objects.filter(assigned_worker=worker, status=JobStatusChoices.RUNNING)

        return Response({
            "worker_id": str(worker.id),
            "name": worker.name,
            "status": worker.status,
            "is_stale": (timezone.now() - worker.last_heartbeat_at).total_seconds() > 60 if worker.last_heartbeat_at else True,
            "cpu_count": worker.cpu_count,
            "memory_bytes": worker.memory_bytes,
            "gpu_count": worker.gpu_count,
            "gpu_model": worker.gpu_model or "N/A",
            "available_gpu_slots": worker.available_gpu_slots,
            "allocated_gpu_slots": worker.allocated_gpu_slots,
            "active_job_ids": [str(j.id) for j in active_jobs]
        })
