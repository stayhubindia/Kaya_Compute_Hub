from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.accounts.permissions import IsAuthenticatedAdmin
from apps.training.models import TrainingRun, TrainingMetric, TrainingCheckpoint, TrainingRunStatus
from apps.training.serializers import (
    TrainingRunSerializer, TrainingRunCreateSerializer,
    TrainingMetricSerializer, TrainingCheckpointSerializer
)
from apps.audit.services import log_audit_event

class TrainingRunViewSet(viewsets.ModelViewSet):
    serializer_class = TrainingRunSerializer
    permission_classes = [IsAuthenticatedAdmin]

    def get_queryset(self):
        return TrainingRun.objects.all()

    def create(self, request, *args, **kwargs):
        serializer = TrainingRunCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        dataset = serializer.validated_data['dataset_id']
        processing_run = serializer.validated_data.get('processing_run_id')

        training_run = TrainingRun.objects.create(
            name=serializer.validated_data['name'],
            dataset=dataset,
            processing_run=processing_run,
            backend=serializer.validated_data['backend'],
            container_image=serializer.validated_data['container_image'],
            configuration=serializer.validated_data['configuration'],
            resource_policy=serializer.validated_data.get('resource_policy', {}),
            status=TrainingRunStatus.QUEUED,
            created_by=request.user
        )

        log_audit_event(
            action="training.created",
            resource_type="training_run",
            resource_id=str(training_run.id),
            actor=request.user,
            metadata={"name": training_run.name, "backend": training_run.backend},
            request=request
        )

        # Enqueue Celery Task
        try:
            from services.worker.tasks.training_tasks import execute_training_run
            execute_training_run.delay(str(training_run.id))
        except Exception:
            pass

        return Response({
            "id": str(training_run.id),
            "status": training_run.status,
            "message": "Training run enqueued successfully."
        }, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        run_obj = self.get_object()
        if run_obj.status in [TrainingRunStatus.SUCCEEDED, TrainingRunStatus.FAILED, TrainingRunStatus.CANCELLED]:
            return Response({"message": f"Training run is already in terminal state '{run_obj.status}'."}, status=status.HTTP_400_BAD_REQUEST)

        run_obj.status = TrainingRunStatus.CANCELLED
        run_obj.save(update_fields=['status', 'updated_at'])

        log_audit_event(
            action="training.cancelled",
            resource_type="training_run",
            resource_id=str(run_obj.id),
            actor=request.user,
            request=request
        )

        return Response({"id": str(run_obj.id), "status": run_obj.status, "message": "Training run cancelled."})

    @action(detail=True, methods=['post'])
    def pause(self, request, pk=None):
        run_obj = self.get_object()
        if run_obj.status not in [TrainingRunStatus.RUNNING, TrainingRunStatus.QUEUED]:
            return Response({"message": f"Cannot pause run in state '{run_obj.status}'."}, status=status.HTTP_400_BAD_REQUEST)

        run_obj.status = TrainingRunStatus.PAUSED
        run_obj.save(update_fields=['status', 'updated_at'])

        log_audit_event(
            action="training.paused",
            resource_type="training_run",
            resource_id=str(run_obj.id),
            actor=request.user,
            request=request
        )

        return Response({"id": str(run_obj.id), "status": run_obj.status, "message": "Training run paused."})

    @action(detail=True, methods=['post'])
    def resume(self, request, pk=None):
        run_obj = self.get_object()
        if run_obj.status not in [TrainingRunStatus.PAUSED, TrainingRunStatus.FAILED]:
            return Response({"message": f"Cannot resume run in state '{run_obj.status}'."}, status=status.HTTP_400_BAD_REQUEST)

        run_obj.status = TrainingRunStatus.QUEUED
        run_obj.save(update_fields=['status', 'updated_at'])

        log_audit_event(
            action="training.resumed",
            resource_type="training_run",
            resource_id=str(run_obj.id),
            actor=request.user,
            request=request
        )

        try:
            from services.worker.tasks.training_tasks import execute_training_run
            execute_training_run.delay(str(run_obj.id))
        except Exception:
            pass

        return Response({"id": str(run_obj.id), "status": run_obj.status, "message": "Training run resumed."})

    @action(detail=True, methods=['get'])
    def metrics(self, request, pk=None):
        run_obj = self.get_object()
        metrics = run_obj.metrics.all()
        serializer = TrainingMetricSerializer(metrics, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def checkpoints(self, request, pk=None):
        run_obj = self.get_object()
        checkpoints = run_obj.checkpoints.all()
        serializer = TrainingCheckpointSerializer(checkpoints, many=True)
        return Response(serializer.data)
