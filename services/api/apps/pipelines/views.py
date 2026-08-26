from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.accounts.permissions import IsAuthenticatedAdmin
from apps.datasets.models import Dataset
from apps.pipelines.models import PipelineDefinition, ProcessingRun, ProcessingStageEvent, DatasetManifest, ProcessingRunStatus
from apps.pipelines.serializers import (
    PipelineDefinitionSerializer, ProcessingRunSerializer,
    ProcessingRunCreateSerializer, ProcessingStageEventSerializer,
    DatasetManifestSerializer
)
from apps.audit.services import log_audit_event

class PipelineDefinitionViewSet(viewsets.ModelViewSet):
    serializer_class = PipelineDefinitionSerializer
    permission_classes = [IsAuthenticatedAdmin]

    def get_queryset(self):
        return PipelineDefinition.objects.all()

    def perform_create(self, serializer):
        pipeline_obj = serializer.save(created_by=self.request.user)
        log_audit_event(
            action="pipeline.created",
            resource_type="pipeline",
            resource_id=str(pipeline_obj.id),
            actor=self.request.user,
            metadata={"name": pipeline_obj.name, "version": pipeline_obj.version},
            request=self.request
        )

class ProcessingRunViewSet(viewsets.ModelViewSet):
    serializer_class = ProcessingRunSerializer
    permission_classes = [IsAuthenticatedAdmin]

    def get_queryset(self):
        return ProcessingRun.objects.all()

    def create(self, request, *args, **kwargs):
        serializer = ProcessingRunCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        pipeline_id = serializer.validated_data['pipeline_id']
        source_dataset_id = serializer.validated_data['source_dataset_id']

        try:
            pipeline_obj = PipelineDefinition.objects.get(id=pipeline_id, enabled=True)
        except PipelineDefinition.DoesNotExist:
            return Response({"error": {"code": "PIPELINE_NOT_FOUND", "message": "Pipeline definition not found or disabled."}}, status=status.HTTP_404_NOT_FOUND)

        try:
            dataset_obj = Dataset.objects.get(id=source_dataset_id)
        except Dataset.DoesNotExist:
            return Response({"error": {"code": "DATASET_NOT_FOUND", "message": "Source dataset not found."}}, status=status.HTTP_404_NOT_FOUND)

        processing_run = ProcessingRun.objects.create(
            pipeline=pipeline_obj,
            source_dataset=dataset_obj,
            status=ProcessingRunStatus.QUEUED,
            created_by=request.user
        )

        log_audit_event(
            action="processing_run.created",
            resource_type="processing_run",
            resource_id=str(processing_run.id),
            actor=request.user,
            metadata={"pipeline": str(pipeline_obj.id), "source_dataset": str(dataset_obj.id)},
            request=request
        )

        # Enqueue Celery Task
        try:
            from services.worker.tasks.processing_tasks import execute_processing_run
            execute_processing_run.delay(str(processing_run.id))
        except Exception:
            pass

        return Response({
            "id": str(processing_run.id),
            "status": processing_run.status,
            "message": "Processing run enqueued successfully."
        }, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        run_obj = self.get_object()
        if run_obj.status in [ProcessingRunStatus.SUCCEEDED, ProcessingRunStatus.FAILED, ProcessingRunStatus.CANCELLED]:
            return Response({"message": f"Processing run is already in terminal state '{run_obj.status}'."}, status=status.HTTP_400_BAD_REQUEST)

        run_obj.status = ProcessingRunStatus.CANCELLED
        run_obj.save(update_fields=['status', 'updated_at'])

        log_audit_event(
            action="processing_run.cancelled",
            resource_type="processing_run",
            resource_id=str(run_obj.id),
            actor=request.user,
            request=request
        )

        return Response({"id": str(run_obj.id), "status": run_obj.status, "message": "Processing run cancelled."})

    @action(detail=True, methods=['post'])
    def pause(self, request, pk=None):
        run_obj = self.get_object()
        if run_obj.status not in [ProcessingRunStatus.RUNNING, ProcessingRunStatus.QUEUED, ProcessingRunStatus.VALIDATING, ProcessingRunStatus.CHECKPOINTING]:
            return Response({"message": f"Cannot pause run in state '{run_obj.status}'."}, status=status.HTTP_400_BAD_REQUEST)

        run_obj.status = ProcessingRunStatus.PAUSED
        run_obj.save(update_fields=['status', 'updated_at'])

        log_audit_event(
            action="processing_run.paused",
            resource_type="processing_run",
            resource_id=str(run_obj.id),
            actor=request.user,
            request=request
        )

        return Response({"id": str(run_obj.id), "status": run_obj.status, "message": "Processing run paused."})

    @action(detail=True, methods=['post'])
    def resume(self, request, pk=None):
        run_obj = self.get_object()
        if run_obj.status not in [ProcessingRunStatus.PAUSED, ProcessingRunStatus.FAILED]:
            return Response({"message": f"Cannot resume run in state '{run_obj.status}'."}, status=status.HTTP_400_BAD_REQUEST)

        run_obj.status = ProcessingRunStatus.QUEUED
        run_obj.save(update_fields=['status', 'updated_at'])

        log_audit_event(
            action="processing_run.resumed",
            resource_type="processing_run",
            resource_id=str(run_obj.id),
            actor=request.user,
            request=request
        )

        try:
            from services.worker.tasks.processing_tasks import execute_processing_run
            execute_processing_run.delay(str(run_obj.id))
        except Exception:
            pass

        return Response({"id": str(run_obj.id), "status": run_obj.status, "message": "Processing run resumed."})

    @action(detail=True, methods=['get'])
    def stages(self, request, pk=None):
        run_obj = self.get_object()
        events = run_obj.stage_events.all()
        serializer = ProcessingStageEventSerializer(events, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def artifacts(self, request, pk=None):
        run_obj = self.get_object()
        out_dataset = run_obj.output_dataset
        manifest_data = None
        if out_dataset and hasattr(out_dataset, 'manifest'):
            manifest_data = DatasetManifestSerializer(out_dataset.manifest).data

        return Response({
            "processing_run_id": str(run_obj.id),
            "output_dataset_id": str(out_dataset.id) if out_dataset else None,
            "output_storage_uri": out_dataset.storage_uri if out_dataset else None,
            "manifest": manifest_data
        })
