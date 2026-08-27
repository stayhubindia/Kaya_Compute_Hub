import logging
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError

from apps.jobs.models import Job, JobStatusChoices, JobTypeChoices
from apps.jobs.serializers import JobSerializer, JobCreateSerializer
from apps.jobs.services import transition_job_status
from apps.accounts.permissions import IsAuthenticatedAdmin
from apps.audit.services import log_audit_event
from services.worker.tasks.job_tasks import execute_job
from apps.logs.views import sanitize_log_message

logger = logging.getLogger(__name__)


class JobViewSet(viewsets.ModelViewSet):
    serializer_class = JobSerializer
    permission_classes = [permissions.IsAuthenticated]
    filterset_fields = ['status', 'job_type', 'created_by']
    search_fields = ['name', 'description', 'idempotency_key']
    ordering_fields = ['created_at', 'priority', 'status']

    def get_queryset(self):
        user = self.request.user
        if not user or user.is_anonymous:
            return Job.objects.none()
        if getattr(user, 'is_staff', False) or getattr(user, 'is_superuser', False):
            return Job.objects.all()
        return Job.objects.filter(created_by=user)

    def destroy(self, request, *args, **kwargs):
        job = self.get_object()
        job_id = str(job.id)
        job.delete()
        log_audit_event(
            action="job.delete",
            resource_type="job",
            resource_id=job_id,
            actor=request.user,
            metadata={"job_id": job_id},
            request=request
        )
        return Response({"message": "Job deleted successfully", "id": job_id}, status=status.HTTP_200_OK)

    def _create_and_dispatch_job(self, request, job_type: str, default_name: str, payload_data: dict, selected_account=None) -> Response:
        idempotency_key = request.data.get('idempotency_key')
        if idempotency_key:
            existing_job = Job.objects.filter(created_by=request.user, idempotency_key=idempotency_key).first()
            if existing_job:
                return Response({
                    "id": str(existing_job.id),
                    "status": existing_job.status,
                    "message": "Job already created (idempotent response)"
                }, status=status.HTTP_200_OK)

        name = request.data.get('name', default_name)
        description = request.data.get('description', f"Automated {job_type} job")
        priority = request.data.get('priority', 0)

        job = Job.objects.create(
            created_by=request.user,
            name=name,
            description=description,
            job_type=job_type,
            status=JobStatusChoices.DRAFT,
            priority=priority,
            payload=payload_data,
            idempotency_key=idempotency_key,
            selected_google_account=selected_account,
        )

        log_audit_event(
            action=f"job.create.{job_type}",
            resource_type="job",
            resource_id=str(job.id),
            actor=request.user,
            metadata={"name": job.name, "job_type": job.job_type},
            request=request
        )

        job = transition_job_status(job, JobStatusChoices.QUEUED, actor=request.user, request=request)

        try:
            execute_job.delay(str(job.id))
        except Exception as exc:
            logger.error(f"Failed to dispatch job {job.id} to queue: {exc}")
            transition_job_status(job, JobStatusChoices.FAILED, error_code="BROKER_UNAVAILABLE", error_message="Task queue broker is unavailable.")
            return Response({
                "error": {
                    "status_code": 503,
                    "message": "Task queue broker is unavailable. Please try again later.",
                    "details": {"job_id": str(job.id), "broker_error": str(exc)}
                }
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        return Response({
            "id": str(job.id),
            "status": job.status,
            "message": f"{name} job accepted and queued."
        }, status=status.HTTP_201_CREATED)

    def create(self, request, *args, **kwargs):
        serializer = JobCreateSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        job_type = serializer.validated_data.get('job_type')
        return self._create_and_dispatch_job(
            request,
            job_type=job_type,
            default_name=serializer.validated_data.get('name', 'Job'),
            payload_data=serializer.validated_data.get('payload', {}),
            selected_account=serializer.validated_data.get('selected_google_account'),
        )

    @action(detail=False, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def ingest(self, request):
        """Submit document extraction and ingestion job."""
        collection_slug = request.data.get("collection_slug", "default_collection")
        input_path = request.data.get("input_path")
        if not input_path:
            raise ValidationError({"input_path": "input_path is required for ingestion job."})
        return self._create_and_dispatch_job(request, JobTypeChoices.INGESTION, f"Ingest: {collection_slug}", request.data)

    @action(detail=False, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def generate(self, request):
        """Submit instruction candidate synthesis job."""
        collection_slug = request.data.get("collection_slug", "default_collection")
        return self._create_and_dispatch_job(request, JobTypeChoices.GENERATION, f"Generate Candidates: {collection_slug}", request.data)

    @action(detail=False, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def qa(self, request):
        """Submit release quality audit job."""
        collection_slug = request.data.get("collection_slug", "default_collection")
        return self._create_and_dispatch_job(request, JobTypeChoices.QUALITY_AUDIT, f"QA Audit: {collection_slug}", request.data)

    @action(detail=False, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def release(self, request):
        """Submit dataset freeze and lock job."""
        collection_slug = request.data.get("collection_slug", "default_collection")
        return self._create_and_dispatch_job(request, JobTypeChoices.FREEZE_DATASET, f"Freeze Dataset: {collection_slug}", request.data)

    @action(detail=False, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def train(self, request):
        """Submit QLoRA model training orchestration job."""
        collection_slug = request.data.get("collection_slug", "default_collection")
        return self._create_and_dispatch_job(request, JobTypeChoices.TRAINING_QLORA, f"Train QLoRA: {collection_slug}", request.data)

    @action(detail=False, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def evaluate(self, request):
        """Submit model evaluation job."""
        collection_slug = request.data.get("collection_slug", "default_collection")
        return self._create_and_dispatch_job(request, JobTypeChoices.EVALUATION, f"Evaluate Model: {collection_slug}", request.data)

    @action(detail=False, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def sync(self, request):
        """Submit Google Drive artifact sync job."""
        collection_slug = request.data.get("collection_slug", "default_collection")
        return self._create_and_dispatch_job(request, JobTypeChoices.SYNC_DRIVE, f"Sync Drive: {collection_slug}", request.data)

    @action(detail=True, methods=['get'], permission_classes=[permissions.IsAuthenticated])
    def logs(self, request, pk=None):
        """Retrieve execution logs formatted for job detail terminal log viewer."""
        job = self.get_object()
        logs_data = []

        from apps.logs.models import JobLog
        persisted_logs = JobLog.objects.filter(job=job).order_by('timestamp')[:1000]
        if persisted_logs:
            logs_data = [{
                "id": str(log.id),
                "timestamp": log.timestamp.isoformat(),
                "level": log.level,
                "module": log.module,
                "message": sanitize_log_message(log.message),
            } for log in persisted_logs]
            return Response({"job_id": str(job.id), "count": len(logs_data), "logs": logs_data}, status=status.HTTP_200_OK)

        logs_data.append({
            "id": f"{job.id}-created",
            "timestamp": job.created_at.isoformat() if job.created_at else "",
            "level": "info",
            "module": "job_manager",
            "message": f"Job created: '{job.name}' (type: {job.job_type})"
        })

        if job.started_at:
            logs_data.append({
                "id": f"{job.id}-started",
                "timestamp": job.started_at.isoformat(),
                "level": "info",
                "module": "job_executor",
                "message": f"Execution started in stage: '{job.current_stage or 'initialization'}'"
            })

        if job.progress_message:
            logs_data.append({
                "id": f"{job.id}-progress",
                "timestamp": job.updated_at.isoformat() if job.updated_at else "",
                "level": "info",
                "module": "job_progress",
                "message": f"Progress ({job.progress_percentage}%): {job.progress_message}"
            })

        if job.payload and isinstance(job.payload, dict):
            arxiv_stats = job.payload.get("arxiv_stats", {})
            if arxiv_stats:
                logs_data.append({
                    "id": f"{job.id}-arxiv-stats",
                    "timestamp": job.updated_at.isoformat() if job.updated_at else "",
                    "level": "info",
                    "module": "arxiv_downloader",
                    "message": f"ArXiv Batch Stats — Total: {arxiv_stats.get('total', 0)} | Downloaded: {arxiv_stats.get('processed', 0)} | PDF: {arxiv_stats.get('pdf', 0)} | HTML: {arxiv_stats.get('html', 0)} | Failed: {arxiv_stats.get('failed', 0)}"
                })

            output_dir = job.payload.get("output_dir")
            if output_dir:
                logs_data.append({
                    "id": f"{job.id}-output-dir",
                    "timestamp": job.created_at.isoformat() if job.created_at else "",
                    "level": "info",
                    "module": "storage_manager",
                    "message": f"Configured output directory: '{output_dir}'"
                })

            execution_result = job.payload.get("execution_result", {})
            if execution_result:
                if execution_result.get("stdout"):
                    logs_data.append({
                        "id": f"{job.id}-colab-stdout",
                        "timestamp": job.updated_at.isoformat() if job.updated_at else "",
                        "level": "info",
                        "module": "colab_runtime",
                        "message": sanitize_log_message(execution_result["stdout"][-20000:]),
                    })
                if execution_result.get("stderr"):
                    logs_data.append({
                        "id": f"{job.id}-colab-stderr",
                        "timestamp": job.updated_at.isoformat() if job.updated_at else "",
                        "level": "warning",
                        "module": "colab_runtime",
                        "message": sanitize_log_message(execution_result["stderr"][-20000:]),
                    })

        if job.error_message:
            logs_data.append({
                "id": f"{job.id}-error",
                "timestamp": job.finished_at.isoformat() if job.finished_at else (job.updated_at.isoformat() if job.updated_at else ""),
                "level": "error",
                "module": "job_executor",
                "message": f"Job Failure [{job.error_code or 'UNKNOWN_ERROR'}]: {job.error_message}"
            })

        if job.status == JobStatusChoices.SUCCEEDED:
            logs_data.append({
                "id": f"{job.id}-finished",
                "timestamp": job.finished_at.isoformat() if job.finished_at else (job.updated_at.isoformat() if job.updated_at else ""),
                "level": "info",
                "module": "job_executor",
                "message": "Job finished successfully."
            })
        elif job.status == JobStatusChoices.CANCELLED:
            logs_data.append({
                "id": f"{job.id}-cancelled",
                "timestamp": job.updated_at.isoformat() if job.updated_at else "",
                "level": "warning",
                "module": "job_executor",
                "message": "Job was cancelled by user request."
            })

        return Response({"job_id": str(job.id), "count": len(logs_data), "logs": logs_data}, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def cancel(self, request, pk=None):
        job = self.get_object()
        if job.status == JobStatusChoices.CANCELLED:
            return Response({"id": str(job.id), "status": job.status, "message": "Job already cancelled"}, status=status.HTTP_200_OK)

        if job.status in [JobStatusChoices.DRAFT, JobStatusChoices.QUEUED, JobStatusChoices.LEASED, JobStatusChoices.RUNNING]:
            job = transition_job_status(job, JobStatusChoices.CANCELLED, actor=request.user, request=request)
            return Response({"id": str(job.id), "status": job.status, "message": "Job cancelled successfully"}, status=status.HTTP_200_OK)
        else:
            raise ValidationError({"status": f"Cannot cancel job in terminal status '{job.status}'."})

    @action(detail=True, methods=['post'], permission_classes=[permissions.IsAuthenticated])
    def retry(self, request, pk=None):
        job = self.get_object()
        if job.status != JobStatusChoices.FAILED:
            raise ValidationError({"status": f"Can only retry jobs in 'failed' status, current status is '{job.status}'."})

        if job.error_code == "NOT_IMPLEMENTED":
            raise ValidationError({"status": "Permanent validation or unsupported job type error cannot be retried."})

        if job.retry_count >= job.max_retries:
            raise ValidationError({"status": f"Job has reached maximum allowed retries ({job.max_retries})."})

        job = transition_job_status(job, JobStatusChoices.RETRYING, actor=request.user, request=request)
        job = transition_job_status(job, JobStatusChoices.QUEUED, actor=request.user, request=request)

        try:
            execute_job.delay(str(job.id))
        except Exception as exc:
            logger.error(f"Failed to re-dispatch job {job.id} to queue: {exc}")
            return Response({
                "error": {
                    "status_code": 503,
                    "message": "Task queue broker is unavailable. Please try again later."
                }
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        return Response({"id": str(job.id), "status": job.status, "message": "Job retried and queued"}, status=status.HTTP_200_OK)
