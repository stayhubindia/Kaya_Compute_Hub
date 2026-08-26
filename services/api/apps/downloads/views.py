from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.accounts.permissions import IsAuthenticatedAdmin
from apps.downloads.models import Download, DownloadStatus
from apps.downloads.serializers import DownloadSerializer, DownloadCreateSerializer
from apps.downloads.quota import check_user_download_quota, QuotaExceededError
from apps.audit.services import log_audit_event
from services.downloader.providers import get_provider_for_url
from services.downloader.security import validate_download_url, SSRFError

class DownloadViewSet(viewsets.ModelViewSet):
    serializer_class = DownloadSerializer
    permission_classes = [IsAuthenticatedAdmin]

    def get_queryset(self):
        return Download.objects.all()

    def create(self, request, *args, **kwargs):
        serializer = DownloadCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        url = serializer.validated_data['url']
        expected_checksum = serializer.validated_data.get('expected_checksum')
        checksum_algorithm = serializer.validated_data.get('checksum_algorithm', 'sha256')
        extract = serializer.validated_data.get('extract', False)

        # 1. SSRF & URL Validation
        try:
            validate_download_url(url)
            provider = get_provider_for_url(url)
            meta = provider.get_metadata(url)
        except SSRFError as e:
            log_audit_event(
                action="download.url_rejected",
                resource_type="download",
                resource_id=url[:100],
                actor=request.user,
                metadata={"reason": str(e)},
                request=request
            )
            return Response({
                "error": {
                    "code": "DOWNLOAD_URL_BLOCKED",
                    "message": str(e)
                }
            }, status=status.HTTP_400_BAD_REQUEST)

        # 2. Check User Download Quota
        try:
            check_user_download_quota(request.user, meta.expected_size_bytes or 0)
        except QuotaExceededError as e:
            log_audit_event(
                action="download.quota_rejected",
                resource_type="download",
                resource_id=url[:100],
                actor=request.user,
                metadata={"reason": str(e)},
                request=request
            )
            return Response({
                "error": {
                    "code": "QUOTA_EXCEEDED",
                    "message": str(e)
                }
            }, status=status.HTTP_403_FORBIDDEN)

        # 3. Create Download Record
        download_obj = Download.objects.create(
            created_by=request.user,
            source_url=url,
            provider=meta.provider_name,
            original_filename=meta.filename or "download.bin",
            content_type=meta.content_type,
            expected_size_bytes=meta.expected_size_bytes,
            checksum_algorithm=checksum_algorithm,
            expected_checksum=expected_checksum,
            extract=extract,
            status=DownloadStatus.QUEUED
        )

        log_audit_event(
            action="download.requested",
            resource_type="download",
            resource_id=str(download_obj.id),
            actor=request.user,
            metadata={"provider": meta.provider_name, "url": url},
            request=request
        )

        # 4. Enqueue Celery Task
        try:
            from services.downloader.tasks.download_tasks import process_download_job
            process_download_job.delay(str(download_obj.id))
        except Exception:
            pass

        return Response({
            "id": str(download_obj.id),
            "status": download_obj.status,
            "message": "Download accepted and enqueued."
        }, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        download_obj = self.get_object()
        if download_obj.status in [DownloadStatus.COMPLETED, DownloadStatus.FAILED, DownloadStatus.CANCELLED]:
            return Response({"message": f"Download is already in terminal state '{download_obj.status}'."}, status=status.HTTP_400_BAD_REQUEST)

        download_obj.status = DownloadStatus.CANCELLED
        download_obj.save(update_fields=['status', 'updated_at'])

        log_audit_event(
            action="download.cancelled",
            resource_type="download",
            resource_id=str(download_obj.id),
            actor=request.user,
            request=request
        )

        return Response({"id": str(download_obj.id), "status": download_obj.status, "message": "Download cancelled successfully."})

    @action(detail=True, methods=['post'])
    def pause(self, request, pk=None):
        download_obj = self.get_object()
        if download_obj.status not in [DownloadStatus.DOWNLOADING, DownloadStatus.QUEUED, DownloadStatus.RESOLVING]:
            return Response({"message": f"Cannot pause download in state '{download_obj.status}'."}, status=status.HTTP_400_BAD_REQUEST)

        download_obj.status = DownloadStatus.PAUSED
        download_obj.save(update_fields=['status', 'updated_at'])

        log_audit_event(
            action="download.paused",
            resource_type="download",
            resource_id=str(download_obj.id),
            actor=request.user,
            request=request
        )

        return Response({"id": str(download_obj.id), "status": download_obj.status, "message": "Download paused."})

    @action(detail=True, methods=['post'])
    def resume(self, request, pk=None):
        download_obj = self.get_object()
        if download_obj.status not in [DownloadStatus.PAUSED, DownloadStatus.FAILED]:
            return Response({"message": f"Cannot resume download in state '{download_obj.status}'."}, status=status.HTTP_400_BAD_REQUEST)

        download_obj.status = DownloadStatus.QUEUED
        download_obj.save(update_fields=['status', 'updated_at'])

        log_audit_event(
            action="download.resumed",
            resource_type="download",
            resource_id=str(download_obj.id),
            actor=request.user,
            request=request
        )

        try:
            from services.downloader.tasks.download_tasks import process_download_job
            process_download_job.delay(str(download_obj.id))
        except Exception:
            pass

        return Response({"id": str(download_obj.id), "status": download_obj.status, "message": "Download resumed and re-enqueued."})

    @action(detail=True, methods=['post'])
    def verify(self, request, pk=None):
        download_obj = self.get_object()
        if download_obj.status != DownloadStatus.COMPLETED:
            return Response({"message": "Only completed downloads can be verified."}, status=status.HTTP_400_BAD_REQUEST)

        return Response({
            "id": str(download_obj.id),
            "status": download_obj.status,
            "actual_checksum": download_obj.actual_checksum,
            "expected_checksum": download_obj.expected_checksum,
            "verified": download_obj.actual_checksum == download_obj.expected_checksum if download_obj.expected_checksum else True
        })


# ─── ArXiv Batch Download API ───────────────────────────────────────────────

from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def arxiv_batch_start(request):
    """
    Start an ArXiv batch paper download job.

    POST body:
      category  - ArXiv category e.g. "cs.AI", "astro-ph", "quant-ph"  (required)
      month     - Month string e.g. "2025-01"  (required)
      workers   - Parallel threads 1-6  (default: 4)
      delay     - Inter-paper delay seconds  (default: 1.0)
    """
    category   = request.data.get("category", "").strip()
    month      = request.data.get("month", "").strip()
    workers    = int(request.data.get("workers", 4))
    delay      = float(request.data.get("delay", 1.0))
    output_dir = request.data.get("output_dir", "").strip()

    if not category:
        return Response({"error": {"message": "category is required."}}, status=status.HTTP_400_BAD_REQUEST)
    if not month:
        return Response({"error": {"message": "month is required (e.g. 2025-01)."}}, status=status.HTTP_400_BAD_REQUEST)

    from apps.jobs.models import Job, JobTypeChoices, JobStatusChoices
    job = Job.objects.create(
        name=f"ArXiv Download: {category} ({month})",
        created_by=request.user,
        job_type=JobTypeChoices.DOWNLOAD,
        status=JobStatusChoices.QUEUED,
        payload={
            "source": "arxiv_batch",
            "category": category,
            "month": month,
            "workers": workers,
            "delay": delay,
            "output_dir": output_dir,
        }
    )

    log_audit_event(
        action="arxiv_batch.started",
        resource_type="job",
        resource_id=str(job.id),
        actor=request.user,
        metadata={"category": category, "month": month},
        request=request
    )

    try:
        from services.downloader.tasks.arxiv_tasks import arxiv_batch_download_task
        try:
            arxiv_batch_download_task.delay(
                job_id=str(job.id),
                category=category,
                month=month,
                workers=max(1, min(workers, 6)),
                delay=max(0.0, delay),
                output_dir=output_dir,
            )
        except Exception:
            # Fallback to daemon background thread if Celery broker is unavailable
            import threading
            t = threading.Thread(
                target=arxiv_batch_download_task,
                args=(str(job.id), category, month, max(1, min(workers, 6)), max(0.0, delay), output_dir),
                daemon=True
            )
            t.start()
    except Exception as exc:
        job.payload["task_error"] = str(exc)
        job.save(update_fields=["payload"])

    return Response({
        "job_id": str(job.id),
        "status": "queued",
        "category": category,
        "month": month,
        "workers": workers,
        "message": f"ArXiv batch download job queued: {category} / {month}"
    }, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def arxiv_batch_status(request, job_id: str):
    """Get progress stats for a running ArXiv batch job."""
    from apps.jobs.models import Job
    try:
        job = Job.objects.get(id=job_id, created_by=request.user)
    except Job.DoesNotExist:
        return Response({"error": {"message": "Job not found."}}, status=status.HTTP_404_NOT_FOUND)

    arxiv_stats = (job.payload or {}).get("arxiv_stats", {})
    return Response({
        "job_id": str(job.id),
        "status": job.status,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "category": (job.payload or {}).get("category"),
        "month": (job.payload or {}).get("month"),
        "arxiv_stats": arxiv_stats,
        "output_dir": arxiv_stats.get("output_dir", ""),
    })
