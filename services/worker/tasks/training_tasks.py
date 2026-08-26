import os
import hashlib
import json
from django.utils import timezone
from celery import shared_task

from apps.training.models import TrainingRun, TrainingRunStatus
from apps.models_registry.models import ModelVersion, ModelStatusChoices
from apps.artifacts.models import Artifact, ArtifactTypeChoices
from apps.audit.services import log_audit_event
from services.trainer import get_trainer_backend, TrainerScheduler, validate_training_configuration
from services.trainer.checkpoints import CheckpointRecoveryManager
from services.trainer.containers.image_registry import get_image_metadata

def _calculate_file_checksum(filepath: str) -> str:
    hasher = hashlib.sha256()
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
    return hasher.hexdigest()

@shared_task(bind=True, max_retries=1, default_retry_delay=5)
def execute_training_run(self, training_run_id: str):
    try:
        run_obj = TrainingRun.objects.get(id=training_run_id)
    except TrainingRun.DoesNotExist:
        return {"status": "error", "message": f"TrainingRun {training_run_id} not found."}

    if run_obj.status == TrainingRunStatus.CANCELLED:
        return {"status": "cancelled", "message": "Training run was cancelled before execution."}

    # 1. Acquire Worker Capacity (GPU or CPU)
    requested_gpus = run_obj.resource_policy.get("requested_gpus", 0)
    worker = TrainerScheduler.find_and_allocate_worker(requested_gpus=requested_gpus)

    if not worker and requested_gpus > 0:
        # GPU capacity not currently available. Keep in queued state.
        return {"status": "queued", "message": "Queued waiting for GPU worker capacity."}

    allocated_worker_id = str(worker.id) if worker else None

    try:
        # 2. Update Status to Scheduled -> Preparing -> Running
        run_obj.status = TrainingRunStatus.RUNNING
        run_obj.started_at = timezone.now() if not run_obj.started_at else run_obj.started_at
        img_meta = get_image_metadata(run_obj.container_image)
        if img_meta:
            run_obj.container_digest = img_meta.get("digest", "")
        run_obj.save(update_fields=['status', 'started_at', 'container_digest', 'updated_at'])

        log_audit_event(
            action="training.started",
            resource_type="training_run",
            resource_id=str(run_obj.id),
            metadata={"worker_id": allocated_worker_id, "gpus": requested_gpus}
        )

        # 3. Check for Checkpoint Resume
        resume_from = CheckpointRecoveryManager.get_latest_valid_checkpoint(str(run_obj.id))

        # 4. Prepare Output Storage
        output_dir = os.path.join("storage/models", str(run_obj.id))
        os.makedirs(output_dir, exist_ok=True)

        def progress_cb(epoch: int, step: int, metrics_dict: dict):
            pct = round((epoch / max(1, run_obj.configuration.get("epochs", 1))) * 100, 2)
            TrainingRun.objects.filter(id=run_obj.id).update(
                current_epoch=epoch,
                current_step=step,
                progress_percent=pct,
                updated_at=timezone.now()
            )

        def cancel_check() -> bool:
            current_status = TrainingRun.objects.values_list('status', flat=True).get(id=run_obj.id)
            return current_status in [TrainingRunStatus.CANCELLED, TrainingRunStatus.CANCELLING, TrainingRunStatus.PAUSED]

        # 5. Execute Approved Backend
        backend_engine = get_trainer_backend(run_obj.backend)
        model_path, summary_metrics = backend_engine.train(
            run_id=str(run_obj.id),
            dataset_uri=run_obj.dataset.storage_uri,
            output_dir=output_dir,
            configuration=run_obj.configuration,
            resume_from=resume_from,
            progress_callback=progress_cb,
            cancel_check=cancel_check
        )

        # 6. Calculate Artifact Checksum & Create Artifact Record
        checksum = _calculate_file_checksum(model_path)
        file_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0

        artifact_obj = Artifact.objects.create(
            name=f"Model Artifact - {run_obj.name}",
            artifact_type=ArtifactTypeChoices.MODEL,
            storage_uri=model_path,
            size_bytes=file_size,
            checksum=checksum,
            metadata={"training_run_id": str(run_obj.id), "metrics": summary_metrics},
            created_by=run_obj.created_by
        )

        # 7. Register Model Version in Model Registry
        model_version_name = run_obj.name.lower().replace(" ", "-")
        version_str = f"v1.0.{run_obj.current_epoch or 1}"

        # Handle potential version collision
        if ModelVersion.objects.filter(name=model_version_name, version=version_str).exists():
            version_str = f"v1.0.{run_obj.current_epoch or 1}-{str(run_obj.id)[:4]}"

        model_version_obj, created = ModelVersion.objects.get_or_create(
            training_run=run_obj,
            defaults={
                "name": model_version_name,
                "version": version_str,
                "artifact": artifact_obj,
                "framework": run_obj.backend,
                "framework_version": "1.0.0",
                "model_format": "bin",
                "checksum": checksum,
                "metadata": {
                    "training_run_id": str(run_obj.id),
                    "dataset_id": str(run_obj.dataset.id),
                    "dataset_uri": run_obj.dataset.storage_uri,
                    "container_digest": run_obj.container_digest,
                    "configuration": run_obj.configuration,
                    "metrics": summary_metrics,
                },
                "status": ModelStatusChoices.REGISTERED,
                "created_by": run_obj.created_by
            }
        )

        log_audit_event(
            action="model.registered",
            resource_type="model_version",
            resource_id=str(model_version_obj.id),
            metadata={"name": model_version_name, "version": version_str, "checksum": checksum}
        )

        # 8. Mark Training Run Succeeded
        run_obj.status = TrainingRunStatus.SUCCEEDED
        run_obj.output_model_uri = model_path
        run_obj.progress_percent = 100.0
        run_obj.finished_at = timezone.now()
        run_obj.save(update_fields=['status', 'output_model_uri', 'progress_percent', 'finished_at', 'updated_at'])

        log_audit_event(
            action="training.succeeded",
            resource_type="training_run",
            resource_id=str(run_obj.id),
            metadata={"model_path": model_path, "metrics": summary_metrics}
        )

        return {"status": "succeeded", "run_id": str(run_obj.id), "model_version_id": str(model_version_obj.id)}

    except InterruptedError:
        run_obj.status = TrainingRunStatus.CANCELLED
        run_obj.save(update_fields=['status', 'updated_at'])
        return {"status": "cancelled", "run_id": str(run_obj.id)}

    except Exception as e:
        run_obj.status = TrainingRunStatus.FAILED
        run_obj.error_code = "TRAINING_EXECUTION_ERROR"
        run_obj.error_message = str(e)
        run_obj.save(update_fields=['status', 'error_code', 'error_message', 'updated_at'])

        log_audit_event(
            action="training.failed",
            resource_type="training_run",
            resource_id=str(run_obj.id),
            metadata={"error": str(e)}
        )
        return {"status": "failed", "reason": str(e)}

    finally:
        # 9. Always Release Worker Capacity
        if allocated_worker_id:
            TrainerScheduler.release_worker_capacity(allocated_worker_id, allocated_gpus=requested_gpus)
