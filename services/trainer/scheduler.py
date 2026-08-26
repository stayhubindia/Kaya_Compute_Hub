import datetime
from typing import Optional, Tuple
from django.utils import timezone
from django.db import transaction
from apps.workers.models import Worker, WorkerStatusChoices

STALE_HEARTBEAT_THRESHOLD_SECONDS = 60

class SchedulingError(Exception):
    pass

class TrainerScheduler:
    @staticmethod
    def find_and_allocate_worker(requested_gpus: int = 0) -> Optional[Worker]:

        cutoff_time = timezone.now() - datetime.timedelta(seconds=STALE_HEARTBEAT_THRESHOLD_SECONDS)

        with transaction.atomic():
            candidates = Worker.objects.select_for_update().filter(
                status__in=[WorkerStatusChoices.IDLE, WorkerStatusChoices.BUSY],
                last_heartbeat_at__gte=cutoff_time
            ).order_by('-last_heartbeat_at')

            for worker in candidates:
                if requested_gpus > 0:
                    if worker.available_gpu_slots >= requested_gpus:
                        worker.available_gpu_slots -= requested_gpus
                        worker.allocated_gpu_slots += requested_gpus
                        if worker.status == WorkerStatusChoices.IDLE:
                            worker.status = WorkerStatusChoices.BUSY
                        worker.save(update_fields=['available_gpu_slots', 'allocated_gpu_slots', 'status', 'updated_at'])
                        return worker
                else:
                    # CPU Job: Any online worker with CPU capacity
                    if worker.cpu_count > 0:
                        return worker

        return None

    @staticmethod
    def release_worker_capacity(worker_id: str, allocated_gpus: int = 0):
        try:
            with transaction.atomic():
                worker = Worker.objects.select_for_update().get(id=worker_id)
                if allocated_gpus > 0:
                    worker.available_gpu_slots = min(
                        worker.gpu_count,
                        worker.available_gpu_slots + allocated_gpus
                    )
                    worker.allocated_gpu_slots = max(0, worker.allocated_gpu_slots - allocated_gpus)

                if worker.allocated_gpu_slots == 0 and worker.status == WorkerStatusChoices.BUSY:
                    worker.status = WorkerStatusChoices.IDLE

                worker.save(update_fields=['available_gpu_slots', 'allocated_gpu_slots', 'status', 'updated_at'])
        except Worker.DoesNotExist:
            pass
