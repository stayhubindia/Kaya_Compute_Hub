from datetime import timedelta
import pytest
from django.utils import timezone
from rest_framework.test import APIClient
from apps.accounts.models import User
from apps.jobs.models import Job, JobStatusChoices, JobTypeChoices
from apps.jobs.services import claim_job_atomically
from apps.workers.models import Worker, WorkerStatusChoices
from services.worker.tasks.heartbeat_tasks import worker_heartbeat_task, mark_stale_workers_task
from services.worker.tasks.job_tasks import execute_job

@pytest.mark.django_db
def test_worker_registration_and_heartbeat():
    res = worker_heartbeat_task(worker_name="worker-test-01", status="idle")
    assert res['worker'] == "worker-test-01"

    worker = Worker.objects.get(name="worker-test-01")
    assert worker.status == "idle"
    assert worker.last_heartbeat_at is not None

@pytest.mark.django_db
def test_stale_worker_detection():
    old_time = timezone.now() - timedelta(seconds=120)
    worker = Worker.objects.create(
        name="stale-worker",
        hostname="localhost",
        status=WorkerStatusChoices.IDLE,
        last_heartbeat_at=old_time
    )

    res = mark_stale_workers_task(stale_seconds=60)
    assert res['stale_workers_marked_offline'] == 1

    worker.refresh_from_db()
    assert worker.status == WorkerStatusChoices.OFFLINE

@pytest.mark.django_db
def test_atomic_job_claiming_and_duplicate_prevention():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    job = Job.objects.create(
        name="Queued Demo Job",
        job_type=JobTypeChoices.DOWNLOAD,
        status=JobStatusChoices.QUEUED,
        created_by=admin
    )

    # First claim -> succeeds
    claimed_job, success1 = claim_job_atomically(str(job.id), worker_name="w1")
    assert success1 is True
    assert claimed_job.status == JobStatusChoices.LEASED

    # Second claim -> fails
    claimed_job2, success2 = claim_job_atomically(str(job.id), worker_name="w2")
    assert success2 is False

@pytest.mark.django_db
def test_execute_job_demo_download_success():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    job = Job.objects.create(
        name="Download Demo Job",
        job_type=JobTypeChoices.DOWNLOAD,
        status=JobStatusChoices.QUEUED,
        created_by=admin
    )

    result = execute_job(str(job.id))
    assert result['status'] == 'succeeded'

    job.refresh_from_db()
    assert job.status == JobStatusChoices.SUCCEEDED
    assert job.progress_percentage == 100

@pytest.mark.django_db
def test_execute_job_unsupported_notebook_fails():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    job = Job.objects.create(
        name="Notebook Execution Job",
        job_type=JobTypeChoices.NOTEBOOK,
        status=JobStatusChoices.QUEUED,
        created_by=admin
    )

    result = execute_job(str(job.id))
    assert result['status'] == 'failed'

    job.refresh_from_db()
    assert job.status == JobStatusChoices.FAILED
    assert job.error_code == 'NOT_IMPLEMENTED'

@pytest.mark.django_db
def test_job_cancellation():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')

    job = Job.objects.create(
        name="Job To Cancel",
        job_type=JobTypeChoices.EXTRACTION,
        status=JobStatusChoices.QUEUED,
        created_by=admin
    )

    client = APIClient()

    # Unauthenticated cancel -> 401/403
    resp_unauth = client.post(f'/api/v1/jobs/{job.id}/cancel/')
    assert resp_unauth.status_code in (401, 403)

    # Admin can cancel job
    client.force_authenticate(user=admin)
    resp_cancel = client.post(f'/api/v1/jobs/{job.id}/cancel/')
    assert resp_cancel.status_code == 200

    job.refresh_from_db()
    assert job.status == JobStatusChoices.CANCELLED
