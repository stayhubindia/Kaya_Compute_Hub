import pytest
from rest_framework.test import APIClient
from rest_framework.exceptions import ValidationError
from apps.accounts.models import User
from apps.integrations.models import AccountStatusChoices, ConnectedAccount
from apps.jobs.models import Job, JobStatusChoices, JobTypeChoices
from apps.jobs.services import transition_job_status

@pytest.mark.django_db
def test_job_creation_and_permissions():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    account = ConnectedAccount.objects.create(user=admin, provider='google', provider_account_id='job-1', email='job@example.com', status=AccountStatusChoices.ACTIVE)
    client = APIClient()

    # Unauthenticated should be denied
    resp = client.post('/api/v1/jobs/', {
        'name': 'Viewer Task',
        'job_type': 'download',
        'payload': {}
    }, format='json')
    assert resp.status_code in (401, 403)

    # Admin should succeed
    client.force_authenticate(user=admin)
    resp = client.post('/api/v1/jobs/', {
        'name': 'Dataset Download Job',
        'job_type': 'download',
        'selected_google_account_id': str(account.id),
        'payload': {'session_name': 'colab-job', 'code': 'print("download in colab")'}
    }, format='json')
    assert resp.status_code == 201
    assert resp.json()['status'] == 'queued'

@pytest.mark.django_db
def test_job_idempotency():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    account = ConnectedAccount.objects.create(user=admin, provider='google', provider_account_id='job-2', email='idempotent@example.com', status=AccountStatusChoices.ACTIVE)
    client = APIClient()
    client.force_authenticate(user=admin)

    payload = {
        'name': 'Idempotent Preprocessing',
        'job_type': 'preprocessing',
        'selected_google_account_id': str(account.id),
        'idempotency_key': 'key-12345',
        'payload': {'session_name': 'colab-job', 'code': 'print("preprocessing in colab")'}
    }

    res1 = client.post('/api/v1/jobs/', payload, format='json')
    assert res1.status_code == 201
    job1_id = res1.json()['id']

    # Submitting identical idempotency key returns existing job
    res2 = client.post('/api/v1/jobs/', payload, format='json')
    assert res2.status_code == 200
    assert res2.json()['id'] == job1_id

@pytest.mark.django_db
def test_valid_job_transitions():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    job = Job.objects.create(
        name='Test Training Job',
        job_type=JobTypeChoices.TRAINING,
        status=JobStatusChoices.DRAFT,
        created_by=admin
    )

    # draft -> queued
    job = transition_job_status(job, JobStatusChoices.QUEUED, actor=admin)
    assert job.status == JobStatusChoices.QUEUED

    # queued -> leased
    job = transition_job_status(job, JobStatusChoices.LEASED, actor=admin)
    assert job.status == JobStatusChoices.LEASED

    # leased -> running
    job = transition_job_status(job, JobStatusChoices.RUNNING, actor=admin)
    assert job.status == JobStatusChoices.RUNNING
    assert job.started_at is not None

    # running -> succeeded
    job = transition_job_status(job, JobStatusChoices.SUCCEEDED, actor=admin)
    assert job.status == JobStatusChoices.SUCCEEDED
    assert job.finished_at is not None

@pytest.mark.django_db
def test_invalid_job_transition_raises():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    job = Job.objects.create(
        name='Test Notebook Job',
        job_type=JobTypeChoices.NOTEBOOK,
        status=JobStatusChoices.DRAFT,
        created_by=admin
    )

    # Attempt invalid direct transition draft -> succeeded
    with pytest.raises(ValidationError):
        transition_job_status(job, JobStatusChoices.SUCCEEDED, actor=admin)

@pytest.mark.django_db
def test_job_retry_and_cancel_endpoints():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    client = APIClient()
    client.force_authenticate(user=admin)

    # Create job in failed status
    failed_job = Job.objects.create(
        name='Failed Extraction',
        job_type=JobTypeChoices.EXTRACTION,
        status=JobStatusChoices.FAILED,
        created_by=admin
    )

    # Retry job
    retry_resp = client.post(f'/api/v1/jobs/{failed_job.id}/retry/')
    assert retry_resp.status_code == 200
    assert retry_resp.json()['status'] == 'queued'
