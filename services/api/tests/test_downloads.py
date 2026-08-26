import pytest
from unittest.mock import patch
from rest_framework.test import APIClient
from apps.accounts.models import User
from apps.downloads.models import Download, DownloadStatus
from apps.audit.models import AuditEvent

@pytest.mark.django_db
def test_create_download_job_success():
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    client = APIClient()
    client.force_authenticate(user=admin)

    with patch('services.downloader.tasks.download_tasks.process_download_job.delay') as mock_task:
        resp = client.post('/api/v1/downloads/', {
            'url': 'https://github.com/example/repo/releases/download/v1.0/dataset.zip',
            'checksum_algorithm': 'sha256',
            'extract': False
        }, format='json')

        assert resp.status_code == 201
        data = resp.json()
        assert data['status'] == 'queued'
        assert 'id' in data

        download_obj = Download.objects.get(id=data['id'])
        assert download_obj.created_by == admin
        assert download_obj.provider == 'github'
        mock_task.assert_called_once_with(str(download_obj.id))

    assert AuditEvent.objects.filter(action='download.requested', resource_id=str(download_obj.id)).exists()

@pytest.mark.django_db
def test_create_download_blocked_by_ssrf():
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    client = APIClient()
    client.force_authenticate(user=admin)

    resp = client.post('/api/v1/downloads/', {
        'url': 'http://127.0.0.1:8000/internal_file.txt'
    }, format='json')

    assert resp.status_code == 400
    data = resp.json()
    assert data['error']['code'] == 'DOWNLOAD_URL_BLOCKED'

@pytest.mark.django_db
def test_download_quota_concurrent_limit():
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    
    for i in range(5):
        Download.objects.create(
            created_by=admin,
            source_url=f'https://github.com/example/repo/file_{i}.zip',
            status=DownloadStatus.DOWNLOADING
        )

    client = APIClient()
    client.force_authenticate(user=admin)

    resp = client.post('/api/v1/downloads/', {
        'url': 'https://github.com/example/repo/file_6.zip'
    }, format='json')

    assert resp.status_code == 403
    assert resp.json()['error']['code'] == 'QUOTA_EXCEEDED'

@pytest.mark.django_db
def test_download_cancel_pause_resume():
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    download_obj = Download.objects.create(
        created_by=admin,
        source_url='https://github.com/example/repo/dataset.parquet',
        status=DownloadStatus.DOWNLOADING
    )

    client = APIClient()
    client.force_authenticate(user=admin)

    # Pause
    pause_resp = client.post(f'/api/v1/downloads/{download_obj.id}/pause/')
    assert pause_resp.status_code == 200
    download_obj.refresh_from_db()
    assert download_obj.status == DownloadStatus.PAUSED

    # Resume with task mock to avoid eager network execution
    with patch('services.downloader.tasks.download_tasks.process_download_job.delay') as mock_task:
        resume_resp = client.post(f'/api/v1/downloads/{download_obj.id}/resume/')
        assert resume_resp.status_code == 200
        download_obj.refresh_from_db()
        assert download_obj.status == DownloadStatus.QUEUED
        mock_task.assert_called_once_with(str(download_obj.id))

    # Cancel
    cancel_resp = client.post(f'/api/v1/downloads/{download_obj.id}/cancel/')
    assert cancel_resp.status_code == 200
    download_obj.refresh_from_db()
    assert download_obj.status == DownloadStatus.CANCELLED
