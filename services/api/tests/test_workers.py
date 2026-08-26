import pytest
from rest_framework.test import APIClient
from apps.accounts.models import User
from apps.workers.models import Worker, WorkerStatusChoices

@pytest.mark.django_db
def test_worker_list_and_heartbeat():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    worker = Worker.objects.create(
        name='worker-node-1',
        hostname='vm-worker-01.kaya.local',
        cpu_count=8,
        memory_bytes=16 * 1024 * 1024 * 1024,
        gpu_count=1,
        status=WorkerStatusChoices.OFFLINE
    )

    client = APIClient()
    client.force_authenticate(user=admin)

    # List workers
    resp = client.get('/api/v1/workers/')
    assert resp.status_code == 200
    assert len(resp.json()['results']) == 1

    # Heartbeat
    hb_resp = client.post(f'/api/v1/workers/{worker.id}/heartbeat/', {
        'status': 'idle',
        'capabilities': {'docker': True, 'cuda': '12.1'}
    }, format='json')
    assert hb_resp.status_code == 200
    updated_data = hb_resp.json()
    assert updated_data['status'] == 'idle'
    assert updated_data['last_heartbeat_at'] is not None
    assert updated_data['capabilities']['docker'] is True
