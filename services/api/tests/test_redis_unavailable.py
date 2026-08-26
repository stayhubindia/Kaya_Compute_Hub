from unittest.mock import patch
import pytest
from rest_framework.test import APIClient
from apps.accounts.models import User

@pytest.mark.django_db
def test_redis_unavailable_returns_503():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    client = APIClient()
    client.force_authenticate(user=admin)

    # Patch execute_job.delay to simulate Redis connection failure
    with patch('services.worker.tasks.job_tasks.execute_job.delay', side_effect=Exception("Redis connection error")):
        response = client.post('/api/v1/jobs/', {
            'name': 'Test Job Broker Fail',
            'job_type': 'download',
            'payload': {}
        }, format='json')

        assert response.status_code == 503
        data = response.json()
        assert data['error']['status_code'] == 503
        assert 'unavailable' in data['error']['message'].lower()
