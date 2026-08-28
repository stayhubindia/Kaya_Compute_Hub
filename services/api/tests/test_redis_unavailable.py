from unittest.mock import patch
import pytest
from rest_framework.test import APIClient
from apps.accounts.models import User
from apps.integrations.models import AccountStatusChoices, ConnectedAccount

@pytest.mark.django_db
def test_redis_unavailable_returns_503():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    account = ConnectedAccount.objects.create(user=admin, provider='google', provider_account_id='redis-1', email='redis@example.com', status=AccountStatusChoices.ACTIVE)
    client = APIClient()
    client.force_authenticate(user=admin)

    # Patch execute_job.delay to simulate Redis connection failure
    with patch('services.worker.tasks.job_tasks.execute_job.delay', side_effect=Exception("Redis connection error")):
        response = client.post('/api/v1/jobs/', {
            'name': 'Test Job Broker Fail',
            'job_type': 'download',
            'selected_google_account_id': str(account.id),
            'payload': {'session_name': 'colab-job', 'code': 'print("broker test")'}
        }, format='json')

        assert response.status_code == 503
        data = response.json()
        assert data['error']['status_code'] == 503
        assert 'unavailable' in data['error']['message'].lower()
