from unittest.mock import patch

import pytest
from rest_framework.test import APIClient

from apps.accounts.models import User
from apps.integrations.models import ConnectedAccount, AccountStatusChoices
from apps.jobs.models import Job


@pytest.mark.django_db
def test_colab_job_requires_owned_active_account():
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    client = APIClient()
    client.force_authenticate(user=admin)

    response = client.post('/api/v1/jobs/', {
        'name': 'Persistent Colab job',
        'job_type': 'custom_script',
        'payload': {'execution_target': 'colab', 'code': 'print("hello")'},
    }, format='json')

    assert response.status_code == 400
    assert 'selected_google_account_id' in response.json()['error']['details']


@pytest.mark.django_db
def test_ui_submission_persists_colab_routing_metadata():
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    account = ConnectedAccount.objects.create(
        user=admin,
        provider='google',
        provider_account_id='google-123',
        email='worker@gmail.com',
        status=AccountStatusChoices.ACTIVE,
    )
    account.set_access_token('test-access-token')
    account.save(update_fields=['encrypted_access_token'])
    client = APIClient()
    client.force_authenticate(user=admin)

    with patch('apps.jobs.views.execute_job.delay') as dispatch:
        response = client.post('/api/v1/jobs/', {
            'name': 'Persistent Colab job',
            'job_type': 'custom_script',
            'selected_google_account_id': str(account.id),
            'payload': {
                'execution_target': 'colab',
                'session_name': 'training-t4',
                'accelerator': 'T4',
                'code': 'print("hello from colab")',
            },
        }, format='json')

    assert response.status_code == 201
    job = Job.objects.get(id=response.json()['id'])
    assert job.selected_google_account == account
    assert job.payload['execution_target'] == 'colab'
    assert job.payload['session_name'] == 'training-t4'
    dispatch.assert_called_once_with(str(job.id))
