import pytest
from rest_framework.test import APIClient
from apps.accounts.models import User
from apps.datasets.models import Dataset, DatasetStatusChoices

@pytest.mark.django_db
def test_dataset_creation_and_listing():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')

    client = APIClient()

    # Unauthenticated should be denied
    denied_resp = client.post('/api/v1/datasets/', {
        'name': 'Sample Dataset',
        'storage_uri': 'storage/datasets/sample.parquet',
        'format': 'parquet'
    }, format='json')
    assert denied_resp.status_code in (401, 403)

    # Admin should succeed
    client.force_authenticate(user=admin)
    resp = client.post('/api/v1/datasets/', {
        'name': 'ImageNet Subset',
        'description': 'Sample images for evaluation',
        'source_url': 'https://example.com/imagenet.tar.gz',
        'storage_uri': 'storage/datasets/imagenet_subset',
        'format': 'tar.gz',
        'size_bytes': 104857600
    }, format='json')
    assert resp.status_code == 201
    assert resp.json()['status'] == 'pending'
