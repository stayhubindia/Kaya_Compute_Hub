import pytest
from rest_framework.test import APIClient
from apps.accounts.models import User
from apps.artifacts.models import Artifact, ArtifactTypeChoices
from apps.jobs.models import Job, JobTypeChoices

@pytest.mark.django_db
def test_artifact_visibility_and_filtering():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')

    job = Job.objects.create(name='Training Run', job_type=JobTypeChoices.TRAINING, created_by=admin)
    artifact = Artifact.objects.create(
        name='model_checkpoint_epoch10.pt',
        artifact_type=ArtifactTypeChoices.CHECKPOINT,
        storage_uri='storage/checkpoints/model_epoch10.pt',
        size_bytes=524288000,
        job=job,
        created_by=admin
    )

    client = APIClient()

    # Unauthenticated rejected
    unauth_resp = client.get('/api/v1/artifacts/')
    assert unauth_resp.status_code in (401, 403)

    client.force_authenticate(user=admin)
    resp = client.get('/api/v1/artifacts/')
    assert resp.status_code == 200
    assert len(resp.json()['results']) == 1
    assert resp.json()['results'][0]['name'] == 'model_checkpoint_epoch10.pt'

    # Filter by type
    filter_resp = client.get('/api/v1/artifacts/?artifact_type=checkpoint')
    assert filter_resp.status_code == 200
    assert len(filter_resp.json()['results']) == 1
