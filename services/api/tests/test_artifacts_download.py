import pytest
from django.urls import reverse
from rest_framework import status
from apps.accounts.models import User
from apps.artifacts.models import Artifact
from apps.audit.models import AuditEvent

@pytest.mark.django_db
def test_artifact_download_unauthenticated(client):
    admin = User.objects.create_admin(email="admin@example.com", password="Password123!")
    art = Artifact.objects.create(name="Dataset Model", artifact_type="model", storage_uri="/storage/models/model.bin", checksum="abc123sha", created_by=admin)

    url = reverse("artifact-download", kwargs={"pk": art.id})
    response = client.get(url)
    assert response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)

@pytest.mark.django_db
def test_artifact_download_authorized_and_audit(client):
    admin = User.objects.create_admin(email="admin@example.com", password="Password123!")
    art = Artifact.objects.create(name="Checkpoints Archive", artifact_type="checkpoint", storage_uri="/storage/checkpoints/ckpt.tar.gz", checksum="xyz789sha", created_by=admin)

    client.force_login(admin)
    url = reverse("artifact-download", kwargs={"pk": art.id})
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.data["name"] == "Checkpoints Archive"

    # Audit event logged
    assert AuditEvent.objects.filter(action="artifact.downloaded", resource_id=str(art.id)).exists()
