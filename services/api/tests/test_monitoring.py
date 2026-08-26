import pytest
from django.urls import reverse
from django.utils import timezone
from datetime import timedelta
from rest_framework import status
from apps.accounts.models import User
from apps.workers.models import Worker, WorkerStatusChoices

@pytest.mark.django_db
def test_worker_monitoring_api(client):
    admin = User.objects.create_admin(email="monitor_user@example.com", password="Password123!")
    client.force_login(admin)

    w1 = Worker.objects.create(
        name="Worker 1",
        hostname="node1.internal.local",
        status=WorkerStatusChoices.IDLE,
        last_heartbeat_at=timezone.now(),
        cpu_count=8,
        memory_bytes=16*1024*1024*1024,
        gpu_count=2,
        gpu_model="NVIDIA T4",
        available_gpu_slots=2
    )

    url = reverse("worker-list")
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert len(response.data["results"]) == 1
    assert response.data["results"][0]["hostname_label"] == "node1"
    assert response.data["results"][0]["is_stale"] is False

@pytest.mark.django_db
def test_stale_worker_detection(client):
    admin = User.objects.create_admin(email="stale_user@example.com", password="Password123!")
    client.force_login(admin)

    w1 = Worker.objects.create(
        name="Stale Worker",
        hostname="stale.local",
        status=WorkerStatusChoices.IDLE,
        last_heartbeat_at=timezone.now() - timedelta(seconds=120)
    )

    url = reverse("worker-list")
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.data["results"][0]["is_stale"] is True
