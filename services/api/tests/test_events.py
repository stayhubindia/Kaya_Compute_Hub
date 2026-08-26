import pytest
from django.urls import reverse
from rest_framework import status
from apps.events.models import SystemEvent
from apps.accounts.models import User
from services.worker.events.publisher import EventPublisher
from services.worker.events.schemas import format_event, sanitize_event_payload

@pytest.mark.django_db
def test_format_and_sanitize_event():
    payload = {"password": "secretpassword123", "normal": "value"}
    sanitized = sanitize_event_payload(payload)
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["normal"] == "value"

    formatted = format_event("job.progress", payload={"progress": 50})
    assert formatted["event_type"] == "job.progress"
    assert formatted["payload"]["progress"] == 50

@pytest.mark.django_db
def test_event_publisher():
    evt = EventPublisher.publish_event("job.started", payload={"step": 1})
    assert evt["event_type"] == "job.started"
    assert SystemEvent.objects.filter(id=evt["event_id"]).exists()

@pytest.mark.django_db
def test_sse_stream_unauthenticated(client):
    url = reverse("events-stream")
    response = client.get(url)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.django_db
def test_sse_stream_authenticated(client):
    admin = User.objects.create_admin(email="events_user@example.com", password="Password123!")
    client.force_login(admin)

    evt = EventPublisher.publish_event("job.succeeded", payload={"status": "done"}, user_id=str(admin.id))

    url = reverse("events-stream")
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response["Content-Type"] == "text/event-stream"
