import pytest
from django.urls import reverse
from rest_framework import status
from apps.accounts.models import User
from apps.jobs.models import Job, JobTypeChoices
from apps.logs.models import JobLog, LogLevelChoices
from apps.logs.views import sanitize_log_message

def test_sanitize_log_message():
    raw = "User password=secret123 logged in with <script>alert(1)</script>"
    clean = sanitize_log_message(raw)
    assert "[REDACTED]" in clean
    assert "<script>" not in clean
    assert "&lt;script&gt;" in clean

@pytest.mark.django_db
def test_job_logs_api(client):
    admin = User.objects.create_admin(email="loguser@example.com", password="Password123!")
    client.force_login(admin)

    job = Job.objects.create(name="Log Job", job_type=JobTypeChoices.PREPROCESSING, created_by=admin)
    JobLog.objects.create(job=job, level=LogLevelChoices.INFO, message="Starting task", module="worker")
    JobLog.objects.create(job=job, level=LogLevelChoices.ERROR, message="Task error password=xyz", module="worker")

    url = reverse("job-logs", kwargs={"job_id": job.id})
    response = client.get(url)
    assert response.status_code == status.HTTP_200_OK
    assert response.data["count"] == 2
    assert "[REDACTED]" in response.data["logs"][1]["message"]
