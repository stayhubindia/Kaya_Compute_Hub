from django.urls import path
from apps.logs.views import JobLogsView

urlpatterns = [
    path("jobs/<uuid:job_id>/logs/", JobLogsView.as_view(), name="job-logs"),
]
