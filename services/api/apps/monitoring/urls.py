from django.urls import path
from apps.monitoring.views import WorkerListView, WorkerMetricsView

urlpatterns = [
    path("workers/", WorkerListView.as_view(), name="monitoring-worker-list"),
    path("workers/<uuid:worker_id>/metrics/", WorkerMetricsView.as_view(), name="monitoring-worker-metrics"),
]
