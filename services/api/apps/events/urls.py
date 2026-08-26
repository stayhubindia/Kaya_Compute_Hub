from django.urls import path
from apps.events.views import SSEEventStreamView

urlpatterns = [
    path("events/stream/", SSEEventStreamView.as_view(), name="events-stream"),
]
