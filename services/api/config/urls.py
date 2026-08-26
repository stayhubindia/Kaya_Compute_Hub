from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions

from apps.accounts.views import (
    LoginView, LogoutView, CurrentUserView
)
from apps.jobs.views import JobViewSet
from apps.workers.views import WorkerViewSet
from apps.datasets.views import DatasetViewSet
from apps.artifacts.views import ArtifactViewSet
from apps.downloads.views import DownloadViewSet
from apps.pipelines.views import PipelineDefinitionViewSet, ProcessingRunViewSet
from apps.training.views import TrainingRunViewSet
from apps.models_registry.views import ModelVersionViewSet
from apps.audit.views import AuditEventListView

class HealthCheckView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        return Response({
            "status": "ok",
            "service": "kaya-compute-api"
        })

router = DefaultRouter()
router.register(r'jobs', JobViewSet, basename='job')
router.register(r'workers', WorkerViewSet, basename='worker')
router.register(r'datasets', DatasetViewSet, basename='dataset')
router.register(r'artifacts', ArtifactViewSet, basename='artifact')
router.register(r'downloads', DownloadViewSet, basename='download')
router.register(r'pipelines', PipelineDefinitionViewSet, basename='pipeline')
router.register(r'processing-runs', ProcessingRunViewSet, basename='processing-run')
router.register(r'training-runs', TrainingRunViewSet, basename='training-run')
router.register(r'models', ModelVersionViewSet, basename='model')

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # API v1 Versioned Authentication & System Endpoints
    path('api/v1/health/', HealthCheckView.as_view(), name='health-check'),
    path('api/v1/auth/login/', LoginView.as_view(), name='auth-login'),
    path('api/v1/auth/logout/', LogoutView.as_view(), name='auth-logout'),
    path('api/v1/auth/me/', CurrentUserView.as_view(), name='auth-me'),
    
    path('api/v1/audit-events/', AuditEventListView.as_view(), name='audit-events-list'),
    path('api/v1/', include(router.urls)),
    path('api/v1/', include('apps.events.urls')),
    path('api/v1/', include('apps.logs.urls')),
    path('api/v1/integrations/', include('apps.integrations.urls')),
]
