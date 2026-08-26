from rest_framework import viewsets, permissions, status
from apps.datasets.models import Dataset
from apps.datasets.serializers import DatasetSerializer
from apps.accounts.permissions import IsAuthenticatedAdmin
from apps.audit.services import log_audit_event

class DatasetViewSet(viewsets.ModelViewSet):
    serializer_class = DatasetSerializer
    permission_classes = [IsAuthenticatedAdmin]
    filterset_fields = ['status', 'format', 'created_by']
    search_fields = ['name', 'description', 'source_url']
    ordering_fields = ['created_at', 'size_bytes']

    def get_queryset(self):
        return Dataset.objects.all()

    def perform_create(self, serializer):
        dataset = serializer.save(created_by=self.request.user)
        log_audit_event(
            action="dataset.create",
            resource_type="dataset",
            resource_id=str(dataset.id),
            actor=self.request.user,
            metadata={"name": dataset.name, "storage_uri": dataset.storage_uri},
            request=self.request
        )
