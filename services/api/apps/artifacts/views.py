import os
from django.http import FileResponse
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from apps.artifacts.models import Artifact
from apps.artifacts.serializers import ArtifactSerializer
from apps.accounts.permissions import IsAuthenticatedAdmin
from apps.audit.services import log_audit_event

class ArtifactViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = ArtifactSerializer
    permission_classes = [IsAuthenticatedAdmin]
    filterset_fields = ['artifact_type', 'job', 'created_by']
    search_fields = ['name', 'storage_uri', 'checksum']
    ordering_fields = ['created_at', 'size_bytes']

    def get_queryset(self):
        return Artifact.objects.all()

    @action(detail=True, methods=['get'])
    def download(self, request, pk=None):
        artifact = self.get_object()
        filepath = artifact.storage_uri

        if not filepath or ".." in filepath or not os.path.exists(filepath):
            log_audit_event(
                action="artifact.downloaded",
                resource_type="artifact",
                resource_id=str(artifact.id),
                actor=request.user,
                metadata={"name": artifact.name, "simulated": True},
                request=request
            )
            return Response({
                "id": str(artifact.id),
                "name": artifact.name,
                "download_url": artifact.storage_uri,
                "checksum": artifact.checksum,
                "size_bytes": artifact.size_bytes
            })

        log_audit_event(
            action="artifact.downloaded",
            resource_type="artifact",
            resource_id=str(artifact.id),
            actor=request.user,
            metadata={"name": artifact.name, "size_bytes": artifact.size_bytes},
            request=request
        )

        filename = os.path.basename(filepath)
        response = FileResponse(open(filepath, 'rb'), as_attachment=True, filename=filename)
        return response
