from rest_framework import generics
from apps.audit.models import AuditEvent
from apps.audit.serializers import AuditEventSerializer
from apps.accounts.permissions import IsAuthenticatedAdmin

class AuditEventListView(generics.ListAPIView):
    serializer_class = AuditEventSerializer
    permission_classes = [IsAuthenticatedAdmin]
    filterset_fields = ['action', 'resource_type', 'actor']
    search_fields = ['action', 'resource_type', 'resource_id']
    ordering_fields = ['created_at']

    def get_queryset(self):
        return AuditEvent.objects.all()
