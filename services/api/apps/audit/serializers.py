from rest_framework import serializers
from apps.audit.models import AuditEvent

class AuditEventSerializer(serializers.ModelSerializer):
    actor_email = serializers.EmailField(source='actor.email', read_only=True)

    class Meta:
        model = AuditEvent
        fields = [
            'id', 'actor', 'actor_email', 'action', 'resource_type',
            'resource_id', 'metadata', 'ip_address', 'user_agent', 'created_at'
        ]
        read_only_fields = fields
