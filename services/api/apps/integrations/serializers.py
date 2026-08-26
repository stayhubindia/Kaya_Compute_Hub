from rest_framework import serializers
from apps.integrations.models import ConnectedAccount, ExternalNotebook, ExternalRun

class ConnectedAccountSerializer(serializers.ModelSerializer):
    """Serializer for connected provider accounts. Explicitly excludes token ciphertexts."""
    class Meta:
        model = ConnectedAccount
        fields = [
            'id',
            'provider',
            'provider_account_id',
            'email',
            'display_name',
            'token_expiry',
            'scopes',
            'status',
            'last_verified_at',
            'connected_at',
            'disconnected_at',
            'created_at',
            'updated_at',
        ]
        read_only_fields = fields


class ExternalNotebookSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExternalNotebook
        fields = [
            'id',
            'provider',
            'project_id',
            'region',
            'notebook_resource_name',
            'display_name',
            'owner',
            'status',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'owner', 'created_at', 'updated_at']


class ExternalRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExternalRun
        fields = [
            'id',
            'notebook',
            'provider',
            'external_run_id',
            'local_job_id',
            'status',
            'output_uri',
            'started_at',
            'finished_at',
            'error_code',
            'error_message',
            'metadata',
            'created_at',
            'updated_at',
        ]
        read_only_fields = fields
