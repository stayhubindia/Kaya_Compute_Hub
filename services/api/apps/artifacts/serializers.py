from rest_framework import serializers
from apps.artifacts.models import Artifact

class ArtifactSerializer(serializers.ModelSerializer):
    created_by_email = serializers.EmailField(source='created_by.email', read_only=True)
    job_name = serializers.CharField(source='job.name', read_only=True)

    class Meta:
        model = Artifact
        fields = [
            'id', 'name', 'artifact_type', 'storage_uri', 'size_bytes',
            'checksum', 'metadata', 'job', 'job_name', 'created_by',
            'created_by_email', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_by', 'created_by_email', 'job_name', 'created_at', 'updated_at']
