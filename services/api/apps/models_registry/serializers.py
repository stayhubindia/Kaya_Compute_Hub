from rest_framework import serializers
from apps.models_registry.models import ModelVersion, ModelStatusChoices

class ModelVersionSerializer(serializers.ModelSerializer):
    created_by = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = ModelVersion
        fields = [
            'id', 'name', 'version', 'training_run', 'artifact',
            'framework', 'framework_version', 'model_format', 'checksum',
            'metadata', 'status', 'created_by', 'created_at'
        ]
        read_only_fields = ['id', 'status', 'created_by', 'created_at']

class ModelVersionCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=255)
    version = serializers.CharField(max_length=50)
    training_run_id = serializers.UUIDField(required=False, allow_null=True)
    framework = serializers.CharField(max_length=100)
    framework_version = serializers.CharField(max_length=50, default='')
    model_format = serializers.CharField(max_length=50)
    checksum = serializers.CharField(max_length=128)
    metadata = serializers.JSONField(default=dict)
