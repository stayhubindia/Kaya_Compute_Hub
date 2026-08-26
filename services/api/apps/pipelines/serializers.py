from rest_framework import serializers
from apps.pipelines.models import PipelineDefinition, ProcessingRun, ProcessingStageEvent, DatasetManifest
from services.processor.pipeline import validate_pipeline_definition
from services.processor.containers import validate_resource_policy, ResourcePolicyError
from services.processor.stages import StageValidationError

class PipelineDefinitionSerializer(serializers.ModelSerializer):
    created_by = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = PipelineDefinition
        fields = [
            'id', 'name', 'description', 'version', 'enabled',
            'stages', 'resource_policy', 'created_by', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_by', 'created_at', 'updated_at']

    def validate_stages(self, value):
        try:
            validate_pipeline_definition(value)
        except StageValidationError as e:
            raise serializers.ValidationError(str(e))
        return value

    def validate_resource_policy(self, value):
        try:
            return validate_resource_policy(value)
        except ResourcePolicyError as e:
            raise serializers.ValidationError(str(e))

class ProcessingRunSerializer(serializers.ModelSerializer):
    created_by = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = ProcessingRun
        fields = [
            'id', 'pipeline', 'source_dataset', 'output_dataset', 'status',
            'current_stage', 'progress_percent', 'input_manifest_uri', 'output_manifest_uri',
            'error_code', 'error_message', 'started_at', 'completed_at',
            'created_by', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'output_dataset', 'status', 'current_stage', 'progress_percent',
            'input_manifest_uri', 'output_manifest_uri', 'error_code', 'error_message',
            'started_at', 'completed_at', 'created_by', 'created_at', 'updated_at'
        ]

class ProcessingRunCreateSerializer(serializers.Serializer):
    pipeline_id = serializers.UUIDField(required=True)
    source_dataset_id = serializers.UUIDField(required=True)

class ProcessingStageEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProcessingStageEvent
        fields = [
            'id', 'processing_run', 'stage_name', 'status',
            'input_uri', 'output_uri', 'metrics', 'log_uri',
            'started_at', 'completed_at', 'created_at'
        ]

class DatasetManifestSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetManifest
        fields = [
            'id', 'dataset', 'schema_version', 'file_count',
            'total_bytes', 'checksum', 'format', 'columns',
            'statistics', 'provenance', 'created_at'
        ]
