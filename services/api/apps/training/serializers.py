from rest_framework import serializers
from apps.training.models import TrainingRun, TrainingMetric, TrainingCheckpoint
from apps.datasets.models import Dataset, DatasetStatusChoices
from apps.pipelines.models import ProcessingRun, ProcessingRunStatus
from services.trainer.policies import validate_training_configuration, validate_training_resource_policy, TrainingPolicyError
from services.trainer.containers.image_registry import is_approved_training_image

class TrainingRunSerializer(serializers.ModelSerializer):
    created_by = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = TrainingRun
        fields = [
            'id', 'name', 'created_by', 'dataset', 'processing_run',
            'backend', 'container_image', 'container_digest',
            'configuration', 'resource_policy', 'status',
            'current_epoch', 'current_step', 'progress_percent',
            'best_metric', 'best_metric_name', 'checkpoint_uri', 'output_model_uri',
            'error_code', 'error_message', 'started_at', 'finished_at',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'created_by', 'status', 'current_epoch', 'current_step',
            'progress_percent', 'best_metric', 'best_metric_name',
            'checkpoint_uri', 'output_model_uri', 'error_code', 'error_message',
            'started_at', 'finished_at', 'created_at', 'updated_at'
        ]

class TrainingRunCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=255)
    dataset_id = serializers.UUIDField(required=True)
    processing_run_id = serializers.UUIDField(required=False, allow_null=True)
    backend = serializers.CharField(default='demo')
    container_image = serializers.CharField(default='kaya/ml-trainer:pytorch-2.2')
    configuration = serializers.JSONField(default=dict)
    resource_policy = serializers.JSONField(default=dict)

    def validate_dataset_id(self, value):
        try:
            ds = Dataset.objects.get(id=value)
            if ds.status != DatasetStatusChoices.AVAILABLE:
                raise serializers.ValidationError("Dataset is not available for training.")
            return ds
        except Dataset.DoesNotExist:
            raise serializers.ValidationError("Target dataset does not exist.")

    def validate_processing_run_id(self, value):
        if not value:
            return None
        try:
            pr = ProcessingRun.objects.get(id=value)
            if pr.status != ProcessingRunStatus.SUCCEEDED:
                raise serializers.ValidationError("Processing run has not completed successfully.")
            return pr
        except ProcessingRun.DoesNotExist:
            raise serializers.ValidationError("Processing run does not exist.")

    def validate_container_image(self, value):
        if not is_approved_training_image(value):
            raise serializers.ValidationError(f"Unapproved container image '{value}'.")
        return value

    def validate_configuration(self, value):
        try:
            return validate_training_configuration(value)
        except TrainingPolicyError as e:
            raise serializers.ValidationError(str(e))

    def validate_resource_policy(self, value):
        try:
            return validate_training_resource_policy(value)
        except TrainingPolicyError as e:
            raise serializers.ValidationError(str(e))

class TrainingMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingMetric
        fields = ['id', 'training_run', 'step', 'epoch', 'name', 'value', 'split', 'timestamp']

class TrainingCheckpointSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingCheckpoint
        fields = ['id', 'training_run', 'step', 'epoch', 'storage_uri', 'checksum', 'size_bytes', 'metrics', 'status', 'created_at']
