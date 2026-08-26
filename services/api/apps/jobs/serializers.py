from rest_framework import serializers
from apps.jobs.models import Job, JobStatusChoices, JobTypeChoices

class JobSerializer(serializers.ModelSerializer):
    created_by_email = serializers.EmailField(source='created_by.email', read_only=True)
    assigned_worker_name = serializers.CharField(source='assigned_worker.name', read_only=True)

    class Meta:
        model = Job
        fields = [
            'id', 'created_by', 'created_by_email', 'name', 'description',
            'job_type', 'status', 'priority', 'payload', 'idempotency_key',
            'progress_percentage', 'current_stage', 'progress_message',
            'assigned_worker', 'assigned_worker_name', 'retry_count', 'max_retries',
            'error_code', 'error_message', 'started_at', 'finished_at',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'created_by', 'created_by_email', 'status',
            'progress_percentage', 'current_stage', 'progress_message',
            'assigned_worker', 'assigned_worker_name', 'retry_count',
            'error_code', 'error_message', 'started_at', 'finished_at',
            'created_at', 'updated_at'
        ]

    def validate_idempotency_key(self, value):
        if not value:
            return value
        user = self.context['request'].user
        if Job.objects.filter(created_by=user, idempotency_key=value).exists():
            raise serializers.ValidationError("A job with this idempotency key already exists for your account.")
        return value

class JobCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Job
        fields = ['name', 'description', 'job_type', 'priority', 'payload', 'idempotency_key', 'max_retries']
