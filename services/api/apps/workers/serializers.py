from django.utils import timezone
from rest_framework import serializers
from apps.workers.models import Worker

class WorkerSerializer(serializers.ModelSerializer):
    is_stale = serializers.SerializerMethodField()
    hostname_label = serializers.SerializerMethodField()
    active_jobs_count = serializers.SerializerMethodField()

    class Meta:
        model = Worker
        fields = [
            'id', 'name', 'hostname', 'hostname_label', 'status', 'is_stale', 'capabilities',
            'cpu_count', 'memory_bytes', 'gpu_count', 'gpu_model',
            'available_gpu_slots', 'allocated_gpu_slots', 'active_jobs_count',
            'last_heartbeat_at', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'last_heartbeat_at', 'created_at', 'updated_at']

    def get_is_stale(self, obj):
        if not obj.last_heartbeat_at:
            return True
        return (timezone.now() - obj.last_heartbeat_at).total_seconds() > 60

    def get_hostname_label(self, obj):
        return obj.hostname.split('.')[0] if obj.hostname else ''

    def get_active_jobs_count(self, obj):
        from apps.jobs.models import Job, JobStatusChoices
        return Job.objects.filter(assigned_worker=obj, status=JobStatusChoices.RUNNING).count()

class WorkerHeartbeatSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=['offline', 'idle', 'busy', 'draining', 'unhealthy'], required=False)
    capabilities = serializers.JSONField(required=False)
