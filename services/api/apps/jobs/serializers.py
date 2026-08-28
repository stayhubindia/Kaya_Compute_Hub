from rest_framework import serializers
import re
from apps.jobs.models import Job, JobStatusChoices, JobTypeChoices

class JobSerializer(serializers.ModelSerializer):
    created_by_email = serializers.EmailField(source='created_by.email', read_only=True)
    assigned_worker_name = serializers.CharField(source='assigned_worker.name', read_only=True)

    class Meta:
        model = Job
        fields = [
            'id', 'created_by', 'created_by_email', 'name', 'description',
            'job_type', 'status', 'priority', 'payload', 'idempotency_key',
            'selected_google_account',
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
    selected_google_account_id = serializers.UUIDField(required=False, allow_null=True, write_only=True)

    class Meta:
        model = Job
        fields = ['name', 'description', 'job_type', 'priority', 'payload', 'idempotency_key', 'max_retries', 'selected_google_account_id']

    def validate(self, attrs):
        request = self.context['request']
        payload = dict(attrs.get('payload') or {})
        # Jobs are compute workloads, not VM workloads.  Always persist the
        # explicit target so a worker can never silently fall back to local CPU.
        payload['execution_target'] = 'colab'
        attrs['payload'] = payload
        account_id = attrs.get('selected_google_account_id')
        pipeline_types = {
            JobTypeChoices.INGESTION, JobTypeChoices.GENERATION,
            JobTypeChoices.QUALITY_AUDIT, JobTypeChoices.FREEZE_DATASET,
            JobTypeChoices.TRAINING_QLORA, JobTypeChoices.SYNC_DRIVE,
        }
        if not account_id:
            raise serializers.ValidationError({'selected_google_account_id': 'Select an active Google account for Colab execution.'})
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', str(payload.get('session_name', ''))):
            raise serializers.ValidationError({'payload': 'Select a valid live Colab session for execution.'})
        if attrs.get('job_type') not in pipeline_types and not payload.get('code', '').strip():
            raise serializers.ValidationError({'payload': 'Python code is required for this Colab job.'})
        if account_id:
            from apps.integrations.models import ConnectedAccount, AccountStatusChoices
            account = ConnectedAccount.objects.filter(id=account_id, user=request.user).first()
            if not account:
                raise serializers.ValidationError({'selected_google_account_id': 'Selected Google account was not found.'})
            if account.status != AccountStatusChoices.ACTIVE:
                raise serializers.ValidationError({'selected_google_account_id': f"Selected Google account is {account.status}."})
            attrs['selected_google_account'] = account
        return attrs
