from rest_framework import serializers
from apps.downloads.models import Download, DownloadStatus

class DownloadCreateSerializer(serializers.Serializer):
    url = serializers.CharField(required=True)
    expected_checksum = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)
    checksum_algorithm = serializers.ChoiceField(
        choices=['sha256', 'sha512', 'md5'],
        default='sha256'
    )
    extract = serializers.BooleanField(default=False)

class DownloadSerializer(serializers.ModelSerializer):
    created_by = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = Download
        fields = [
            'id',
            'created_by',
            'source_url',
            'provider',
            'original_filename',
            'storage_uri',
            'content_type',
            'expected_size_bytes',
            'downloaded_size_bytes',
            'checksum_algorithm',
            'expected_checksum',
            'actual_checksum',
            'extract',
            'status',
            'progress_percent',
            'current_speed_bytes',
            'retry_count',
            'error_code',
            'error_message',
            'started_at',
            'completed_at',
            'created_at',
            'updated_at',
        ]
        read_only_fields = [
            'id', 'created_by', 'provider', 'storage_uri', 'downloaded_size_bytes',
            'actual_checksum', 'status', 'progress_percent', 'current_speed_bytes',
            'retry_count', 'error_code', 'error_message', 'started_at', 'completed_at',
            'created_at', 'updated_at'
        ]
