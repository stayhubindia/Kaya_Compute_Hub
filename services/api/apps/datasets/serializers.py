from rest_framework import serializers
from apps.datasets.models import Dataset

class DatasetSerializer(serializers.ModelSerializer):
    created_by_email = serializers.EmailField(source='created_by.email', read_only=True)

    class Meta:
        model = Dataset
        fields = [
            'id', 'name', 'description', 'source_url', 'storage_uri',
            'format', 'size_bytes', 'checksum', 'status',
            'created_by', 'created_by_email', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_by', 'created_by_email', 'created_at', 'updated_at']
