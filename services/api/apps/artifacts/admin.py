from django.contrib import admin
from apps.artifacts.models import Artifact

@admin.register(Artifact)
class ArtifactAdmin(admin.ModelAdmin):
    list_display = ('name', 'artifact_type', 'size_bytes', 'job', 'created_by', 'created_at')
    list_filter = ('artifact_type', 'created_at')
    search_fields = ('name', 'storage_uri', 'checksum')
    readonly_fields = ('id', 'created_at', 'updated_at')
