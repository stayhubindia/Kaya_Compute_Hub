from django.contrib import admin
from apps.datasets.models import Dataset

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'format', 'status', 'size_bytes', 'created_by', 'created_at')
    list_filter = ('status', 'format', 'created_at')
    search_fields = ('name', 'description', 'source_url', 'storage_uri')
    readonly_fields = ('id', 'created_at', 'updated_at')
