from django.contrib import admin
from apps.jobs.models import Job

@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ('name', 'job_type', 'status', 'created_by', 'priority', 'created_at')
    list_filter = ('job_type', 'status', 'created_at')
    search_fields = ('name', 'description', 'idempotency_key', 'created_by__email')
    readonly_fields = ('id', 'started_at', 'finished_at', 'created_at', 'updated_at')
