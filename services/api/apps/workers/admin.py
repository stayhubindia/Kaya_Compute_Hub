from django.contrib import admin
from apps.workers.models import Worker

@admin.register(Worker)
class WorkerAdmin(admin.ModelAdmin):
    list_display = ('name', 'hostname', 'status', 'cpu_count', 'gpu_count', 'last_heartbeat_at')
    list_filter = ('status', 'last_heartbeat_at')
    search_fields = ('name', 'hostname')
    readonly_fields = ('id', 'last_heartbeat_at', 'created_at', 'updated_at')
