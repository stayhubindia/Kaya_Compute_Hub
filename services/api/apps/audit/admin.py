from django.contrib import admin
from apps.audit.models import AuditEvent

@admin.register(AuditEvent)
class AuditEventAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'actor', 'action', 'resource_type', 'resource_id', 'ip_address')
    list_filter = ('action', 'resource_type', 'created_at')
    search_fields = ('action', 'resource_type', 'resource_id', 'actor__email')
    readonly_fields = ('id', 'actor', 'action', 'resource_type', 'resource_id', 'metadata', 'ip_address', 'user_agent', 'created_at')

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
