from django.contrib import admin
from apps.accounts.models import User

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('email', 'is_active', 'created_at', 'last_login')
    list_filter = ('is_active',)
    search_fields = ('email',)
    ordering = ('-created_at',)
    readonly_fields = ('created_at', 'updated_at', 'last_login')
