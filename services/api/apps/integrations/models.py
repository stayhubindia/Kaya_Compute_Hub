import uuid
from django.db import models
from django.conf import settings
from services.integrations.google.token_store import encrypt_token, decrypt_token
from services.integrations.colab_enterprise.executions import ExternalRunStatus

class AccountStatusChoices(models.TextChoices):
    ACTIVE = 'active', 'Active'
    EXPIRED = 'expired', 'Expired'
    REVOKED = 'revoked', 'Revoked'
    DISCONNECTED = 'disconnected', 'Disconnected'
    QUOTA_EXHAUSTED = 'quota_exhausted', 'Quota Exhausted'
    ERROR = 'error', 'Error'

class ConnectedAccount(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='connected_accounts')
    provider = models.CharField(max_length=50, default='google')
    provider_account_id = models.CharField(max_length=255)
    email = models.EmailField(blank=True, default='')
    display_name = models.CharField(max_length=255, blank=True, default='')
    encrypted_access_token = models.TextField(blank=True, default='')
    encrypted_refresh_token = models.TextField(blank=True, default='')
    token_expiry = models.DateTimeField(null=True, blank=True)
    scopes = models.JSONField(default=list)
    status = models.CharField(max_length=30, choices=AccountStatusChoices.choices, default=AccountStatusChoices.ACTIVE)
    last_verified_at = models.DateTimeField(null=True, blank=True)
    connected_at = models.DateTimeField(auto_now_add=True)
    disconnected_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-connected_at']
        unique_together = ('user', 'provider', 'provider_account_id')

    def __str__(self):
        return f"{self.provider.capitalize()} ({self.email or self.provider_account_id}) - {self.user.email}"

    def set_access_token(self, raw_token: str):
        self.encrypted_access_token = encrypt_token(raw_token)

    def get_access_token(self) -> str:
        return decrypt_token(self.encrypted_access_token)

    def set_refresh_token(self, raw_token: str):
        if raw_token:
            self.encrypted_refresh_token = encrypt_token(raw_token)

    def get_refresh_token(self) -> str:
        return decrypt_token(self.encrypted_refresh_token)


class OAuthState(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    state = models.CharField(max_length=255, unique=True, db_index=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='oauth_states')
    code_verifier = models.CharField(max_length=255)
    redirect_uri = models.CharField(max_length=500)
    scopes = models.JSONField(default=list)
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"OAuthState {self.state[:8]} for {self.user.email}"


class ExternalNotebook(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    provider = models.CharField(max_length=50, default='colab_enterprise')
    project_id = models.CharField(max_length=255)
    region = models.CharField(max_length=100, default='us-central1')
    notebook_resource_name = models.CharField(max_length=500)
    display_name = models.CharField(max_length=255)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='external_notebooks')
    status = models.CharField(max_length=50, default='ready')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.display_name} ({self.project_id}/{self.region})"


class ExternalRun(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    notebook = models.ForeignKey(ExternalNotebook, on_delete=models.SET_NULL, null=True, blank=True, related_name='runs')
    selected_google_account = models.ForeignKey(ConnectedAccount, on_delete=models.SET_NULL, null=True, blank=True, related_name='external_runs')
    provider = models.CharField(max_length=50, default='colab_enterprise')
    external_run_id = models.CharField(max_length=255, db_index=True)
    local_job_id = models.CharField(max_length=255, null=True, blank=True)
    status = models.CharField(max_length=30, choices=ExternalRunStatus.CHOICES, default=ExternalRunStatus.REQUESTED)
    output_uri = models.CharField(max_length=500, blank=True, default='')
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    error_code = models.CharField(max_length=100, null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"ExternalRun {self.external_run_id} [{self.status}]"
