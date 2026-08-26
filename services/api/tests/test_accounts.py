import pytest
from django.core.management import call_command, CommandError
from django.core.exceptions import ValidationError
from rest_framework.test import APIClient
from django.urls import reverse
from apps.accounts.models import User

@pytest.mark.django_db
def test_single_admin_creation_and_duplicate_rejection():
    admin_user = User.objects.create_admin('admin@kaya.local', 'SecureAdminPass123!')
    assert admin_user.email == 'admin@kaya.local'
    assert admin_user.is_active is True
    assert not hasattr(admin_user, 'password_hash') # Password is stored in password field as hash
    assert admin_user.password.startswith('argon2') or admin_user.password.startswith('pbkdf2')

    # Duplicate active admin creation MUST be rejected
    with pytest.raises(ValidationError) as excinfo:
        User.objects.create_admin('another_admin@kaya.local', 'SecureAdminPass123!')
    assert 'An active admin account already exists' in str(excinfo.value)

@pytest.mark.django_db
def test_create_admin_management_command_duplicate_rejection(monkeypatch):
    User.objects.create_admin('admin@kaya.local', 'SecureAdminPass123!')
    with pytest.raises(CommandError) as excinfo:
        call_command('create_admin')
    assert 'An active admin account already exists' in str(excinfo.value)

@pytest.mark.django_db
def test_auth_me_endpoint_single_admin():
    admin = User.objects.create_admin('admin@kaya.local', 'SecureAdminPass123!')
    client = APIClient()
    url = reverse('auth-me')

    # Unauthenticated check returns 401/403
    unauth_resp = client.get(url)
    assert unauth_resp.status_code in (401, 403)

    # Authenticated admin check
    client.force_authenticate(user=admin)
    auth_resp = client.get(url)
    assert auth_resp.status_code == 200
    data = auth_resp.json()
    assert data['email'] == 'admin@kaya.local'
    assert 'password' not in data
    assert 'role' not in data
    assert 'totp_enabled' not in data
