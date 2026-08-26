import pytest
from rest_framework.test import APIClient
from django.urls import reverse
from apps.accounts.models import User
from apps.integrations.models import ConnectedAccount, AccountStatusChoices
from apps.audit.models import AuditEvent

@pytest.mark.django_db
def test_login_success_and_session_rotation():
    user = User.objects.create_admin('admin@kaya.local', 'Password123!')
    client = APIClient()

    resp = client.post('/api/v1/auth/login/', {
        'email': 'admin@kaya.local',
        'password': 'Password123!'
    }, format='json')

    assert resp.status_code == 200
    data = resp.json()
    assert data['user']['email'] == 'admin@kaya.local'
    assert 'password' not in data['user']
    assert 'sessionid' in client.cookies
    assert client.cookies['sessionid']['httponly'] is True
    assert client.cookies['sessionid']['samesite'] in ('Lax', 'Strict')

    # Verify audit log recorded
    assert AuditEvent.objects.filter(action='auth.login_success', resource_id=str(user.id)).exists()

@pytest.mark.django_db
def test_login_invalid_password_generic_error():
    User.objects.create_admin('admin@kaya.local', 'CorrectPassword!')
    client = APIClient()

    resp = client.post('/api/v1/auth/login/', {
        'email': 'admin@kaya.local',
        'password': 'WrongPassword!'
    }, format='json')

    assert resp.status_code in (401, 403)
    assert resp.json()['error']['message'] == "Invalid email or password."
    assert AuditEvent.objects.filter(action='auth.login_failure').exists()

@pytest.mark.django_db
def test_inactive_user_cannot_login():
    user = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    user.is_active = False
    user.save(update_fields=['is_active'])

    client = APIClient()

    resp = client.post('/api/v1/auth/login/', {
        'email': 'admin@kaya.local',
        'password': 'Pass123!'
    }, format='json')

    assert resp.status_code in (401, 403)

@pytest.mark.django_db
def test_logout_invalidates_session():
    user = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    client = APIClient()

    # Login via endpoint to set real session cookie
    login_resp = client.post('/api/v1/auth/login/', {
        'email': 'admin@kaya.local',
        'password': 'Pass123!'
    }, format='json')
    assert login_resp.status_code == 200

    # Logout
    logout_resp = client.post('/api/v1/auth/logout/')
    assert logout_resp.status_code == 200

    # Current user check should be 401/403
    me_resp = client.get('/api/v1/auth/me/')
    assert me_resp.status_code in (401, 403)

@pytest.mark.django_db
def test_external_google_account_not_panel_user():
    admin = User.objects.create_admin('admin@kaya.local', 'Pass123!')
    google_acc = ConnectedAccount.objects.create(
        user=admin,
        provider_account_id='google_123',
        email='external_google@gmail.com',
        encrypted_access_token='cipher',
        status=AccountStatusChoices.ACTIVE
    )

    client = APIClient()
    # Attempting to log into panel with external google email must fail
    resp = client.post('/api/v1/auth/login/', {
        'email': 'external_google@gmail.com',
        'password': 'Pass123!'
    }, format='json')

    assert resp.status_code in (401, 403)
    assert resp.json()['error']['message'] == "Invalid email or password."
