import pytest
from rest_framework.test import APIClient
from apps.accounts.models import User
from apps.audit.models import AuditEvent
from apps.audit.services import log_audit_event

@pytest.mark.django_db
def test_audit_event_logging_and_listing():
    admin = User.objects.create_admin('admin@kaya.local', 'pass')
    
    # Log event via service helper
    event = log_audit_event(
        action='system.security_check',
        resource_type='system',
        resource_id='host-vm',
        actor=admin,
        metadata={'status': 'passed'}
    )
    assert event.id is not None

    client = APIClient()
    client.force_authenticate(user=admin)

    resp = client.get('/api/v1/audit-events/')
    assert resp.status_code == 200
    assert len(resp.json()['results']) >= 1
    assert resp.json()['results'][0]['action'] == 'system.security_check'
