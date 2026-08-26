from apps.audit.models import AuditEvent

def log_audit_event(action: str, resource_type: str, resource_id: str = "", actor=None, metadata: dict = None, request=None) -> AuditEvent:
    """
    Append-only helper to record administrative, system, or operational audit events.
    """
    ip_address = None
    user_agent = ""

    if request:
        if not actor and hasattr(request, 'user') and request.user.is_authenticated:
            actor = request.user
        
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip_address = x_forwarded_for.split(',')[0].strip()
        else:
            ip_address = request.META.get('REMOTE_ADDR')
            
        user_agent = request.META.get('HTTP_USER_AGENT', '')

    return AuditEvent.objects.create(
        actor=actor,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id),
        metadata=metadata or {},
        ip_address=ip_address,
        user_agent=user_agent
    )
