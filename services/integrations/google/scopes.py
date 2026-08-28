from typing import List

# Scope definitions
SCOPE_OPENID = "openid"
SCOPE_EMAIL = "email"
SCOPE_PROFILE = "profile"
SCOPE_DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"
SCOPE_DRIVE_READONLY = "https://www.googleapis.com/auth/drive.readonly"
SCOPE_COLABORATORY = "https://www.googleapis.com/auth/colaboratory"
SCOPE_CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"

DEFAULT_SCOPES = [
    SCOPE_OPENID,
    SCOPE_EMAIL,
    SCOPE_PROFILE,
    SCOPE_CLOUD_PLATFORM,
    SCOPE_DRIVE_FILE,
    SCOPE_COLABORATORY
]

def get_configured_scopes() -> List[str]:
    return list(DEFAULT_SCOPES)

def validate_scopes(requested_scopes: List[str]) -> bool:
    """Ensure requested scopes are in the permitted allowlist."""
    allowed = set(DEFAULT_SCOPES + [SCOPE_DRIVE_READONLY])
    return all(scope in allowed for scope in requested_scopes)
