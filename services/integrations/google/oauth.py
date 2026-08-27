import os
import secrets
import hashlib
import base64
import requests
from typing import Dict, Any, Tuple
from services.integrations.google.scopes import get_configured_scopes
from services.integrations.google.errors import GoogleOAuthError, TokenRevokedError

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

def generate_pkce_pair() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    code_verifier = secrets.token_urlsafe(64)
    hashed = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    code_challenge = base64.urlsafe_b64encode(hashed).decode("utf-8").rstrip("=")
    return code_verifier, code_challenge

def generate_state() -> str:
    """Generate cryptographically secure OAuth state parameter."""
    return secrets.token_urlsafe(32)

DEFAULT_GOOGLE_REDIRECT_URI = "http://localhost:8000/api/v1/integrations/google/callback/"


def _required_setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise GoogleOAuthError(f"{name} is not configured on the VM.")
    return value

def get_authorization_url(state: str, code_challenge: str, redirect_uri: str, client_id: str = None) -> str:
    """Construct Google OAuth 2.0 Authorization URL with PKCE."""
    if not client_id:
        client_id = _required_setting("GOOGLE_OAUTH_CLIENT_ID")
    
    if not redirect_uri or "localhost" in redirect_uri:
        redirect_uri = os.environ.get("GOOGLE_OAUTH_REDIRECT_URI", DEFAULT_GOOGLE_REDIRECT_URI)

    scopes = " ".join(get_configured_scopes())

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scopes,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
        "token_usage": "remote",
    }

    req = requests.Request("GET", GOOGLE_AUTH_URL, params=params)
    prepared = req.prepare()
    return prepared.url

def exchange_code_for_tokens(code: str, code_verifier: str, redirect_uri: str) -> Dict[str, Any]:
    """Exchange authorization code for access and refresh tokens."""
    client_id = _required_setting("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = _required_setting("GOOGLE_OAUTH_CLIENT_SECRET")

    if not redirect_uri or "localhost" in redirect_uri:
        redirect_uri = os.environ.get("GOOGLE_OAUTH_REDIRECT_URI", DEFAULT_GOOGLE_REDIRECT_URI)

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "code_verifier": code_verifier,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }

    try:
        response = requests.post(GOOGLE_TOKEN_URL, data=payload, timeout=10)
        if response.status_code != 200:
            err_data = response.json() if response.content else {}
            raise GoogleOAuthError(f"Token exchange failed: {err_data.get('error_description', response.text)}")
        return response.json()
    except requests.RequestException as e:
        raise GoogleOAuthError(f"Network error during OAuth token exchange: {str(e)}") from e

def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """Refresh access token using valid refresh_token."""
    client_id = _required_setting("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = _required_setting("GOOGLE_OAUTH_CLIENT_SECRET")

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    try:
        response = requests.post(GOOGLE_TOKEN_URL, data=payload, timeout=10)
        if response.status_code == 400 and "invalid_grant" in response.text:
            raise TokenRevokedError("Google refresh token has been revoked or expired.")
        if response.status_code != 200:
            err_data = response.json() if response.content else {}
            raise GoogleOAuthError(f"Token refresh failed: {err_data.get('error_description', response.text)}")
        return response.json()
    except requests.RequestException as e:
        raise GoogleOAuthError(f"Network error during token refresh: {str(e)}") from e

def fetch_google_userinfo(access_token: str) -> Dict[str, Any]:
    """Fetch profile information from Google userinfo API."""
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        response = requests.get(GOOGLE_USERINFO_URL, headers=headers, timeout=10)
        if response.status_code != 200:
            raise GoogleOAuthError(f"Failed to fetch Google userinfo: {response.text}")
        return response.json()
    except requests.RequestException as e:
        raise GoogleOAuthError(f"Network error fetching Google userinfo: {str(e)}") from e

def revoke_token(token: str) -> bool:
    """Revoke token with Google revocation endpoint."""
    if not token:
        return True
    payload = {"token": token}
    try:
        resp = requests.post(GOOGLE_REVOKE_URL, data=payload, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False
