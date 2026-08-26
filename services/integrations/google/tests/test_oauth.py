import pytest
from unittest.mock import patch, MagicMock
from services.integrations.google.oauth import (
    generate_pkce_pair,
    generate_state,
    get_authorization_url,
    exchange_code_for_tokens,
    refresh_access_token,
    fetch_google_userinfo,
    revoke_token
)
from services.integrations.google.token_store import encrypt_token, decrypt_token
from services.integrations.google.errors import GoogleOAuthError, TokenRevokedError

def test_pkce_and_state_generation():
    verifier, challenge = generate_pkce_pair()
    assert len(verifier) >= 43
    assert len(challenge) >= 43

    state = generate_state()
    assert len(state) >= 32

def test_token_encryption_decryption():
    raw_token = "ya29.a0ARdaC0B_sample_google_access_token_12345"
    encrypted = encrypt_token(raw_token)
    assert encrypted != raw_token
    assert "ya29" not in encrypted

    decrypted = decrypt_token(encrypted)
    assert decrypted == raw_token

def test_authorization_url_construction():
    state = "test_state_123"
    code_challenge = "test_challenge_456"
    redirect_uri = "http://localhost:8000/api/v1/integrations/google/callback/"

    url = get_authorization_url(state, code_challenge, redirect_uri)
    assert "https://accounts.google.com/o/oauth2/v2/auth" in url
    assert "state=test_state_123" in url
    assert "code_challenge=test_challenge_456" in url
    assert "code_challenge_method=S256" in url

@patch("requests.post")
def test_exchange_code_for_tokens(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "ya29.mock_access_token",
        "refresh_token": "1//mock_refresh_token",
        "expires_in": 3600,
        "token_type": "Bearer"
    }
    mock_post.return_value = mock_resp

    tokens = exchange_code_for_tokens("test_code", "test_verifier", "http://localhost/callback")
    assert tokens["access_token"] == "ya29.mock_access_token"
    assert tokens["refresh_token"] == "1//mock_refresh_token"

@patch("requests.post")
def test_refresh_access_token_revoked(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = '{"error": "invalid_grant", "error_description": "Token has been expired or revoked."}'
    mock_post.return_value = mock_resp

    with pytest.raises(TokenRevokedError):
        refresh_access_token("revoked_refresh_token")

@patch("requests.post")
def test_revoke_token_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_post.return_value = mock_resp

    res = revoke_token("sample_token_to_revoke")
    assert res is True
