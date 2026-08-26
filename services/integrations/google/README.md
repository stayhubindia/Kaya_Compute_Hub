# Google Integration Package

This package provides secure Google OAuth 2.0 PKCE authentication, encrypted token storage, and Google Drive API integration for Kaya Compute Hub.

## Architectural Guarantees

1. **Token Security at Rest**: Access tokens and refresh tokens are encrypted at rest using AES/Fernet (`GOOGLE_TOKEN_ENCRYPTION_KEY`). Raw tokens are never logged or returned in API responses.
2. **PKCE & State Validation**: Authorizations use cryptographically random PKCE code verifiers/challenges and single-use `OAuthState` records to prevent replay and authorization code injection attacks.
3. **Resumable Drive Adapter**: Includes streaming download support and exponential backoff retry on 429/503 rate limits.

## Configuration Variables

- `GOOGLE_OAUTH_CLIENT_ID`: Google Cloud Console OAuth 2.0 Web Client ID.
- `GOOGLE_OAUTH_CLIENT_SECRET`: Google OAuth 2.0 Web Client Secret.
- `GOOGLE_OAUTH_REDIRECT_URI`: Registered redirect URI callback (`/api/v1/integrations/google/callback/`).
- `GOOGLE_OAUTH_SCOPES`: Comma-separated scopes (`openid,email,profile,https://www.googleapis.com/auth/drive.file`).
- `GOOGLE_TOKEN_ENCRYPTION_KEY`: 32-byte Fernet symmetric encryption key for token ciphertexts.
