# Google Integration Package

This package provides direct Colab CLI credential import, encrypted token storage, and Google Drive API integration for Kaya Compute Hub.

## Architectural Guarantees

1. **Token Security at Rest**: Access tokens and refresh tokens are encrypted at rest using AES/Fernet (`GOOGLE_TOKEN_ENCRYPTION_KEY`). Raw tokens are never logged or returned in API responses.
2. **Direct account verification**: Imported credentials are checked with the Drive `about.get` endpoint before they are used for jobs.
3. **Resumable Drive Adapter**: Includes streaming download support and exponential backoff retry on 429/503 rate limits.

## Configuration Variables

- Account credentials are imported through the dashboard from the account's official Colab CLI `token.json`.
- `GOOGLE_TOKEN_ENCRYPTION_KEY`: 32-byte Fernet symmetric encryption key for token ciphertexts.
