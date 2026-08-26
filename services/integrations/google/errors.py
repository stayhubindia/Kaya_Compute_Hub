class GoogleIntegrationError(Exception):
    """Base exception for Google Integration errors."""
    pass

class GoogleOAuthError(GoogleIntegrationError):
    """Exception raised during OAuth flow validation or token exchange."""
    pass

class TokenRevokedError(GoogleOAuthError):
    """Exception raised when an access or refresh token has been revoked by Google."""
    pass

class GoogleDriveError(GoogleIntegrationError):
    """Exception raised during Google Drive API operations."""
    pass

class RateLimitError(GoogleIntegrationError):
    """Exception raised when Google API rate limit (429/503) is encountered."""
    pass
