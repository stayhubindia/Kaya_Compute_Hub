class GoogleIntegrationError(Exception):
    """Base exception for Google Integration errors."""
    pass

class GoogleDriveError(GoogleIntegrationError):
    """Exception raised during Google Drive API operations."""
    pass

class RateLimitError(GoogleIntegrationError):
    """Exception raised when Google API rate limit (429/503) is encountered."""
    pass
