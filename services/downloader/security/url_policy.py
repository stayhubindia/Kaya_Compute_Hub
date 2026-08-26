from services.downloader.security.ssrf_protection import validate_url_security, SSRFError

def validate_download_url(url: str) -> bool:
    """
    Returns True if the URL satisfies all SSRF and policy requirements.
    Raises SSRFError if invalid.
    """
    validate_url_security(url)
    return True
