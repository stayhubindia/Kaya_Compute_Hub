import socket
import ipaddress
import urllib.parse
from typing import Tuple, Optional, List
from django.conf import settings

# Forbidden IPv4 and IPv6 Networks
FORBIDDEN_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"), # Link-local & Cloud Metadata (169.254.169.254)
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443, 8080, 8443}

class SSRFError(ValueError):
    """Raised when a URL violates SSRF security policy."""
    pass

def is_ip_allowed(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        for net in FORBIDDEN_NETWORKS:
            if ip in net:
                return False
        return True
    except ValueError:
        return False

def validate_url_security(url: str) -> Tuple[str, str, int]:
    """
    Validates a URL against SSRF policy rules.
    Returns (scheme, hostname, port).
    Raises SSRFError if invalid or blocked.
    """
    if not url:
        raise SSRFError("URL cannot be empty.")

    parsed = urllib.parse.urlparse(url)

    # 1. Scheme Check
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise SSRFError(f"Unsupported scheme '{parsed.scheme}'. Only http and https are allowed.")

    # HTTP scheme policy check if enforced
    allow_http = getattr(settings, 'DOWNLOAD_ALLOW_HTTP', True)
    if parsed.scheme.lower() == 'http' and not allow_http:
        raise SSRFError("Plain HTTP is disabled in production settings. Please use HTTPS.")

    # 2. Embedded Credentials Check
    if parsed.username or parsed.password:
        raise SSRFError("URLs with embedded credentials (user:pass@host) are strictly prohibited.")

    # 3. Hostname Check
    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL missing valid hostname.")

    hostname_lower = hostname.lower()
    if hostname_lower in ('localhost', 'loopback', '127.0.0.1', '::1'):
        raise SSRFError(f"Access to local host '{hostname}' is blocked.")

    # 4. Port Check
    port = parsed.port or (443 if parsed.scheme.lower() == 'https' else 80)
    if port not in ALLOWED_PORTS:
        allowed_extra_ports = getattr(settings, 'DOWNLOAD_ALLOWED_PORTS', set())
        if port not in allowed_extra_ports:
            raise SSRFError(f"Access to port {port} is blocked by security policy.")

    # 5. Domain Allowlist / Blocklist Check
    allowed_domains: List[str] = getattr(settings, 'DOWNLOAD_ALLOWED_DOMAINS', [])
    blocked_domains: List[str] = getattr(settings, 'DOWNLOAD_BLOCKED_DOMAINS', [])

    if blocked_domains:
        for b_dom in blocked_domains:
            if hostname_lower == b_dom.lower() or hostname_lower.endswith('.' + b_dom.lower()):
                raise SSRFError(f"Domain '{hostname}' is explicitly blocked by policy.")

    if allowed_domains:
        domain_matched = False
        for a_dom in allowed_domains:
            if hostname_lower == a_dom.lower() or hostname_lower.endswith('.' + a_dom.lower()):
                domain_matched = True
                break
        if not domain_matched:
            raise SSRFError(f"Domain '{hostname}' is not in the allowed domains list.")

    # 6. Pre-flight DNS Resolution & IP Check
    try:
        addr_info = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise SSRFError(f"Failed to resolve DNS for hostname '{hostname}': {str(e)}")

    if not addr_info:
        raise SSRFError(f"No IP addresses resolved for hostname '{hostname}'.")

    for item in addr_info:
        ip_str = item[4][0]
        if not is_ip_allowed(ip_str):
            raise SSRFError(f"Resolved IP address '{ip_str}' for domain '{hostname}' belongs to a blocked/private network range.")

    return parsed.scheme.lower(), hostname, port
