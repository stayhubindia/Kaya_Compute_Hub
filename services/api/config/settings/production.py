from .base import *
from django.core.exceptions import ImproperlyConfigured

DEBUG = False

for required_name in ('DJANGO_SECRET_KEY', 'TOTP_ENCRYPTION_KEY', 'GOOGLE_TOKEN_ENCRYPTION_KEY'):
    if not os.environ.get(required_name):
        raise ImproperlyConfigured(f'{required_name} must be configured in production.')

# Security Enhancements for Production
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = os.environ.get('DJANGO_SESSION_COOKIE_SECURE', 'True').lower() in ('true', '1', 'yes')
SESSION_COOKIE_SAMESITE = 'Lax'
CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SECURE = os.environ.get('DJANGO_CSRF_COOKIE_SECURE', 'True').lower() in ('true', '1', 'yes')
CSRF_COOKIE_SAMESITE = 'Lax'
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = int(os.environ.get('DJANGO_SECURE_HSTS_SECONDS', '31536000'))
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
