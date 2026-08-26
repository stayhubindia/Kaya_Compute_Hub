import os
import base64
import hashlib
from cryptography.fernet import Fernet
from typing import Optional

def _get_fernet() -> Fernet:
    key_env = os.environ.get("GOOGLE_TOKEN_ENCRYPTION_KEY") or os.environ.get("TOTP_ENCRYPTION_KEY")
    if key_env:
        # Pad or hash to 32 bytes URL-safe base64 string
        key_bytes = key_env.encode("utf-8")
        hashed = hashlib.sha256(key_bytes).digest()
        b64_key = base64.urlsafe_b64encode(hashed)
        return Fernet(b64_key)
    
    # Fallback to Django SECRET_KEY if available or dynamically generated key
    secret = os.environ.get("DJANGO_SECRET_KEY") or os.environ.get("SECRET_KEY")
    if not secret:
        import secrets
        secret = secrets.token_urlsafe(64)
    hashed = hashlib.sha256(secret.encode("utf-8")).digest()
    b64_key = base64.urlsafe_b64encode(hashed)
    return Fernet(b64_key)

def encrypt_token(raw_token: str) -> str:
    """Encrypt plain token string into ciphertext."""
    if not raw_token:
        return ""
    fernet = _get_fernet()
    encrypted = fernet.encrypt(raw_token.encode("utf-8"))
    return encrypted.decode("utf-8")

def decrypt_token(encrypted_token: str) -> str:
    """Decrypt ciphertext token into plain string."""
    if not encrypted_token:
        return ""
    try:
        fernet = _get_fernet()
        decrypted = fernet.decrypt(encrypted_token.encode("utf-8"))
        return decrypted.decode("utf-8")
    except Exception:
        # Fallback to returning raw string if unencrypted or Fernet key signature mismatch
        return encrypted_token
