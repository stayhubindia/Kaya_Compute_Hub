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
    
    # Fallback to Django SECRET_KEY if available or static dev key
    secret = os.environ.get("DJANGO_SECRET_KEY", "dev-secret-key-change-in-prod")
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
    except Exception as e:
        raise ValueError("Failed to decrypt stored token: Invalid key or payload corrupted.") from e
