import requests
from typing import Dict, Any, Optional
from services.integrations.google.errors import GoogleIntegrationError

class GoogleStorageClient:
    """Helper client for reading Colab Enterprise outputs from Google Cloud Storage."""

    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token

    def read_object_metadata(self, bucket: str, object_name: str) -> Dict[str, Any]:
        """Read GCS object metadata via JSON API."""
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{object_name}"
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if not resp.ok:
                raise GoogleIntegrationError(f"Failed to fetch GCS object metadata ({resp.status_code}): {resp.text}")
            return resp.json()
        except requests.RequestException as e:
            raise GoogleIntegrationError(f"Network error reading GCS object: {str(e)}") from e
