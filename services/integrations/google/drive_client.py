import time
import requests
from typing import Dict, Any, List, Optional, Generator
from services.integrations.google.errors import GoogleDriveError, RateLimitError

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
UPLOAD_API_BASE = "https://www.googleapis.com/upload/drive/v3"

class GoogleDriveClient:
    def __init__(self, access_token: str):
        self.access_token = access_token

    def _headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _request_with_retry(
        self, method: str, url: str, max_retries: int = 3, **kwargs
    ) -> requests.Response:
        """Execute request with exponential backoff on 429/503 rate limit responses."""
        backoff = 1.0
        for attempt in range(max_retries + 1):
            try:
                kwargs["headers"] = self._headers(kwargs.get("headers"))
                response = requests.request(method, url, timeout=30, **kwargs)
                
                if response.status_code in (429, 503):
                    if attempt == max_retries:
                        raise RateLimitError(f"Google Drive API rate limit exceeded ({response.status_code}).")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                
                if not response.ok:
                    raise GoogleDriveError(f"Drive API Error ({response.status_code}): {response.text}")
                
                return response
            except requests.RequestException as e:
                if attempt == max_retries:
                    raise GoogleDriveError(f"Drive API Network Failure: {str(e)}") from e
                time.sleep(backoff)
                backoff *= 2
        
        raise GoogleDriveError("Max retries exceeded.")

    def list_files(self, query: Optional[str] = None, page_size: int = 50) -> Dict[str, Any]:
        """List permitted Google Drive files."""
        url = f"{DRIVE_API_BASE}/files"
        params = {
            "pageSize": page_size,
            "fields": "files(id, name, mimeType, size, createdTime, modifiedTime, md5Checksum), nextPageToken",
        }
        if query:
            params["q"] = query
        
        resp = self._request_with_retry("GET", url, params=params)
        return resp.json()

    def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Get file metadata for specific fileId."""
        url = f"{DRIVE_API_BASE}/files/{file_id}"
        params = {
            "fields": "id, name, mimeType, size, createdTime, modifiedTime, md5Checksum, parents, webViewLink",
        }
        resp = self._request_with_retry("GET", url, params=params)
        return resp.json()

    def download_file_stream(self, file_id: str, chunk_size: int = 1024 * 1024) -> Generator[bytes, None, None]:
        """Stream file bytes from Google Drive API."""
        url = f"{DRIVE_API_BASE}/files/{file_id}"
        params = {"alt": "media"}
        headers = self._headers()
        
        try:
            with requests.get(url, headers=headers, params=params, stream=True, timeout=60) as resp:
                if not resp.ok:
                    raise GoogleDriveError(f"Failed to download file {file_id}: HTTP {resp.status_code}")
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        yield chunk
        except requests.RequestException as e:
            raise GoogleDriveError(f"Network error downloading file {file_id}: {str(e)}") from e

    def upload_file(self, file_name: str, content_bytes: bytes, mime_type: str = "application/octet-stream", parent_folder_id: Optional[str] = None) -> Dict[str, Any]:
        """Upload file artifact using simple/multipart protocol."""
        url = f"{UPLOAD_API_BASE}/files?uploadType=multipart"
        metadata = {"name": file_name}
        if parent_folder_id:
            metadata["parents"] = [parent_folder_id]

        files = {
            "data": ("metadata", str(metadata).replace("'", '"'), "application/json; charset=UTF-8"),
            "file": (file_name, content_bytes, mime_type),
        }
        
        resp = self._request_with_retry("POST", url, files=files)
        return resp.json()

    def create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> Dict[str, Any]:
        """Create application-owned folder in Google Drive."""
        url = f"{DRIVE_API_BASE}/files"
        payload = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_folder_id:
            payload["parents"] = [parent_folder_id]

        resp = self._request_with_retry("POST", url, json=payload)
        return resp.json()
