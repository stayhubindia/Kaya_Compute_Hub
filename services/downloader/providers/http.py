import os
import time
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any, Callable

from services.downloader.providers.base import BaseProvider
from packages.contracts.download_contracts import ProviderMetadataContract
from services.downloader.security import validate_url_security, SSRFError, sanitize_filename

class GenericHTTPProvider(BaseProvider):
    name: str = "generic_http"

    def can_handle(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        return parsed.scheme.lower() in ('http', 'https')

    def get_metadata(self, url: str) -> ProviderMetadataContract:
        self.validate(url)

        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'KayaComputeHub-Downloader/1.0'})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                content_type = resp.headers.get('Content-Type')
                content_length_str = resp.headers.get('Content-Length')
                accept_ranges = resp.headers.get('Accept-Ranges', '')
                content_disp = resp.headers.get('Content-Disposition', '')

                expected_size = int(content_length_str) if content_length_str and content_length_str.isdigit() else None
                supports_range = (accept_ranges.lower() == 'bytes')

                filename = None
                if 'filename=' in content_disp:
                    filename = content_disp.split('filename=')[-1].strip('"\'')
                else:
                    parsed_path = urllib.parse.urlparse(url).path
                    if parsed_path:
                        filename = os.path.basename(parsed_path)

                return ProviderMetadataContract(
                    provider_name=self.name,
                    filename=sanitize_filename(filename or "download.bin"),
                    content_type=content_type,
                    expected_size_bytes=expected_size,
                    supports_range=supports_range
                )
        except Exception:
            # Fallback metadata if HEAD request fails or is blocked
            parsed_path = urllib.parse.urlparse(url).path
            filename = os.path.basename(parsed_path) if parsed_path else "download.bin"
            return ProviderMetadataContract(
                provider_name=self.name,
                filename=sanitize_filename(filename),
                content_type="application/octet-stream",
                expected_size_bytes=None,
                supports_range=False
            )

    def supports_resume(self, url: str) -> bool:
        meta = self.get_metadata(url)
        return meta.supports_range

    def download(
        self,
        url: str,
        destination_path: str,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> Dict[str, Any]:
        self.validate(url)

        existing_bytes = 0
        if os.path.exists(destination_path):
            existing_bytes = os.path.getsize(destination_path)

        headers = {'User-Agent': 'KayaComputeHub-Downloader/1.0'}
        mode = 'wb'

        # Check Range support if partial file exists
        if existing_bytes > 0 and self.supports_resume(url):
            headers['Range'] = f'bytes={existing_bytes}-'
            mode = 'ab'
        else:
            existing_bytes = 0

        req = urllib.request.Request(url, headers=headers)
        start_time = time.time()
        downloaded = existing_bytes

        with urllib.request.urlopen(req, timeout=30) as resp:
            content_length_str = resp.headers.get('Content-Length')
            content_length = int(content_length_str) if content_length_str and content_length_str.isdigit() else None
            total_bytes = (existing_bytes + content_length) if content_length else 0

            with open(destination_path, mode) as f:
                chunk_size = 64 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    elapsed = time.time() - start_time
                    speed = (downloaded - existing_bytes) / elapsed if elapsed > 0 else 0.0

                    if progress_callback:
                        progress_callback(downloaded, total_bytes, speed)

        return {
            "status": "completed",
            "destination_path": destination_path,
            "downloaded_bytes": downloaded,
            "total_bytes": total_bytes
        }
