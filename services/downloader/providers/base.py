from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from packages.contracts.download_contracts import ProviderMetadataContract
from services.downloader.security import validate_url_security

class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Returns True if this provider can handle the given URL."""
        pass

    def validate(self, url: str) -> bool:
        """Validates that the URL satisfies SSRF policy and provider constraints."""
        validate_url_security(url)
        return True

    @abstractmethod
    def get_metadata(self, url: str) -> ProviderMetadataContract:
        """Retrieves remote metadata (content length, filename, range support)."""
        pass

    @abstractmethod
    def download(
        self,
        url: str,
        destination_path: str,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Downloads content to destination_path.
        progress_callback(downloaded_bytes, total_bytes, speed_bytes_per_sec)
        """
        pass

    @abstractmethod
    def supports_resume(self, url: str) -> bool:
        """Returns True if HTTP Range requests are supported."""
        pass
