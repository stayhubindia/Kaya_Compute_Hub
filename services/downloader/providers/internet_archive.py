import urllib.parse
from services.downloader.providers.http import GenericHTTPProvider

class InternetArchiveProvider(GenericHTTPProvider):
    name: str = "internet_archive"

    def can_handle(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or "").lower()
        return 'archive.org' in hostname
