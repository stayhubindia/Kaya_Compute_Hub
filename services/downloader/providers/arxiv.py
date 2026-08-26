import urllib.parse
from services.downloader.providers.http import GenericHTTPProvider

class ArXivProvider(GenericHTTPProvider):
    name: str = "arxiv"

    def can_handle(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or "").lower()
        return 'arxiv.org' in hostname

    def normalize_url(self, url: str) -> str:
        if '/abs/' in url:
            return url.replace('/abs/', '/pdf/') + '.pdf'
        return url

    def download(self, url: str, destination_path: str, options=None, progress_callback=None):
        target_url = self.normalize_url(url)
        return super().download(target_url, destination_path, options, progress_callback)
