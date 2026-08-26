import urllib.parse
from services.downloader.providers.http import GenericHTTPProvider

class GitHubProvider(GenericHTTPProvider):
    name: str = "github"

    def can_handle(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or "").lower()
        return hostname in ('github.com', 'raw.githubusercontent.com', 'codeload.github.com')
