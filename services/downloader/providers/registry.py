from typing import List
from services.downloader.providers.base import BaseProvider
from services.downloader.providers.github import GitHubProvider
from services.downloader.providers.arxiv import ArXivProvider
from services.downloader.providers.internet_archive import InternetArchiveProvider
from services.downloader.providers.http import GenericHTTPProvider

class ProviderRegistry:
    def __init__(self):
        self._providers: List[BaseProvider] = [
            GitHubProvider(),
            ArXivProvider(),
            InternetArchiveProvider(),
            GenericHTTPProvider(), # Fallback default
        ]

    def get_provider(self, url: str) -> BaseProvider:
        for provider in self._providers:
            if provider.can_handle(url):
                return provider
        return GenericHTTPProvider()

registry = ProviderRegistry()

def get_provider_for_url(url: str) -> BaseProvider:
    return registry.get_provider(url)
