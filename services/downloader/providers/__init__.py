from .base import BaseProvider
from .http import GenericHTTPProvider
from .github import GitHubProvider
from .arxiv import ArXivProvider
from .internet_archive import InternetArchiveProvider
from .registry import ProviderRegistry, registry, get_provider_for_url

__all__ = [
    'BaseProvider',
    'GenericHTTPProvider',
    'GitHubProvider',
    'ArXivProvider',
    'InternetArchiveProvider',
    'ProviderRegistry',
    'registry',
    'get_provider_for_url',
]
