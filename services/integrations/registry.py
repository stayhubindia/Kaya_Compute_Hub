from typing import Dict, Any, Type
from services.integrations.google.drive_client import GoogleDriveClient
from services.integrations.colab_enterprise.client import ColabEnterpriseClient

INTEGRATION_PROVIDERS = {
    "google": GoogleDriveClient,
    "colab_enterprise": ColabEnterpriseClient,
}

def get_provider_client_class(provider_name: str):
    """Retrieve integration provider client class by provider identifier."""
    client_cls = INTEGRATION_PROVIDERS.get(provider_name.lower())
    if not client_cls:
        raise ValueError(f"Unknown integration provider: '{provider_name}'. Supported: {list(INTEGRATION_PROVIDERS.keys())}")
    return client_cls
