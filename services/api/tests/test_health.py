import pytest
from rest_framework.test import APIClient
from django.urls import reverse

@pytest.mark.django_db
def test_health_endpoint():
    client = APIClient()
    url = reverse('health-check')
    response = client.get(url)
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "kaya-compute-api"
    }
