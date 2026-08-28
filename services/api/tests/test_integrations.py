import pytest
from django.urls import reverse
from rest_framework import status
from apps.accounts.models import User
from apps.integrations.models import ConnectedAccount, AccountStatusChoices, ExternalNotebook, ExternalRun

@pytest.mark.django_db
def test_google_accounts_list(client):
    admin = User.objects.create_admin(email="admin@example.com", password="Password123!")
    acc1 = ConnectedAccount.objects.create(user=admin, provider="google", provider_account_id="sub_1", email="u1@gmail.com")

    client.force_login(admin)

    url = reverse("google-accounts-list")
    res = client.get(url)
    assert res.status_code == status.HTTP_200_OK
    data = res.json()["results"]
    assert len(data) == 1
    assert data[0]["id"] == str(acc1.id)
    assert data[0]["email"] == "u1@gmail.com"

@pytest.mark.django_db
def test_direct_colab_token_import(client):
    admin = User.objects.create_admin(email="direct@example.com", password="Password123!")
    client.force_login(admin)
    response = client.post(
        reverse("google-account-direct-connect"),
        {
            "email": "drive@example.com",
            "raw_json": '{"token":"access-token","refresh_token":"refresh-token","scopes":["drive"]}',
        },
        format="json",
    )
    assert response.status_code == status.HTTP_201_CREATED
    account = ConnectedAccount.objects.get(email="drive@example.com")
    assert account.get_access_token() == "access-token"
    assert "refresh_token" in account.get_credential_json()

@pytest.mark.django_db
def test_colab_enterprise_register_and_run(client):
    admin = User.objects.create_admin(email="colab_user@example.com", password="Password123!")
    client.force_login(admin)

    # Register notebook
    reg_url = reverse("colab-list-notebooks")
    payload = {
        "provider": "colab_enterprise",
        "project_id": "test-gcp-project",
        "region": "us-central1",
        "notebook_resource_name": "projects/test-gcp-project/locations/us-central1/notebooks/my-notebook",
        "display_name": "Customer Segmentation Model",
        "environment_spec": {"machine_type": "n1-standard-4"}
    }
    res_reg = client.post(reg_url, payload, format="json")
    assert res_reg.status_code == status.HTTP_201_CREATED
    nb_id = res_reg.data["id"]

    nb = ExternalNotebook.objects.get(id=nb_id)
    assert nb.owner == admin

    # Run notebook
    run_url = reverse("colab-run-notebook", kwargs={"pk": nb.id})
    res_run = client.post(run_url, {"output_uri": "gs://test-bucket/output"}, format="json")
    assert res_run.status_code == status.HTTP_201_CREATED
    assert res_run.data["status"] == "requested"

@pytest.mark.django_db
def test_selected_google_account_quota_exhausted_rejection(client):
    admin = User.objects.create_admin(email="quota_user@example.com", password="Password123!")
    acc = ConnectedAccount.objects.create(
        user=admin,
        provider="google",
        provider_account_id="sub_quota",
        email="quota@gmail.com",
        status=AccountStatusChoices.QUOTA_EXHAUSTED
    )
    nb = ExternalNotebook.objects.create(
        provider="colab_enterprise",
        project_id="test-gcp-project",
        region="us-central1",
        notebook_resource_name="projects/test-gcp-project/locations/us-central1/notebooks/test-nb",
        display_name="Test Notebook",
        owner=admin
    )

    client.force_login(admin)

    # Attempting to pass account marked quota_exhausted must return 400 Bad Request
    run_url = reverse("colab-run-notebook", kwargs={"pk": nb.id})
    res = client.post(run_url, {"selected_google_account_id": str(acc.id)}, format="json")
    assert res.status_code == status.HTTP_400_BAD_REQUEST
    assert "marked quota_exhausted" in res.data["error"]["message"]
