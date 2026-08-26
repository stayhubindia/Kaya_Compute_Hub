import pytest
from unittest.mock import patch, MagicMock
from services.integrations.colab_enterprise.client import ColabEnterpriseClient, validate_project_and_region
from services.integrations.colab_enterprise.executions import map_colab_state_to_run_status, ExternalRunStatus
from services.integrations.google.errors import GoogleIntegrationError

def test_project_and_region_allowlist_validation(monkeypatch):
    monkeypatch.setenv("GOOGLE_ALLOWED_PROJECTS", "allowed-project-1,allowed-project-2")
    monkeypatch.setenv("GOOGLE_ALLOWED_REGIONS", "us-central1,europe-west1")

    # Valid project & region
    validate_project_and_region("allowed-project-1", "us-central1")

    # Invalid project
    with pytest.raises(GoogleIntegrationError) as exc_info:
        validate_project_and_region("unauthorized-project", "us-central1")
    assert "not in the allowed project allowlist" in str(exc_info.value)

    # Invalid region
    with pytest.raises(GoogleIntegrationError) as exc_info:
        validate_project_and_region("allowed-project-1", "ap-southeast-1")
    assert "not in the allowed region allowlist" in str(exc_info.value)

def test_gcp_state_mapping():
    assert map_colab_state_to_run_status("JOB_STATE_RUNNING") == ExternalRunStatus.RUNNING
    assert map_colab_state_to_run_status("JOB_STATE_SUCCEEDED") == ExternalRunStatus.COMPLETED
    assert map_colab_state_to_run_status("JOB_STATE_FAILED") == ExternalRunStatus.FAILED
    assert map_colab_state_to_run_status("JOB_STATE_CANCELLED") == ExternalRunStatus.CANCELLED
    assert map_colab_state_to_run_status("JOB_STATE_EXPIRED") == ExternalRunStatus.TIMED_OUT

@patch("requests.post")
def test_create_execution_submission(mock_post, monkeypatch):
    monkeypatch.setenv("GOOGLE_ALLOWED_PROJECTS", "test-project")
    monkeypatch.setenv("GOOGLE_ALLOWED_REGIONS", "us-central1")

    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "name": "projects/test-project/locations/us-central1/notebookExecutionJobs/exec-123",
        "jobState": "JOB_STATE_PENDING"
    }
    mock_post.return_value = mock_resp

    client = ColabEnterpriseClient("mock_token", "test-project", "us-central1")
    res = client.create_execution("projects/test-project/locations/us-central1/notebooks/sample-nb")
    assert "exec-123" in res["name"]
    assert res["jobState"] == "JOB_STATE_PENDING"
