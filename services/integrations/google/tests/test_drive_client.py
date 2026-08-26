import pytest
from unittest.mock import patch, MagicMock
from services.integrations.google.drive_client import GoogleDriveClient
from services.integrations.google.errors import GoogleDriveError, RateLimitError

@patch("requests.request")
def test_list_files_success(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "files": [
            {"id": "file_1", "name": "dataset.csv", "mimeType": "text/csv", "size": "1048576"}
        ]
    }
    mock_request.return_value = mock_resp

    client = GoogleDriveClient("mock_token")
    res = client.list_files()
    assert len(res["files"]) == 1
    assert res["files"][0]["name"] == "dataset.csv"

@patch("requests.request")
def test_rate_limit_exponential_backoff(mock_request):
    mock_rate_limit_resp = MagicMock()
    mock_rate_limit_resp.status_code = 429
    mock_rate_limit_resp.ok = False

    mock_request.return_value = mock_rate_limit_resp

    client = GoogleDriveClient("mock_token")
    with pytest.raises(RateLimitError):
        client.get_file_metadata("file_123")

    assert mock_request.call_count == 4  # Initial + 3 retries

@patch("requests.get")
def test_download_file_stream(mock_get):
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_get.return_value.__enter__.return_value = mock_resp

    client = GoogleDriveClient("mock_token")
    chunks = list(client.download_file_stream("file_123"))
    assert len(chunks) == 2
    assert b"".join(chunks) == b"chunk1chunk2"
