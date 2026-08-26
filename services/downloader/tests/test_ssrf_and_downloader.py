import os
import zipfile
import tarfile
import pytest
from services.downloader.security import (
    validate_url_security, SSRFError,
    sanitize_filename, generate_safe_internal_filename,
    validate_zip_safety, validate_tar_safety, ArchiveSafetyError
)
from services.downloader.storage import calculate_file_checksum, verify_file_checksum
from services.downloader.providers import get_provider_for_url, GitHubProvider, ArXivProvider, InternetArchiveProvider, GenericHTTPProvider

def test_ssrf_blocked_localhost_and_private_ips():
    with pytest.raises(SSRFError, match="blocked"):
        validate_url_security("http://localhost:8000/file.zip")

    with pytest.raises(SSRFError, match="blocked"):
        validate_url_security("http://127.0.0.1/file.zip")

    with pytest.raises(SSRFError, match="blocked"):
        validate_url_security("http://10.0.0.1/dataset.tar.gz")

    with pytest.raises(SSRFError, match="blocked"):
        validate_url_security("http://169.254.169.254/latest/meta-data/")

def test_ssrf_blocked_unsupported_schemes():
    with pytest.raises(SSRFError, match="Unsupported scheme"):
        validate_url_security("file:///etc/passwd")

    with pytest.raises(SSRFError, match="Unsupported scheme"):
        validate_url_security("ftp://example.com/file.txt")

    with pytest.raises(SSRFError, match="Unsupported scheme"):
        validate_url_security("gopher://example.com")

def test_ssrf_blocked_embedded_credentials():
    with pytest.raises(SSRFError, match="embedded credentials"):
        validate_url_security("http://admin:password@example.com/file.zip")

def test_ssrf_allowed_https_url():
    scheme, hostname, port = validate_url_security("https://github.com/repository/release.zip")
    assert scheme == "https"
    assert hostname == "github.com"
    assert port == 443

def test_filename_sanitization():
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("safe_file.csv") == "safe_file.csv"
    assert sanitize_filename("") == "download.bin"

    internal_name = generate_safe_internal_filename("550e8400-e29b-41d4-a716-446655440000", "my_dataset.zip")
    assert internal_name == "550e8400-e29b-41d4-a716-446655440000.zip"

def test_checksum_calculation_and_verification(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_bytes(b"Hello Kaya Compute Hub!")

    checksum = calculate_file_checksum(str(test_file), algorithm="sha256")
    assert len(checksum) == 64

    is_valid, _ = verify_file_checksum(str(test_file), checksum, algorithm="sha256")
    assert is_valid is True

    is_valid_wrong, _ = verify_file_checksum(str(test_file), "0000000000000000000000000000000000000000000000000000000000000000", algorithm="sha256")
    assert is_valid_wrong is False

def test_zip_archive_path_traversal_detection(tmp_path):
    malicious_zip = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious_zip, 'w') as zf:
        zf.writestr("../../evil.sh", b"echo hacked")

    with zipfile.ZipFile(malicious_zip, 'r') as zf:
        with pytest.raises(ArchiveSafetyError, match="path traversal"):
            validate_zip_safety(zf, str(tmp_path))

def test_tar_archive_symlink_rejection(tmp_path):
    tar_path = tmp_path / "symlink.tar"
    with tarfile.open(tar_path, 'w') as tf:
        ti = tarfile.TarInfo(name="link.txt")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)

    with tarfile.open(tar_path, 'r') as tf:
        with pytest.raises(ArchiveSafetyError, match="Symlink or hardlink"):
            validate_tar_safety(tf, str(tmp_path))

def test_provider_registry_resolution():
    p_gh = get_provider_for_url("https://github.com/user/repo/releases/download/v1.0/file.zip")
    assert isinstance(p_gh, GitHubProvider)

    p_arxiv = get_provider_for_url("https://arxiv.org/abs/2301.00001")
    assert isinstance(p_arxiv, ArXivProvider)

    p_ia = get_provider_for_url("https://archive.org/details/sample_item")
    assert isinstance(p_ia, InternetArchiveProvider)

    p_http = get_provider_for_url("https://example.com/dataset.parquet")
    assert isinstance(p_http, GenericHTTPProvider)
