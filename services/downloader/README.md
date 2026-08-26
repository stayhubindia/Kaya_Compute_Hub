# Downloader Subsystem

The Downloader service provides a secure, resumable background dataset download manager for the Kaya Compute Hub.

## Architectural Components

- **Providers (`providers/`)**: Isolated source providers (`http.py`, `github.py`, `arxiv.py`, `internet_archive.py`, `registry.py`).
- **Security (`security/`)**:
  - `ssrf_protection.py`: Pre-flight DNS IP resolution, blocking loopback, private RFC1918, link-local, cloud metadata IP (`169.254.169.254`), non-HTTP schemes, embedded credentials, and unsafe ports.
  - `archive_safety.py`: Safe ZIP/TAR/GZIP archive extraction with path traversal (`../`) prevention, symlink rejection, zip bomb protection, and max file count checks.
  - `filename_policy.py`: Safe internal filename generation.
- **Storage & Integrity (`storage/`)**:
  - Streaming checksum validation (`sha256`, `sha512`, `md5`).
  - Temporary download management and atomic dataset rename into `storage/datasets/`.
  - Quarantine of failed or invalid downloads into `storage/quarantine/`.
- **Tasks (`tasks/`)**:
  - Celery task `process_download_job` handling background execution, progress persistence, and audit logging.

## Usage & Testing

Run downloader unit tests:
```bash
pytest services/downloader/tests/
```
