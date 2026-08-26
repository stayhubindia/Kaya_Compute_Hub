# Downloads App (`apps/downloads`)

Django application exposing secure dataset download endpoints and quota management for the Kaya Compute Hub control plane.

## Endpoints

- `POST /api/v1/downloads/`: Submit a download request. Validates URL against SSRF policy and checks user quotas before queueing.
- `GET /api/v1/downloads/`: List permitted downloads. Operators/admins see all downloads; users see their own.
- `GET /api/v1/downloads/<uuid>/`: Retrieve download details.
- `POST /api/v1/downloads/<uuid>/cancel/`: Cancel an active download.
- `POST /api/v1/downloads/<uuid>/pause/`: Pause a download.
- `POST /api/v1/downloads/<uuid>/resume/`: Resume a paused download.
- `POST /api/v1/downloads/<uuid>/verify/`: Re-verify checksum integrity.

## User Quota Enforcement (`quota.py`)

- `DOWNLOAD_MAX_FILE_SIZE_BYTES`: Maximum per-file size (default 10 GB).
- `DOWNLOAD_DAILY_QUOTA_BYTES`: Maximum daily downloaded bytes per user (default 50 GB).
- `DOWNLOAD_MAX_CONCURRENT_PER_USER`: Maximum active downloads per user (default 5).
