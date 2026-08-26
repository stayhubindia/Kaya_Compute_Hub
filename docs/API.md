# Kaya Compute Hub — REST API Specification

## 1. Authentication Endpoints
All API endpoints require administrative authentication via HTTP-only session cookies.

- `POST /api/v1/auth/login/`
  - Payload: `{"email": "admin@kaya.local", "password": "..."}`
- `POST /api/v1/auth/logout/`
- `GET /api/v1/auth/me/`

---

## 2. Dataset Factory Job Endpoints

### Dispatch Document Ingestion
- `POST /api/v1/jobs/ingest/`
  - Payload:
    ```json
    {
      "collection_slug": "cybersecurity_v1",
      "input_path": "/srv/kaya-data/raw_sources",
      "source": "arXiv",
      "resume": true
    }
    ```

### Dispatch Candidate Generation
- `POST /api/v1/jobs/generate/`
  - Payload:
    ```json
    {
      "collection_slug": "cybersecurity_v1",
      "source": "generated",
      "seed": 42
    }
    ```

### Dispatch QA Audit
- `POST /api/v1/jobs/qa/`
  - Payload: `{"collection_slug": "cybersecurity_v1"}`

### Dispatch Freeze Dataset
- `POST /api/v1/jobs/release/`
  - Payload:
    ```json
    {
      "collection_slug": "cybersecurity_v1",
      "version_name": "v1.0"
    }
    ```

### Dispatch QLoRA Model Training
- `POST /api/v1/jobs/train/`
  - Payload:
    ```json
    {
      "collection_slug": "cybersecurity_v1",
      "base_model": "Qwen/Qwen3-4B-Base"
    }
    ```

### Dispatch Google Drive Sync
- `POST /api/v1/jobs/sync/`
  - Payload:
    ```json
    {
      "collection_slug": "cybersecurity_v1",
      "account_id": "account_uuid"
    }
    ```

---

## 3. Job Monitoring Endpoints
- `GET /api/v1/jobs/` — List active & completed jobs.
- `GET /api/v1/jobs/{job_id}/` — View detailed status and progress.
- `POST /api/v1/jobs/{job_id}/cancel/` — Cancel running or queued job.
- `POST /api/v1/jobs/{job_id}/retry/` — Retry failed job.
