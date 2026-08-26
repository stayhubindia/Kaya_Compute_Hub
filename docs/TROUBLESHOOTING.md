# Kaya Compute Hub — Troubleshooting & Diagnostic Guide

## 1. Common Operational Issues

### Issue: Task Queue Broker Unavailable (HTTP 503)
- **Symptom**: Submitting a job returns `503 Service Unavailable: Task queue broker is unavailable`.
- **Cause**: Redis server or Celery worker is down.
- **Resolution**:
  1. Verify Redis process: `redis-cli ping` (should return `PONG`).
  2. Restart Celery worker: `celery -A services.worker.celery_app worker -l info`.

---

### Issue: Training Preflight Blocked ("Dataset Lifecycle State Fail")
- **Symptom**: `train_qlora` job fails with `Dataset for collection_slug has not been frozen!`.
- **Cause**: Attempting to launch model training on an unfrozen candidate directory.
- **Resolution**:
  1. Run Quality Audit: `POST /api/v1/jobs/qa/`.
  2. Freeze Dataset: `POST /api/v1/jobs/release/` to create `70-training-ready/FROZEN` lock file.

---

### Issue: Google Drive Sync Auth Failure (HTTP 401 / 403)
- **Symptom**: `sync_to_drive` job pauses and logs authentication/permission error.
- **Policy Reminder**: Automatic account rotation is strictly disabled per security policy.
- **Resolution**:
  1. Re-authorize account via `/dashboard/settings/connections`.
  2. Submit sync job specifying valid `account_id`.
