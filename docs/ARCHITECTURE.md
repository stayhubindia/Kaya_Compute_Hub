# Kaya Compute Hub - System Architecture Specification

---

## 1. Overview & High-Level Architecture

**Kaya Compute Hub** is a secure, private, web-accessible control plane hosted on a Virtual Machine (VM). It is designed to orchestrate long-running data ingestion, web downloading, dataset preprocessing, and model training tasks.

### Core Architectural Axiom
**The browser never executes long-running compute directly.**
When a task is submitted from the web dashboard:
1. The dashboard submits an HTTP REST request (`credentials: "include"`) to the Django API control plane.
2. The Django API validates single-user admin authentication via HttpOnly session cookies, records state in PostgreSQL, and enqueues task identifiers into Redis.
3. Asynchronous Celery workers pick up task identifiers, lease row locks (`SELECT ... FOR UPDATE`), and execute tasks in isolated environments.
4. Execution state, progress percentage, and stage messages are persisted back to PostgreSQL and logged in audit logs.
5. If the client disconnects or closes the browser, the job continues executing uninterrupted on the VM.

```
Dashboard (Next.js) ──► Django API (REST + Session Cookies) ──► PostgreSQL (State & Row Lock)
                                                                       │
                                                                       ▼
                                                                 Redis Queue
                                                                       │
                                                                       ▼
Isolated Container Job ◄── Celery Worker Engine ───────────────────────┘
        │
        ▼
Persistent Storage (Artifacts, Datasets, Checkpoints, Logs)
```

---

## 2. Single-User Private Admin Panel Authentication

- **Single Admin Model**: Kaya Compute Hub operates as a private single-user admin control plane. Only one active admin account is permitted in PostgreSQL.
- **Admin Setup**: Initial admin account creation is performed via the secure CLI command:
  ```bash
  python manage.py create_admin --email admin@example.com --password "SecurePassword123!"
  ```
- **Cookie-Based Sessions**: HttpOnly, SameSite=Lax session cookies (`sessionid`). Zero storage of passwords or access tokens in browser `localStorage`.
- **Argon2 Password Hashing**: Passwords are saved exclusively as strong Argon2 password hashes.
- **Rate Limiting**: Throttling enforced on login requests (`10/minute`).
- **CSRF Protection**: State-changing requests enforce standard Django CSRF token headers (`X-CSRFToken`).
- **External Integration Accounts**: Connected Google Accounts for Google Drive and Colab Enterprise (`ConnectedAccount`) remain isolated external OAuth integrations and cannot be used to log into the panel.

---

## 3. Queueing & Asynchronous Job Lifecycle

```
[ draft ] ──► [ queued ] ──► [ leased ] ──► [ running ] ──► [ succeeded ]
                  │                             │                 │
                  ▼                             ▼                 ▼
             [ failed ] ◄─────────────── [ cancelled ]       [ persistent artifacts ]
                  │
                  ▼
            [ retrying ] ──► [ queued ]
```

---

## 4. Worker Fleet Registration & Heartbeat Loop

Worker nodes maintain active registration in the database:
- **Registration Metrics**: Worker name, hostname, CPU count, memory size, GPU count, GPU model, CUDA version, available GPU slots, allocated GPU slots, and capabilities (`{"docker": True, "demo_executors": True}`).
- **Heartbeat Daemon**: Periodic heartbeat updates (`POST /api/v1/workers/<id>/heartbeat/`) update `last_heartbeat_at`.
- **Stale Worker Task**: Celery periodic task sweeps workers where `last_heartbeat_at < NOW() - 60s`, setting status to `offline`.

---

## 5. Subsystem Pipelines

### Downloader Subsystem
- **Approved Source Providers**: HTTP/HTTPS, GitHub Releases, arXiv, Internet Archive.
- **SSRF Safety**: Rejects private IP addresses, loopback, link-local, and cloud metadata IPs (`169.254.169.254`).
- **Integrity Validation**: Computes streaming checksums (`sha256`, `md5`) without holding files in memory.

### Processor Subsystem
- **Allowlisted Stages**: `validate_files`, `inspect_schema`, `normalize_text`, `deduplicate`, `split_dataset`, `convert_format`, `generate_statistics`.
- **Immutability & Lineage**: Inputs are read-only. Clean outputs generate a new dataset record with explicit `parent_dataset_id`.

### Trainer Subsystem & Model Registry
- **GPU Scheduler**: Atomically reserves available GPU slots across active worker nodes before dispatching training runs.
- **Model Registry**: Registered model versions (`ModelVersion`) are stored with immutable checksums and require explicit approval before deployment.

### Operations Dashboard & Live Telemetry
- **Authenticated SSE Event Stream**: Real-time event streaming (`/api/v1/events/stream/`) for active job progress, status transitions, worker status changes, and metrics.
- **Log Streaming & Sanitization**: Live job logs are sanitized on-the-fly (`sanitize_log_message`) to redact passphrases, tokens, and credentials.
