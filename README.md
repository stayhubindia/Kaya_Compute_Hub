# Kaya Compute Hub 🚀

**Kaya Compute Hub** is a secure, private, web-accessible control plane running on a Dedicated Virtual Machine (VM). It enables users to submit, monitor, and manage compute-heavy tasks—such as data extraction, web downloads, notebook execution, dataset preprocessing, and model training—from a modern web dashboard.

---

## 🏗️ Core Architecture Overview

Kaya Compute Hub uses a decoupled, asynchronous architecture designed to handle long-running compute workloads reliably without depending on an active web browser session.

```
Dashboard (Next.js) ──► Django API (REST + Cookies) ──► PostgreSQL + Redis
                                                               │
                                                               ▼
Isolated Docker Job ◄── Celery Worker Engine ──────────────────┘
        │
        ▼
Persistent VM Storage (Artifacts, Models, Checkpoints, Logs, Datasets)
```

- **Dashboard**: Modern Web UI built with Next.js & React for task submission, worker monitoring, security controls, and dataset tracking.
- **Django API**: Backend REST service (`services/api`) managing single-admin cookie authentication, database state, user quotas, and Redis task queues.
- **PostgreSQL & Redis**: Persistent state database storing admin, job, dataset, download, processing run, training run, model registry, and audit metadata alongside Redis in-memory message queue.
- **Celery Worker**: Asynchronous background job processor managing task leasing, worker heartbeats, retries, and task execution lifecycle.
- **Downloader Subsystem**: Resumable, SSRF-protected dataset downloading engine with provider support, streaming checksum verification, safe archive extraction, and quota management.
- **Processor Subsystem**: Safe, resumable dataset preprocessing pipeline engine (`services/processor`) supporting allowlisted stages, checkpoint recovery, dataset immutability, and manifest generation.
- **Trainer Subsystem & Model Registry**: Controlled ML model training engine (`services/trainer`) supporting GPU capacity scheduling, hyperparameter validation, atomic checkpoint recovery, metric streaming, and admin-gatekept immutable model registration (`apps/models_registry`).
- **Live Operations Dashboard & Telemetry**: Real-time operations dashboard supporting authenticated Server-Sent Events (SSE), sanitized log streaming, uniform stride downsampled metric visualizations, worker fleet monitoring, and secure artifact browsing.
- **Integration Subsystem (Google OAuth, Drive, & Colab Enterprise)**: External cloud integration service (`services/integrations`) supporting server-side Google OAuth 2.0 PKCE authentication, AES/Fernet token encryption at rest, Google Drive file browsing & streaming download import, and Google Cloud Colab Enterprise notebook execution management (`apps/integrations`).
- **Isolated Docker Job**: Sandboxed execution runtime with strictly limited CPU, memory, GPU, disk, and execution timeout boundaries.

---

## 📁 Repository Directory Architecture

```
kaya-compute-hub/
├── app/                  # Next.js web dashboard routes & pages
│   ├── login/            # Secure admin login page (email & password)
│   ├── dashboard/        # Dashboard overview, jobs (operations center), workers, datasets, artifacts, integrations
├── components/           # DashboardNavbar, terminal log viewer, metrics chart panel, worker cards, artifact tables
├── lib/                  # Frontend utilities & API clients
│   ├── api/              # Typed API clients (authClient, jobsClient, workersClient, eventsClient, etc.)
│   └── hooks/            # Custom React hooks (useJobEvents, useJobLogs, useWorkerStatus)
├── packages/             # Shared contracts & SDKs
│   └── contracts/        # Data contracts & payload schemas
├── services/             # Backend microservices & workloads
│   ├── api/              # Django API server (auth, models, API endpoints)
│   │   ├── config/       # Django project configuration & settings (celery.py, settings)
│   │   ├── apps/         # Django apps (accounts, jobs, workers, datasets, artifacts, audit, downloads, pipelines, training, models_registry, events, logs, monitoring)
│   │   ├── tests/        # Pytest test suite (auth, state machine, downloads, pipelines, training, events, logs, monitoring)
│   │   └── manage.py     # Django management CLI
│   ├── downloader/       # Resumable dataset & download manager
│   ├── processor/        # Safe dataset processing pipeline engine
│   ├── trainer/          # Controlled model training engine & GPU scheduler
│   └── worker/           # Celery worker service & execution engine
│       └── events/       # Event publishing & payload sanitization helpers
├── infra/                # System & deployment infrastructure
├── storage/              # VM-persistent runtime storage (datasets, models, temp, quarantine, checkpoints)
```

---

## 🔒 Security Baseline Highlights
- **Single Admin Private Panel**: Kaya Compute Hub operates as a private single-user admin control plane. Only one active admin account exists in PostgreSQL, created via `python manage.py create_admin`.
- **HttpOnly Session Cookies**: Authentication uses HttpOnly, SameSite=Lax session cookies. Zero exposure of tokens or passwords in browser `localStorage`.
- **Argon2 Password Hashing**: Passwords are hashed using Argon2 (`argon2-cffi`).
- **External Integration Isolation**: Connected Google Accounts for Google Drive and Colab Enterprise remain external integration credentials and cannot log into the admin panel.
