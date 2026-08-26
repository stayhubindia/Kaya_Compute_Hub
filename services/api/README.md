# Kaya Compute Hub - Django REST API Service (`/services/api`)

This directory contains the Django REST API control plane service for **Kaya Compute Hub**.

---

## 🛠️ Local Development Setup

### 1. Create & Activate Virtual Environment
```bash
# From workspace root
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Backend & Worker Dependencies
```bash
pip install django djangorestframework django-filter psycopg2-binary dj-database-url python-dotenv redis celery argon2-cffi cryptography django-cors-headers pytest pytest-django
```

### 3. Environment Configuration
Copy `.env.example` to create your local `.env` file:
```bash
cp services/api/.env.example services/api/.env
```

Key environment variables:
- `DATABASE_URL`: PostgreSQL connection string (defaults to SQLite if omitted in dev).
- `REDIS_URL`: Redis connection string (`redis://localhost:6379/0`).
- `CORS_ALLOWED_ORIGINS`: Allowed origins for cookie-based CORS authentication (`http://localhost:3000`).

### 4. Database Migrations & Single Admin Setup
Run schema migrations against local/test database:
```bash
python services/api/manage.py migrate
```

Create the single private admin account:
```bash
python services/api/manage.py create_admin --email admin@example.com --password "SecurePassword123!"
```

### 5. Start Development Server & Celery Worker
Launch Django API server:
```bash
python services/api/manage.py runserver 0.0.0.0:8000
```

In a separate terminal, start the Celery worker:
```bash
celery -A services.worker.celery_app worker --loglevel=info
```

### 6. Run Test Suite
Run backend security & worker test suites using Pytest:
```bash
cd services/api
CELERY_TASK_ALWAYS_EAGER=True pytest
```

---

## 🔑 Registered API Endpoints (`/api/v1/`)

### Authentication
- `POST /api/v1/auth/login/` — Cookie-based single admin login with session rotation
- `POST /api/v1/auth/logout/` — Idempotent session invalidation & cookie purge
- `GET /api/v1/auth/me/` — Retrieve currently authenticated admin profile

### Compute Infrastructure & Jobs
- `GET /api/v1/health/` — Health check status
- `GET /api/v1/jobs/` — List jobs with progress percentages & stages
- `POST /api/v1/jobs/` — Create & enqueue job
- `GET /api/v1/jobs/<uuid>/` — Retrieve job details & progress
- `POST /api/v1/jobs/<uuid>/cancel/` — Cancel running/queued job
- `POST /api/v1/jobs/<uuid>/retry/` — Retry failed job
- `GET /api/v1/workers/` — List worker fleet
- `GET /api/v1/datasets/` — List datasets
- `POST /api/v1/datasets/` — Register dataset metadata
- `GET /api/v1/artifacts/` — List generated artifacts & checkpoints
- `GET /api/v1/audit-events/` — Read-only audit log stream

---

## 🔒 Security Baseline Highlights
- **HttpOnly Cookies**: Session cookies are marked `HttpOnly` and rotated upon authentication (`sessionid`).
- **No Token LocalStorage**: Access tokens, passwords, and tokens are never exposed to browser localStorage.
- **Argon2 Password Hashing**: Passwords are hashed using Argon2 (`argon2-cffi`).
- **Single Admin Invariant**: Only one active admin account is permitted in PostgreSQL.
