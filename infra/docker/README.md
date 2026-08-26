# Docker Infrastructure & Local Orchestration (`/infra/docker`)

This directory contains the Docker Compose environment for local development and testing of **Kaya Compute Hub**.

---

## 🚀 Services Overview

The development compose setup (`docker-compose.dev.yml`) includes:
- **`postgres`**: PostgreSQL 16 relational database for state persistence.
- **`redis`**: Redis 7 in-memory broker & task result store.
- **`api`**: Django REST API control plane service listening on `127.0.0.1:8000`.
- **`worker`**: Celery worker engine processing async job queues as a non-root user.

---

## 💻 Launching Infrastructure

```bash
# Start all services in detached mode
docker compose -f infra/docker/docker-compose.dev.yml up -d

# View logs
docker compose -f infra/docker/docker-compose.dev.yml logs -f worker

# Stop services
docker compose -f infra/docker/docker-compose.dev.yml down
```

---

## 🔒 Security Restrictions
- All database ports (`5432`) and Redis ports (`6379`) are bound strictly to `127.0.0.1` (localhost).
- The worker container executes using a non-root user (`user: "1000:1000"`).
- No production secrets or arbitrary command injection endpoints are exposed.
