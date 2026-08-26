# Kaya Compute Hub - Worker Engine (`/services/worker`)

This directory contains the Celery asynchronous worker implementation, demo job execution handlers, worker heartbeat loops, and task state management.

---

## 🛠️ Worker Architecture

```
Celery Queue (Redis) ──► execute_job task ──► Atomic Row Lock (select_for_update)
                                                      │
                                                      ▼
  Progress Events ◄── Approved Demo Executor ◄── Lease & State Transition
```

---

## 🔒 Approved Demo Executors
The worker engine enforces a strict allowlist mapping of `job_type` to approved demo handlers:

- **`download`**: Simulates non-network data downloading and progress reporting (10% $\rightarrow$ 50% $\rightarrow$ 100%).
- **`extraction`**: Simulates file extraction and progress reporting (20% $\rightarrow$ 70% $\rightarrow$ 100%).
- **`preprocessing`**: Simulates dataset cleaning and progress reporting (15% $\rightarrow$ 60% $\rightarrow$ 100%).
- **`notebook` / `training` / `evaluation`**: Returns `NotImplementedError` safely without executing code.

**Security Rule**: Arbitrary shell execution, payload Python code evaluation (`exec`/`eval`), and unsafe SQL string concatenation are strictly prohibited.

---

## 💻 Running Worker Locally

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Ensure Redis is running locally or via Docker
redis-cli ping

# 3. Launch Celery worker
celery -A services.worker.celery_app worker --loglevel=info
```
