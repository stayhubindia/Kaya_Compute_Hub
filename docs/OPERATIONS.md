# Kaya Compute Hub — Operations & Administration Guide

## 1. System Architecture Overview
Kaya Compute Hub is a production-ready single-admin Dataset Factory + Training Orchestrator.

### Service Stack
- **Frontend Dashboard**: Next.js 14 (React 18 + Tailwind CSS) running on port `3000`.
- **API Backend**: Django 5.0 (REST Framework + Argon2id Auth) running on port `8000`.
- **Relational Database**: PostgreSQL 16 storing users, connected accounts, jobs, events, and dataset metadata.
- **Queue & Broker**: Redis 7 + Celery 5 worker fleet for async background job execution.
- **Storage Subsystem**: Folder Contract Manager targeting `DATA_ROOT` (`/srv/kaya-data`) and Google Drive API.

---

## 2. Dependency Separation Architecture

To prevent unnecessary heavy packages on the base VM, dependencies are cleanly separated into two tiers:

1. **VM Base Tier (Extraction, Panel, QA, Drive Sync)**:
   - Installed via `requirements-pipeline.txt` and `requirements-api.txt`.
   - Packages: `pydantic`, `pyyaml`, `pymupdf`, `pypdf`, `beautifulsoup4`, `lxml`, `requests`, `httpx`, `pandas`, `numpy`, `django`, `celery`, `redis`.

2. **VM GPU Training Tier (Optional Local GPU Fine-Tuning)**:
   - Installed via `requirements-gpu.txt` or `pip install -e .[gpu]`.
   - Installed **only** if QLoRA training runs on the local VM's GPU.
   - Packages: `torch`, `transformers`, `peft`, `accelerate`, `datasets`, `bitsandbytes`, `trl`.

---

## 3. Managing Services

### Starting All Services
```bash
# 1. Run PostgreSQL and Redis (via Docker Compose or local services)
docker compose up -d db redis

# 2. Apply Database Migrations
python3 services/api/manage.py migrate

# 3. Bootstrap Administrator Account (Single Admin)
python3 services/api/manage.py bootstrap_admin --email admin@kaya.local --password "ChangeMeInProduction123!"

# 4. Start Celery Worker
celery -A services.worker.celery_app worker -l info -c 4

# 5. Start API Server & Next.js Dashboard
npm run dev
```

---

## 3. Storage & Dataset Management

### Folder Contract Structure
Every dataset collection is stored under `$DATA_ROOT/<collection_slug>/` according to the 10-stage lifecycle contract:
- `00-source/`: Source metadata manifests and raw URLs.
- `10-raw/`: Unprocessed PDFs, HTML, JSON, archives.
- `20-extracted/`: Parsed `documents.jsonl`, `sections.jsonl`, OCR output.
- `30-normalized/`: Cleaned text, LaTeX equations, structured pipe tables.
- `40-chunks/`: Deterministic semantic chunks (`chunks.jsonl`).
- `50-generated/`: Synthesized Q&A instruction candidates.
- `60-qa/`: Quality audit, rights audit, deduplication, and leakage reports.
- `70-training-ready/`: Frozen `train.jsonl`, `val.jsonl`, `test.jsonl` + `FROZEN` lock flag.
- `80-training/`: Training run checkpoints, logs, and config.yaml.
- `90-evaluation/`: Model evaluation reports and benchmark predictions.

### Freezing Datasets
Datasets must be frozen via the API or Dashboard (`POST /api/v1/jobs/release/`) before training can be launched. The freeze process computes cryptographic SHA-256 signatures for all splits and writes the `FROZEN` lock file.
