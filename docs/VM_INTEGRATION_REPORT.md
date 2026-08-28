# VM Integration & Compatibility Report (Phase 0)

**Date**: 2026-08-26  
**Target Platform**: Single-Admin Dataset Factory & Training Orchestrator (VM Application)  
**Source Archive**: `platform_integration_20260826_140041.zip`

---

## 1. Archive Inventory & Module Mapping

The supplied platform integration archive contains 292 files spanning 8 core domain packages in `src/`, entrypoint scripts in `scripts/`, YAML specifications in `configs/`, and comprehensive unit tests in `tests/`.

### 1.1 Core Domain Packages (`src/`)

| Archive Package | Primary Responsibilities | Target Integration Strategy |
| :--- | :--- | :--- |
| `src/ingestion/` | PDF (PyMuPDF/pypdf), HTML (MathML equation extraction), Markdown, TXT, JSON, JSONL discovery, deduplication, section parsing, deterministic semantic chunking, license/provenance classification. | **Reused as pipeline-core service**. Wrapped in celery tasks (`ingest_documents`, `discover_sources`). Improved JSONL streaming to process multi-line records without loading full files into memory. Pluggable OCR adapter. |
| `src/generation/` | Source analysis, task taxonomy selection, prompt synthesis, equation/calculation grounding, quality evaluation, exact/near deduplication, source-aware train/val/test splits. | **Reused as pipeline-core service**. Wrapped in `generate_candidates` job. Gated by zero-fabrication and rights rules. Preserves candidate rejection logs and reports. |
| `src/dataset/` | Data schemas (`schema.py`), readiness scorecards, scientific QA, rights audit, source leakage prevention, dataset freeze lifecycle. | **Reused as release QA engine**. Enforces mandatory dataset freeze (`70-training-ready/FROZEN`) before training can begin. |
| `src/training/` | QLoRA/SFT configuration, dataset loaders, preflight validation, PyTorch/Transformers/PEFT trainer helpers, checkpoint managers. | **Reused as VM GPU training orchestrator**. Exposes preflight, dry-run, atomic checkpointing, and metrics collection. Config-driven paths; removes hardcoded `/content/drive` paths. |
| `src/evaluation/` | Benchmark datasets, evaluation orchestrator, metrics computation, regression testing. | **Reused as evaluation service**. Wrapped in `evaluate_model` job. |
| `src/distribution/` | Package bundling, export, Hugging Face / Drive sync helpers, secret scanner, checksum verification. | **Reused as storage/distribution service**. Wrapped in `package_download`, `sync_to_drive`, `sync_from_drive` jobs. |
| `src/release/` | Production gate checks, reproducibility records, manifest integrity, versioning. | **Reused as release gate service**. Wrapped in `build_release` job. |
| `src/panel/` | Aiohttp prototype server & static HTML dashboard. | **DEPRECATED**. Replaced by single-admin Django API + Next.js dashboard control plane. Useful UI concepts retained. |

---

## 2. Policy-Sensitive Code Audit & Non-Negotiable Boundaries

### 2.1 Audited Legacy Files

| Legacy Archive File | Policy-Sensitive Issue / Risk | Mandatory VM Platform Behavior |
| :--- | :--- | :--- |
| `scripts/colab_account_manager.py` | Copies `~/.config/colab-cli/token.json` files and performs automatic multi-account rotation on rate limits / quota errors. | **QUARANTINED / DEPRECATED**. Automatic failover and token copy are strictly forbidden. No token-vault or rate-limit bypass. |
| `scripts/run_colab_job.py` | Automates private browser/CLI Colab sessions using copied credentials. | **REPLACED**. Supported Google Cloud / Colab Enterprise REST connector or local VM GPU execution only. |
| `MULTI_ACCOUNT_COLAB_GUIDE.md` | Documents direct Colab CLI token-vault account management. | **REFERENCE**. The dashboard imports credentials explicitly; it does not perform account login or automatic harvesting. |
| `src/panel/server.py` | Unauthenticated subprocess execution, arbitrary path input, permissive CORS (`*`), in-memory jobs. | **REPLACED**. Replaced by single-admin Django API (`/api/v1/jobs/`) with strict typed payload allowlisting and CSRF enforcement. |

### 2.2 Enforced Compliance Policy
1. **Single Local Administrator**: The web panel permits exactly **one** local admin account authenticated by email + password. No public signup, team roles, or TOTP flow.
2. **Direct Google Credential Binding**: Google accounts (`ConnectedAccount`) are external compute/storage connections managed through explicit Colab CLI token import and encrypted credential storage.
3. **Explicit Connection Selection**: Every Google Drive or Colab job must specify an explicitly selected connection. On 401/403/412/429/503 errors, the job is **paused** and recorded; silent account switching is strictly prohibited.

---

## 3. Storage Architecture & Folder Contract

All dataset collections, extracted documents, generated candidates, frozen releases, training runs, and artifacts will follow a strict, deterministic directory structure on the VM disk (`DATA_ROOT=/srv/kaya-data`) and Google Drive (`My Drive/Colab Notebooks/Datasets/CyberSecurity/`):

```
<storage-root>/
  <collection_slug>/
    00-source/           # Source manifest (source-manifest.jsonl, links.jsonl)
    10-raw/              # Raw ingested archives (pdf/, html/, json/, archives/, images/)
    20-extracted/        # Extracted documents (documents.jsonl, sections.jsonl, pages/, ocr/)
    30-normalized/       # Standardized records (normalized.jsonl, equations.jsonl, tables.jsonl)
    40-chunks/           # Deterministic chunks (chunks.jsonl, chunk-manifest.json)
    50-generated/        # Candidate Q&A / instructions (candidates.jsonl, qa.jsonl, instruction.jsonl, rejected.jsonl)
    60-qa/               # QA & audit reports (quality-report.json, rights-report.json, leakage-report.json)
    70-training-ready/   # Frozen training sets (train.jsonl, validation.jsonl, test.jsonl, FROZEN)
    80-training/         # Model runs (runs/<run_id>/ config, checkpoints, logs, metrics)
    90-evaluation/       # Benchmark reports (reports/, predictions/)
    manifests/           # Collection manifests
    reports/             # Summary reports
    logs/                # Task execution logs
```

---

## 4. Compatibility Matrix

### 4.1 CLI Commands to API Job Types

| Old CLI Command | New Typed API Job Type | REST Endpoint |
| :--- | :--- | :--- |
| `scripts/ingest_documents.py` | `ingest_documents` | `POST /api/v1/jobs/ingest` |
| `scripts/generate_instruction_dataset.py` | `generate_candidates` | `POST /api/v1/jobs/generate` |
| `src/dataset/release_qa.py` | `run_quality_audit` | `POST /api/v1/jobs/qa` |
| `scripts/build_dataset_v2.py` | `build_release` / `freeze_dataset` | `POST /api/v1/jobs/release` |
| `scripts/train_production_v2.py` | `train_qlora` | `POST /api/v1/jobs/train` |
| `scripts/monitor_training.py` | Live SSE Stream | `GET /api/v1/events/stream` |
| `scripts/pack_sync_payload.py` | `package_download` / `sync_to_drive` | `POST /api/v1/jobs/sync` |
| `scripts/chat_inference.py` | `run_inference` | `POST /api/v1/jobs/evaluate` |

---

## 5. Dependency Management & Environment Isolation

The platform dependencies are partitioned into explicit, pinned lockfiles in the `requirements/` directory to enforce environment boundaries between VM control plane, local fallback workers, and Colab remote runtimes:

1. **`requirements/vm-base.lock`**: Control plane, Django API, Celery worker queue, Google Drive sync (`pydantic`, `pyyaml`, `jinja2`, `httpx`, `requests`, `aiohttp`, `tqdm`, `django`, `celery`, `redis`).
2. **`requirements/vm-extraction.lock`**: Optional local VM fallback for document parsing (`pymupdf`, `pypdf`, `beautifulsoup4`, `lxml`, `numpy`, `scipy`, `pandas`, `scikit-learn`).
3. **`requirements/vm-training-gpu.lock`**: Optional local VM GPU worker fine-tuning (`torch`, `transformers`, `peft`, `accelerate`, `datasets`, `bitsandbytes`, `trl`).
4. **`requirements/colab-pipeline.lock`**: Full Colab remote pipeline runtime (extraction, generation, QA/release, QLoRA training).
5. **`requirements/colab-training.lock`**: Standalone Colab remote training stage runtime.

---

## 6. Rollback Plan

1. **Database Rollback**: Django database migrations are reversible (`python manage.py migrate <app> <previous_migration>`).
2. **Pipeline Core Isolation**: Core processing logic remains pure Python in `services/pipeline_core/` (adapted from `src/`), keeping CLI wrapper fallback operational.
3. **Artifact Integrity**: All mutations write to temporary paths first before atomic fsync + rename. Incomplete runs retain raw inputs and checkpoints without overwriting existing frozen datasets.
