# Backend Services (`/services`)

This directory houses the core backend microservices and workload engines:

- **`api/`**: Django REST API server handling authentication, database models (`accounts`, `jobs`, `workers`, `datasets`, `artifacts`, `audit`), and REST endpoints.
- **`worker/`**: Celery worker engine managing job execution, Docker container sandboxing, task leases, and heartbeats.
- **`downloader/`**: Data ingestion service with providers for HTTP, S3, FTP, HuggingFace, and direct-token Google Drive downloads.
- **`trainer/`**: Model training orchestration framework with backends for PyTorch, Transformers, and custom model execution.
