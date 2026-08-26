# Colab Enterprise Connector Package

This package provides a secure REST/gcloud-compatible connector for executing managed notebooks on Google Cloud Colab Enterprise (Vertex AI Workbench/Notebooks).

## Architectural Guarantees

1. **Explicit Allowlisting**: Projects and regions must be explicitly configured in `GOOGLE_ALLOWED_PROJECTS` and `GOOGLE_ALLOWED_REGIONS`. Unlisted projects or regions are rejected immediately.
2. **No Arbitrary Code Execution**: Execution is restricted to managed notebook resource paths registered in `ExternalNotebook`. Free Google Colab browser sessions are never automated.
3. **Execution Monitoring**: Asynchronous Celery polling monitors execution state (`requested` -> `submitted` -> `running` -> `completed` / `failed` / `cancelled` / `timed_out`) and imports GCS output artifacts.

## Configuration Variables

- `GOOGLE_ALLOWED_PROJECTS`: Comma-separated GCP Project IDs.
- `GOOGLE_ALLOWED_REGIONS`: Comma-separated GCP regions (`us-central1,us-east4,europe-west1`).
- `COLAB_ENTERPRISE_ENABLED`: Toggle flag (`True`/`False`).
- `COLAB_ENTERPRISE_DEFAULT_OUTPUT_BUCKET`: GCS output bucket URI (`gs://my-colab-outputs`).
