# Dataset Processing Pipelines API (`apps.pipelines`)

The **Pipelines App** exposes REST API endpoints for defining dataset processing pipelines and triggering asynchronous background processing runs.

## Endpoints Summary

| Method | Endpoint | Description | Permission |
|---|---|---|---|
| `POST` | `/api/v1/pipelines/` | Create a pipeline definition (with stage validation). | Authenticated |
| `GET` | `/api/v1/pipelines/` | List pipeline definitions. | Authenticated |
| `GET` | `/api/v1/pipelines/<uuid>/` | Retrieve pipeline definition details. | Authenticated |
| `POST` | `/api/v1/processing-runs/` | Enqueue a pipeline processing run. | Authenticated |
| `GET` | `/api/v1/processing-runs/` | List processing runs (filtered by RBAC). | Authenticated |
| `GET` | `/api/v1/processing-runs/<uuid>/` | Retrieve processing run status. | Authenticated |
| `POST` | `/api/v1/processing-runs/<uuid>/cancel/` | Cancel processing run. | Owner / Operator |
| `POST` | `/api/v1/processing-runs/<uuid>/pause/` | Pause processing run. | Owner / Operator |
| `POST` | `/api/v1/processing-runs/<uuid>/resume/` | Resume paused/failed processing run. | Owner / Operator |
| `GET` | `/api/v1/processing-runs/<uuid>/stages/` | List per-stage execution events and metrics. | Owner / Operator |
| `GET` | `/api/v1/processing-runs/<uuid>/artifacts/` | Retrieve output dataset and manifest. | Owner / Operator |

## Example Creation Payload

```json
{
  "name": "NLP Preprocessing Pipeline",
  "description": "Standard cleaning, deduplication, and train/val/test splitting",
  "version": "1.0.0",
  "stages": [
    { "name": "validate_files", "params": {} },
    { "name": "inspect_schema", "params": {} },
    { "name": "normalize_text", "params": { "remove_control_chars": true } },
    { "name": "deduplicate", "params": {} },
    { "name": "split_dataset", "params": { "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1, "seed": 42 } },
    { "name": "generate_statistics", "params": {} }
  ],
  "resource_policy": {
    "max_cpu_cores": 2.0,
    "max_memory_mb": 4096,
    "run_as_non_root": true
  }
}
```
