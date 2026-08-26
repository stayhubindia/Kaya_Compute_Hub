# Standalone Trainer Subsystem (`services/trainer/`)

The Trainer subsystem provides secure, resource-aware machine learning model training orchestration and metric tracking.

## Architecture

- **`registry.py`**: Trainer backend registry (`demo`, `pytorch`).
- **`policies.py`**: Training configuration hyperparameter validator and container resource boundaries.
- **`scheduler.py`**: Resource matching engine evaluating GPU capacity, worker status, and locking slots atomically.
- **`checkpoints/`**:
  - `manager.py`: Atomic stage checkpoint creation with SHA-256 integrity validation.
  - `recovery.py`: Verified recovery engine for resuming interrupted training runs.
- **`metrics/`**:
  - `collector.py`: Scalar metrics collector & best metric tracking (`val_loss`, `accuracy`).
  - `schemas.py`: Metric Record data validation.
- **`containers/`**:
  - `image_registry.py`: Strict allowlist of approved ML training images (`kaya/ml-trainer:pytorch-2.2`).

## Execution Flow

1. Training run requested via API (`POST /api/v1/training-runs/`).
2. Celery task `execute_training_run` acquires worker GPU/CPU capacity.
3. Trainer backend executes validated configuration and streams metrics.
4. Checkpoints are atomically saved with SHA-256 checksums.
5. Model artifact is created and registered in `ModelVersion` registry (`REGISTERED` status).
6. Worker capacity is released atomically upon completion or failure.
