# Training API App (`apps/training`)

Provides Django REST Framework endpoints for ML training run orchestration, status tracking, metric retrieval, and checkpoint management.

## API Endpoints

- `POST /api/v1/training-runs/`: Enqueue training run.
- `GET /api/v1/training-runs/`: List training runs (filtered by owner or operator role).
- `GET /api/v1/training-runs/<uuid>/`: Retrieve training run details.
- `POST /api/v1/training-runs/<uuid>/cancel/`: Cancel active run.
- `POST /api/v1/training-runs/<uuid>/pause/`: Pause active run.
- `POST /api/v1/training-runs/<uuid>/resume/`: Resume paused run.
- `POST /api/v1/training-runs/<uuid>/retry/`: Retry failed run.
- `GET /api/v1/training-runs/<uuid>/metrics/`: Get scalar training metrics stream.
- `GET /api/v1/training-runs/<uuid>/checkpoints/`: Get recorded training checkpoints.
- `GET /api/v1/training-runs/<uuid>/artifacts/`: Get generated model artifacts and registered versions.
