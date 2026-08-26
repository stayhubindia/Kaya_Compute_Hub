# Persistent Storage Volume (`/storage`)

This directory serves as the non-volatile filesystem mount for system runtime assets:

- **`datasets/`**: Raw, ingested, and preprocessed datasets.
- **`artifacts/`**: Final job outputs, exported packages, evaluation reports, and model binaries.
- **`checkpoints/`**: Training checkpoints and state files (`.pt`, `.safetensors`).
- **`logs/`**: Per-job execution logs (`stdout.log`, `stderr.log`).
- **`tmp/`**: Temporary scratch space for working files.

> **Note**: All contents inside `/storage/*` (except `.gitkeep` files) are excluded from git commits via `.gitignore`.
