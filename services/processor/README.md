# Safe Dataset Processor Subsystem (`services/processor`)

The **Standalone Processor Subsystem** provides safe, reproducible, checkpointed dataset preprocessing pipelines for Kaya Compute Hub.

## Architectural Architecture

```
Source Dataset (Immutable)
        │
        ▼
   Pipeline Definition (Allowlisted Stages)
        │
        ▼
  Stage Execution Engine & Checkpoint Manager
  (validate_files -> inspect_schema -> normalize_text -> deduplicate -> split_dataset -> generate_statistics)
        │
        ▼
   Derived Output Dataset + Dataset Manifest (JSON)
```

## Allowlisted Processing Stages

| Stage Name | Description | Key Parameters |
|---|---|---|
| `validate_files` | Verifies file existence, readability, non-emptiness, and optional checksums. | `{}` |
| `inspect_schema` | Detects format (CSV/JSONL/JSON), column headers, record structure, and malformed rows. | `{}` |
| `normalize_text` | Normalizes Unicode (NFC), line endings (`\n`), and removes control characters. | `{"remove_control_chars": true}` |
| `convert_format` | Safely converts formats (CSV $\rightarrow$ JSONL, JSON $\rightarrow$ JSONL, JSONL $\rightarrow$ CSV). | `{"target_format": "jsonl"}` |
| `deduplicate` | Purges duplicate records using SHA-256 fingerprinting. | `{}` |
| `split_dataset` | Splitting into train/val/test splits deterministically with a random seed. | `{"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1, "seed": 42}` |
| `generate_statistics` | Calculates dataset metrics (row counts, file sizes, missing values, duplicates). | `{}` |

## Security & Container Isolation Rules

- **No Arbitrary Code Execution**: User-supplied Python strings, shell commands, or unapproved stage parameters are strictly rejected during pipeline validation.
- **Approved Images Only**: Container execution requires images in `APPROVED_IMAGES` (`kaya/dataset-processor:latest`).
- **Resource Limits**: Enforces non-root container users, CPU limits (max 16 cores), memory limits (max 64GB), and execution timeouts.
- **Dataset Immutability**: Source datasets are never overwritten or mutated. Output datasets are produced as new derived entities linked via `parent_dataset` lineage.

## Checkpoint & Failure Recovery

- Atomically saves intermediate stage outputs and metrics under `storage/checkpoints/<run_id>/<stage_name>.json`.
- Restarts cleanly resume from the last completed stage checkpoint without re-running verified stages.
- Corrupted checkpoints (checksum mismatches or missing files) are automatically invalidated and re-executed.
