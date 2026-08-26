import os
from pathlib import Path

ckpt_dir = Path("/content/outputs/training/dataset-v2.0/qlora-v2/production/checkpoints")
print(f"Checking directory: {ckpt_dir} (Exists: {ckpt_dir.exists()})")
if ckpt_dir.exists():
    for p in sorted(ckpt_dir.glob("*")):
        print(f" - {p.name} (is_dir: {p.is_dir()})")
