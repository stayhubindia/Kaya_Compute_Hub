import os
from pathlib import Path

root = Path("/content/outputs")
print(f"Checking {root} (Exists: {root.exists()})")
if root.exists():
    for p in root.glob("**/*"):
        if p.is_file():
            print(f"File: {p} ({p.stat().st_size} bytes)")
