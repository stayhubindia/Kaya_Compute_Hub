import tarfile
import os
import sys
from pathlib import Path

print("Extracting /content/project_v2_bundle.tar.gz...")
with tarfile.open("/content/project_v2_bundle.tar.gz", "r:gz") as tar:
    tar.extractall("/content")
print("Extraction complete. Listing /content:")
for p in sorted(Path("/content").iterdir()):
    print(f"  {p.name} ({'DIR' if p.is_dir() else 'FILE'})")

print("\nChecking Python package versions:")
packages = ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "pydantic", "yaml", "datasets"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f"  {pkg:15s}: {getattr(mod, '__version__', 'available')}")
    except ImportError:
        print(f"  {pkg:15s}: NOT INSTALLED")
