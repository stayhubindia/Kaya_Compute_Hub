from pathlib import Path

BASE = Path("/content/drive/MyDrive/GoogleColab/AI/Qwen3")

for folder in [
    "models",
    "datasets/raw",
    "datasets/processed",
    "checkpoints",
    "outputs",
]:
    path = BASE / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {path}")
