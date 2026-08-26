from pathlib import Path
import json

base_dir = Path("/content/drive/MyDrive/GoogleColab/AI/Qwen3")
print("Base directory exists:", base_dir.exists())

model_path = base_dir / "models/Qwen3-4B-Base"
print("Model path exists:", model_path.exists())

if model_path.exists():
    files = sorted([f for f in model_path.iterdir() if f.is_file()])
    print(f"Total files in model dir: {len(files)}")
    total_size = sum(f.stat().st_size for f in files)
    print(f"Total model size: {total_size / (1024**3):.2f} GB ({total_size / (1024**2):.1f} MB)")
    for f in files:
        print(f"  {f.name:35s} {f.stat().st_size / (1024**2):8.2f} MB")
else:
    print("Listing parent models directory:")
    models_parent = base_dir / "models"
    if models_parent.exists():
        for item in models_parent.iterdir():
            print(f"  {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
    else:
        print("Models parent directory does not exist.")
