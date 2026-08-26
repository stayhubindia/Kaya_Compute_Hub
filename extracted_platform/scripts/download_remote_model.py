import time
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = Path("/content/drive/MyDrive/GoogleColab/AI/Qwen3/models/Qwen3-4B-Base")
model_dir.mkdir(parents=True, exist_ok=True)

print(f"Target model directory: {model_dir}")
print("Starting download of Qwen/Qwen3-4B-Base...")
t0 = time.perf_counter()
snapshot_download(
    repo_id="Qwen/Qwen3-4B-Base",
    local_dir=str(model_dir),
)
t1 = time.perf_counter()
print(f"Download complete in {t1 - t0:.2f} seconds!")

print("Files in model dir:")
for f in sorted(model_dir.iterdir()):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
