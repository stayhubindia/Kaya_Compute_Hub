from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-4B-Base"
LOCAL_DIR = "/content/drive/MyDrive/GoogleColab/AI/Qwen3/models/Qwen3-4B-Base"

print(f"Downloading {MODEL_ID}")
print(f"Destination: {LOCAL_DIR}")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
)

print("Download complete.")
