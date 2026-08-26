import os
from pathlib import Path

print("Checking /content/drive existence...")
drive_path = Path("/content/drive")
print(f"/content/drive exists: {drive_path.exists()}")

if not (drive_path / "MyDrive").exists():
    print("Attempting to mount Google Drive via google.colab.drive...")
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=True)
        print("Drive mount called successfully.")
    except Exception as e:
        print("Drive mount exception:", e)

print(f"Post-mount check: /content/drive/MyDrive exists: {(drive_path / 'MyDrive').exists()}")
if (drive_path / "MyDrive").exists():
    qwen_dir = drive_path / "MyDrive/GoogleColab/AI/Qwen3"
    print(f"Qwen directory ({qwen_dir}) exists: {qwen_dir.exists()}")
    if qwen_dir.exists():
        for item in sorted(qwen_dir.iterdir()):
            print(f"  {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
