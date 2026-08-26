from pathlib import Path

p = Path("/content/drive/MyDrive/GoogleColab/AI/Qwen3/models/Qwen3-4B-Base")

print("Path exists:", p.exists())

if p.exists():
    files = list(p.rglob("*"))
    files = [f for f in files if f.is_file()]

    total = sum(f.stat().st_size for f in files)

    print("Files:", len(files))
    print("Total size:", round(total / (1024**3), 2), "GB")

    print("\nFiles:")
    for f in files:
        print(f"{f.stat().st_size / (1024**2):8.1f} MB  {f.name}")
