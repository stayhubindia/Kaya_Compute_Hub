import json
from pathlib import Path

rel_dir = Path("/content/drive/MyDrive/GoogleColab/AI/Qwen3/releases/qwen3-4b-qlora-v1.0")
files = {}
for p in rel_dir.rglob("*"):
    if p.is_file() and not p.name.endswith(".pt") and not p.name.endswith(".safetensors"):
        files[p.relative_to(rel_dir).as_posix()] = p.read_text(encoding="utf-8")

print("JSON_START")
print(json.dumps(files))
print("JSON_END")
