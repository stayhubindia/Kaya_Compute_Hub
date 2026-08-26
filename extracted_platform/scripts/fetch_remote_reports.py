import json
from pathlib import Path

reports = {}
for p in Path("reports").glob("*"):
    if p.is_file():
        reports[p.name] = p.read_text()

out_dir = Path("/content/drive/MyDrive/GoogleColab/AI/Qwen3/training/dataset-v1.0/qlora-v1")
for p in ["training_completion_manifest.json", "training_run_manifest.json"]:
    fp = out_dir / p
    if fp.exists():
        reports[p] = fp.read_text()

print("JSON_START")
print(json.dumps(reports))
print("JSON_END")
