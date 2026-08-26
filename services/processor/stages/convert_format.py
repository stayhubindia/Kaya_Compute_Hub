import os
import csv
import json
from typing import Dict, Any, Tuple
from services.processor.stages.base import BaseStage, StageValidationError

class ConvertFormatStage(BaseStage):
    @property
    def name(self) -> str:
        return "convert_format"

    @property
    def description(self) -> str:
        return "Converts dataset formats safely (CSV -> JSONL, JSON -> JSONL, JSONL -> CSV)."

    def validate_params(self, params: Dict[str, Any]) -> None:
        target_fmt = params.get("target_format")
        if not target_fmt or target_fmt not in ["jsonl", "csv"]:
            raise StageValidationError("target_format must be 'jsonl' or 'csv'.")

    def execute(self, input_path: str, output_dir: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.validate_params(params)
        os.makedirs(output_dir, exist_ok=True)
        target_fmt = params.get("target_format")

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        dest_filename = f"{base_name}.{target_fmt}"
        dest_path = os.path.join(output_dir, dest_filename)

        src_ext = os.path.splitext(input_path)[1].lower()
        records_converted = 0

        if src_ext == ".csv" and target_fmt == "jsonl":
            with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
                 open(dest_path, "w", encoding="utf-8") as fout:
                reader = csv.DictReader(fin)
                for row in reader:
                    records_converted += 1
                    fout.write(json.dumps(row) + "\n")
        elif src_ext == ".json" and target_fmt == "jsonl":
            with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
                data = json.load(fin)
            with open(dest_path, "w", encoding="utf-8") as fout:
                if isinstance(data, list):
                    for item in data:
                        records_converted += 1
                        fout.write(json.dumps(item) + "\n")
                elif isinstance(data, dict):
                    records_converted += 1
                    fout.write(json.dumps(data) + "\n")
        elif src_ext in [".jsonl", ".ndjson"] and target_fmt == "csv":
            rows = []
            keys = set()
            with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                        keys.update(obj.keys())
            fieldnames = sorted(list(keys))
            with open(dest_path, "w", encoding="utf-8", newline="") as fout:
                writer = csv.DictWriter(fout, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    records_converted += 1
                    writer.writerow(r)
        else:
            raise StageValidationError(f"Conversion from '{src_ext}' to '{target_fmt}' is not supported.")

        metrics = {
            "source_format": src_ext.replace(".", ""),
            "target_format": target_fmt,
            "records_converted": records_converted
        }
        return dest_path, metrics
