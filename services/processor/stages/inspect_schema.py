import os
import csv
import json
import shutil
from typing import Dict, Any, Tuple
from services.processor.stages.base import BaseStage, StageValidationError

class InspectSchemaStage(BaseStage):
    @property
    def name(self) -> str:
        return "inspect_schema"

    @property
    def description(self) -> str:
        return "Inspects dataset format, columns, record structure, and malformed rows."

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params, dict):
            raise StageValidationError("Parameters must be a dictionary.")

    def execute(self, input_path: str, output_dir: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.validate_params(params)
        os.makedirs(output_dir, exist_ok=True)

        target_file = input_path
        if os.path.isdir(input_path):
            files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
            if not files:
                raise StageValidationError("No files found in directory for schema inspection.")
            target_file = files[0]

        ext = os.path.splitext(target_file)[1].lower()
        format_detected = "unknown"
        columns = []
        row_count = 0
        malformed_rows = 0

        if ext == ".csv":
            format_detected = "csv"
            with open(target_file, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                    columns = header
                    for row in reader:
                        row_count += 1
                        if len(row) != len(header):
                            malformed_rows += 1
                except StopIteration:
                    pass
        elif ext in [".jsonl", ".ndjson"]:
            format_detected = "jsonl"
            col_set = set()
            with open(target_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row_count += 1
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            col_set.update(obj.keys())
                    except Exception:
                        malformed_rows += 1
            columns = sorted(list(col_set))
        elif ext == ".json":
            format_detected = "json"
            with open(target_file, "r", encoding="utf-8", errors="replace") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        row_count = len(data)
                        col_set = set()
                        for item in data:
                            if isinstance(item, dict):
                                col_set.update(item.keys())
                        columns = sorted(list(col_set))
                    elif isinstance(data, dict):
                        row_count = 1
                        columns = sorted(list(data.keys()))
                except Exception:
                    malformed_rows += 1

        dest_filename = os.path.basename(input_path)
        dest_path = os.path.join(output_dir, dest_filename)
        if os.path.isfile(input_path):
            shutil.copy2(input_path, dest_path)
        else:
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(input_path, dest_path)

        metrics = {
            "format": format_detected,
            "row_count": row_count,
            "columns": columns,
            "column_count": len(columns),
            "malformed_rows": malformed_rows
        }
        return dest_path, metrics
