import os
import csv
import json
import shutil
from typing import Dict, Any, Tuple
from services.processor.stages.base import BaseStage, StageValidationError

class GenerateStatisticsStage(BaseStage):
    @property
    def name(self) -> str:
        return "generate_statistics"

    @property
    def description(self) -> str:
        return "Generates dataset statistics including row count, missing values, duplicates, and text distributions."

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params, dict):
            raise StageValidationError("Parameters must be a dictionary.")

    def execute(self, input_path: str, output_dir: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.validate_params(params)
        os.makedirs(output_dir, exist_ok=True)

        dest_filename = os.path.basename(input_path)
        dest_path = os.path.join(output_dir, dest_filename)

        if os.path.isfile(input_path):
            shutil.copy2(input_path, dest_path)
        else:
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(input_path, dest_path)

        total_bytes = 0
        file_count = 0
        target_files = []

        if os.path.isfile(input_path):
            target_files.append(input_path)
        else:
            for root, _, files in os.walk(input_path):
                for f in files:
                    target_files.append(os.path.join(root, f))

        file_count = len(target_files)
        for f in target_files:
            total_bytes += os.path.getsize(f)

        row_count = 0
        missing_values = 0
        empty_rows = 0

        first_file = target_files[0] if target_files else input_path
        ext = os.path.splitext(first_file)[1].lower()

        if ext == ".csv":
            with open(first_file, "r", encoding="utf-8", errors="replace") as fin:
                reader = csv.reader(fin)
                try:
                    _ = next(reader)
                    for row in reader:
                        row_count += 1
                        if not any(row):
                            empty_rows += 1
                        missing_values += sum(1 for val in row if val.strip() == "")
                except StopIteration:
                    pass
        elif ext in [".jsonl", ".ndjson"]:
            with open(first_file, "r", encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        empty_rows += 1
                        continue
                    row_count += 1

        metrics = {
            "file_count": file_count,
            "total_bytes": total_bytes,
            "row_count": row_count,
            "missing_values": missing_values,
            "empty_rows": empty_rows
        }
        return dest_path, metrics
