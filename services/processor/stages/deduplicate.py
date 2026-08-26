import os
import csv
import json
import hashlib
from typing import Dict, Any, Tuple
from services.processor.stages.base import BaseStage, StageValidationError

class DeduplicateStage(BaseStage):
    @property
    def name(self) -> str:
        return "deduplicate"

    @property
    def description(self) -> str:
        return "Deduplicates records using deterministic record fingerprinting."

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params, dict):
            raise StageValidationError("Parameters must be a dictionary.")

    def _fingerprint(self, data: Any) -> str:
        if isinstance(data, dict):
            raw = json.dumps(data, sort_keys=True)
        elif isinstance(data, list):
            raw = json.dumps(data)
        else:
            raw = str(data).strip()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def execute(self, input_path: str, output_dir: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.validate_params(params)
        os.makedirs(output_dir, exist_ok=True)

        dest_filename = os.path.basename(input_path)
        dest_path = os.path.join(output_dir, dest_filename)

        ext = os.path.splitext(input_path)[1].lower()
        seen = set()
        records_total = 0
        records_retained = 0
        records_removed = 0

        if ext == ".csv":
            with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
                 open(dest_path, "w", encoding="utf-8", newline="") as fout:
                reader = csv.reader(fin)
                writer = csv.writer(fout)
                try:
                    header = next(reader)
                    writer.writerow(header)
                except StopIteration:
                    header = None

                for row in reader:
                    records_total += 1
                    fp = self._fingerprint(row)
                    if fp not in seen:
                        seen.add(fp)
                        writer.writerow(row)
                        records_retained += 1
                    else:
                        records_removed += 1
        elif ext in [".jsonl", ".ndjson"]:
            with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
                 open(dest_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    records_total += 1
                    try:
                        obj = json.loads(line_str)
                        fp = self._fingerprint(obj)
                    except Exception:
                        fp = self._fingerprint(line_str)

                    if fp not in seen:
                        seen.add(fp)
                        fout.write(line_str + "\n")
                        records_retained += 1
                    else:
                        records_removed += 1
        else:
            # Fallback line-by-line deduplication
            with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
                 open(dest_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    records_total += 1
                    fp = self._fingerprint(line)
                    if fp not in seen:
                        seen.add(fp)
                        fout.write(line)
                        records_retained += 1
                    else:
                        records_removed += 1

        metrics = {
            "records_total": records_total,
            "records_retained": records_retained,
            "records_removed": records_removed
        }
        return dest_path, metrics
