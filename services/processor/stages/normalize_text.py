import os
import unicodedata
import csv
from typing import Dict, Any, Tuple
from services.processor.stages.base import BaseStage, StageValidationError

class NormalizeTextStage(BaseStage):
    @property
    def name(self) -> str:
        return "normalize_text"

    @property
    def description(self) -> str:
        return "Normalizes Unicode, line endings, and strips control characters."

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params, dict):
            raise StageValidationError("Parameters must be a dictionary.")

    def _normalize_string(self, text: str, remove_control_chars: bool) -> Tuple[str, bool]:
        if not text:
            return text, False
        norm = unicodedata.normalize("NFC", text)
        norm = norm.replace("\r\n", "\n").replace("\r", "\n")
        if remove_control_chars:
            norm = "".join(ch for ch in norm if unicodedata.category(ch)[0] != "C" or ch in ["\n", "\t"])
        return norm, (norm != text)

    def execute(self, input_path: str, output_dir: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.validate_params(params)
        os.makedirs(output_dir, exist_ok=True)
        remove_cc = params.get("remove_control_chars", True)

        dest_filename = os.path.basename(input_path)
        dest_path = os.path.join(output_dir, dest_filename)

        records_processed = 0
        records_changed = 0

        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".csv":
            with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
                 open(dest_path, "w", encoding="utf-8", newline="") as fout:
                reader = csv.reader(fin)
                writer = csv.writer(fout)
                for row in reader:
                    records_processed += 1
                    new_row = []
                    row_changed = False
                    for cell in row:
                        norm_cell, changed = self._normalize_string(cell, remove_cc)
                        if changed:
                            row_changed = True
                        new_row.append(norm_cell)
                    if row_changed:
                        records_changed += 1
                    writer.writerow(new_row)
        else:
            with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
                 open(dest_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    records_processed += 1
                    norm_line, changed = self._normalize_string(line, remove_cc)
                    if changed:
                        records_changed += 1
                    fout.write(norm_line)

        metrics = {
            "records_processed": records_processed,
            "records_changed": records_changed
        }
        return dest_path, metrics
