import os
import csv
import json
import random
from typing import Dict, Any, Tuple
from services.processor.stages.base import BaseStage, StageValidationError

class SplitDatasetStage(BaseStage):
    @property
    def name(self) -> str:
        return "split_dataset"

    @property
    def description(self) -> str:
        return "Splits dataset into train/val/test splits deterministically with a random seed."

    def validate_params(self, params: Dict[str, Any]) -> None:
        train_ratio = params.get("train_ratio", 0.8)
        val_ratio = params.get("val_ratio", 0.1)
        test_ratio = params.get("test_ratio", 0.1)

        if not (0 <= train_ratio <= 1.0 and 0 <= val_ratio <= 1.0 and 0 <= test_ratio <= 1.0):
            raise StageValidationError("Split ratios must be between 0.0 and 1.0.")
        if round(train_ratio + val_ratio + test_ratio, 4) != 1.0:
            raise StageValidationError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    def execute(self, input_path: str, output_dir: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.validate_params(params)
        os.makedirs(output_dir, exist_ok=True)

        seed = params.get("seed", 42)
        train_ratio = params.get("train_ratio", 0.8)
        val_ratio = params.get("val_ratio", 0.1)

        ext = os.path.splitext(input_path)[1].lower()
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        rows = []
        header = None

        if ext == ".csv":
            with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
                reader = csv.reader(fin)
                try:
                    header = next(reader)
                    rows = list(reader)
                except StopIteration:
                    pass
        elif ext in [".jsonl", ".ndjson"]:
            with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
                rows = [line.strip() for line in fin if line.strip()]

        # Shuffle deterministically
        rng = random.Random(seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)

        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_data = shuffled[:train_end]
        val_data = shuffled[train_end:val_end]
        test_data = shuffled[val_end:]

        def write_split(suffix: str, data: list):
            out_file = os.path.join(output_dir, f"{base_name}_{suffix}{ext}")
            if ext == ".csv":
                with open(out_file, "w", encoding="utf-8", newline="") as fout:
                    writer = csv.writer(fout)
                    if header:
                        writer.writerow(header)
                    writer.writerows(data)
            else:
                with open(out_file, "w", encoding="utf-8") as fout:
                    for line in data:
                        fout.write(line + "\n")
            return out_file

        write_split("train", train_data)
        write_split("val", val_data)
        write_split("test", test_data)

        metrics = {
            "train_count": len(train_data),
            "val_count": len(val_data),
            "test_count": len(test_data),
            "total_count": total,
            "seed": seed
        }
        return output_dir, metrics
