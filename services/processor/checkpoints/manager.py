import os
import json
import hashlib
from typing import Dict, Any, Optional, Tuple

CHECKPOINT_DIR_BASE = os.environ.get("CHECKPOINT_STORAGE_ROOT", "storage/checkpoints")

class CheckpointManager:
    def __init__(self, base_dir: str = CHECKPOINT_DIR_BASE):
        self.base_dir = base_dir

    def _get_run_dir(self, run_id: str) -> str:
        d = os.path.join(self.base_dir, str(run_id))
        os.makedirs(d, exist_ok=True)
        return d

    def _calculate_checksum(self, filepath: str) -> str:
        hasher = hashlib.sha256()
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    hasher.update(chunk)
        elif os.path.isdir(filepath):
            for root, _, files in os.walk(filepath):
                for file in sorted(files):
                    fp = os.path.join(root, file)
                    with open(fp, 'rb') as f:
                        for chunk in iter(lambda: f.read(65536), b''):
                            hasher.update(chunk)
        return hasher.hexdigest()

    def save_checkpoint(self, run_id: str, stage_name: str, output_path: str, metrics: Dict[str, Any]) -> str:
        run_dir = self._get_run_dir(run_id)
        ckpt_file = os.path.join(run_dir, f"{stage_name}.json")

        checksum = self._calculate_checksum(output_path) if os.path.exists(output_path) else ""

        data = {
            "run_id": str(run_id),
            "stage_name": stage_name,
            "output_path": output_path,
            "checksum": checksum,
            "metrics": metrics,
            "valid": True
        }

        temp_ckpt = f"{ckpt_file}.tmp"
        with open(temp_ckpt, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_ckpt, ckpt_file)
        return ckpt_file

    def get_checkpoint(self, run_id: str, stage_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        run_dir = self._get_run_dir(run_id)
        ckpt_file = os.path.join(run_dir, f"{stage_name}.json")

        if not os.path.exists(ckpt_file):
            return None

        try:
            with open(ckpt_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            out_path = data.get("output_path")
            expected_checksum = data.get("checksum")

            if not out_path or not os.path.exists(out_path):
                return None

            if expected_checksum:
                actual_checksum = self._calculate_checksum(out_path)
                if actual_checksum != expected_checksum:
                    # Corrupted output file! Reject checkpoint.
                    return None

            return out_path, data.get("metrics", {})
        except Exception:
            return None
