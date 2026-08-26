import os
import json
import hashlib
from typing import Dict, Any, Tuple
from apps.training.models import TrainingCheckpoint

class TrainingCheckpointManager:
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.environ.get("TRAINING_CHECKPOINT_ROOT", "storage/checkpoints/training")

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

    def create_checkpoint(
        self,
        training_run_id: str,
        epoch: int,
        step: int,
        checkpoint_data: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Tuple[str, TrainingCheckpoint]:

        run_dir = os.path.join(self.base_dir, str(training_run_id))
        os.makedirs(run_dir, exist_ok=True)

        ckpt_file_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch}_step_{step}.json")
        temp_file_path = f"{ckpt_file_path}.tmp"

        save_payload = {
            "training_run_id": str(training_run_id),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "checkpoint_data": checkpoint_data
        }

        with open(temp_file_path, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, indent=2)

        os.replace(temp_file_path, ckpt_file_path)

        checksum = self._calculate_checksum(ckpt_file_path)
        size_bytes = os.path.getsize(ckpt_file_path)

        ckpt_record = TrainingCheckpoint.objects.create(
            training_run_id=training_run_id,
            epoch=epoch,
            step=step,
            storage_uri=ckpt_file_path,
            checksum=checksum,
            size_bytes=size_bytes,
            metrics=metrics,
            status="valid"
        )

        return ckpt_file_path, ckpt_record
