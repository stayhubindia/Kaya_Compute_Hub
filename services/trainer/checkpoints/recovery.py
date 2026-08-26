import os
import json
import hashlib
from typing import Optional, Tuple, Dict, Any
from apps.training.models import TrainingCheckpoint

class CheckpointRecoveryError(Exception):
    pass

class CheckpointRecoveryManager:
    @staticmethod
    def _calculate_checksum(filepath: str) -> str:
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def get_latest_valid_checkpoint(cls, training_run_id: str) -> Optional[Tuple[int, int, Dict[str, Any], Dict[str, Any]]]:
        checkpoints = TrainingCheckpoint.objects.filter(
            training_run_id=training_run_id,
            status="valid"
        ).order_by('-epoch', '-step')

        for ckpt in checkpoints:
            if not os.path.exists(ckpt.storage_uri):
                ckpt.status = "missing"
                ckpt.save(update_fields=['status'])
                continue

            # Verify Checksum
            actual_checksum = cls._calculate_checksum(ckpt.storage_uri)
            if actual_checksum != ckpt.checksum:
                ckpt.status = "corrupted"
                ckpt.save(update_fields=['status'])
                continue

            try:
                with open(ckpt.storage_uri, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                ckpt_data = payload.get("checkpoint_data", {})
                metrics = payload.get("metrics", {})
                return ckpt.epoch, ckpt.step, ckpt_data, metrics
            except Exception:
                ckpt.status = "corrupted"
                ckpt.save(update_fields=['status'])
                continue

        return None
