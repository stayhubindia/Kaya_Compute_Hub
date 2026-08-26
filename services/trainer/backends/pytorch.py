import os
import json
from typing import Dict, Any, Tuple, Optional, Callable
from services.trainer.backends.base import BaseTrainerBackend
from services.trainer.containers.image_registry import is_approved_training_image

class PyTorchTrainerBackend(BaseTrainerBackend):
    def train(
        self,
        run_id: str,
        dataset_uri: str,
        output_dir: str,
        configuration: Dict[str, Any],
        resume_from: Optional[Tuple[int, int, Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Tuple[str, Dict[str, Any]]:

        image_name = configuration.get("container_image", "kaya/ml-trainer:pytorch-2.2")
        if not is_approved_training_image(image_name):
            raise ValueError(f"Unapproved container image '{image_name}' for PyTorch training.")

        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "pytorch_model.bin")

        with open(model_path, "w", encoding="utf-8") as f:
            json.dump({
                "framework": "pytorch",
                "framework_version": "2.2.0",
                "status": "simulated_pytorch_execution",
                "run_id": run_id
            }, f, indent=2)

        return model_path, {"final_val_loss": 0.25, "final_accuracy": 0.95}
