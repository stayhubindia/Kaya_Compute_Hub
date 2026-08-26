import os
import json
import time
from typing import Dict, Any, Tuple, Optional, Callable
from services.trainer.backends.base import BaseTrainerBackend
from services.trainer.checkpoints.manager import TrainingCheckpointManager
from services.trainer.metrics.collector import MetricCollector
from services.trainer.metrics.schemas import MetricRecord

class DemoTrainerBackend(BaseTrainerBackend):
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

        os.makedirs(output_dir, exist_ok=True)
        ckpt_mgr = TrainingCheckpointManager()
        collector = MetricCollector(
            training_run_id=run_id,
            target_metric_name=configuration.get("metric_name", "val_loss"),
            metric_direction=configuration.get("metric_direction", "minimize")
        )

        total_epochs = configuration.get("epochs", 3)
        start_epoch = 1
        start_step = 1

        if resume_from:
            start_epoch = resume_from[0] + 1
            start_step = resume_from[1] + 1

        current_step = start_step - 1
        initial_loss = 2.5
        lr = configuration.get("learning_rate", 0.001)
        val_loss = 0.5
        accuracy = 0.90

        for epoch in range(start_epoch, total_epochs + 1):
            if cancel_check and cancel_check():
                raise InterruptedError("Training run cancelled by user.")

            current_step += 10
            # Simulating loss reduction
            decay_factor = (0.75 ** epoch)
            train_loss = round(initial_loss * decay_factor, 4)
            val_loss = round(train_loss * 1.05, 4)
            accuracy = round(min(0.99, 0.50 + (epoch * 0.15)), 4)

            # Record metrics via MetricCollector
            metrics_records = [
                MetricRecord(step=current_step, epoch=epoch, name="loss", value=train_loss, split="train"),
                MetricRecord(step=current_step, epoch=epoch, name="val_loss", value=val_loss, split="val"),
                MetricRecord(step=current_step, epoch=epoch, name="accuracy", value=accuracy, split="val"),
                MetricRecord(step=current_step, epoch=epoch, name="learning_rate", value=lr, split="train"),
            ]
            collector.record_metrics(metrics_records)

            # Save epoch checkpoint
            ckpt_mgr.create_checkpoint(
                training_run_id=run_id,
                epoch=epoch,
                step=current_step,
                checkpoint_data={"epoch": epoch, "weights_summary": f"demo_weights_epoch_{epoch}"},
                metrics={"loss": train_loss, "val_loss": val_loss, "accuracy": accuracy}
            )

            if progress_callback:
                progress_callback(epoch, current_step, {"loss": train_loss, "val_loss": val_loss, "accuracy": accuracy})

        # Save final output model artifact
        model_artifact_path = os.path.join(output_dir, "model.bin")
        model_meta = {
            "model_type": configuration.get("model_name", "demo_model"),
            "backend": "demo",
            "epochs": total_epochs,
            "final_accuracy": accuracy,
            "final_val_loss": val_loss
        }
        with open(model_artifact_path, "w", encoding="utf-8") as f:
            json.dump(model_meta, f, indent=2)

        return model_artifact_path, {
            "epochs": total_epochs,
            "steps": current_step,
            "final_val_loss": val_loss,
            "final_accuracy": accuracy
        }
