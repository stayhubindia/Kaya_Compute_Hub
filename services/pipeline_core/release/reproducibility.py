"""
Reproducibility Metadata & Audit Framework (Phase 5.1).
Records exact seeds, configurations, library versions, hardware constraints,
and CLI reproduction recipes to ensure end-to-end determinism.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from src.release.provenance import DatasetProvenance, HardwareProvenance, TrainingProvenance

logger = logging.getLogger(__name__)


class ReproducibilityRecord(BaseModel):
    """Immutable reproducibility specification for the release."""
    schema_version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    random_seed: int = 42
    dataset_version: str = "dataset-v1.0"
    dataset_manifest_sha256: str = ""
    training_config_hash: str = ""
    generation_config_hash: str = ""
    environment_versions: Dict[str, str] = Field(default_factory=dict)
    hardware_specification: Dict[str, Any] = Field(default_factory=dict)
    training_hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    quantization_configuration: Dict[str, Any] = Field(default_factory=dict)
    lora_configuration: Dict[str, Any] = Field(default_factory=dict)
    training_reproduction_command: str = "python scripts/train_qwen.py --config configs/training.yaml"
    evaluation_reproduction_command: str = "python scripts/run_evaluation.py --model adapter --benchmark benchmark-v1.0"
    reproducibility_notes: str = (
        "Determinism guaranteed for input tokens and pipeline parameters under identical "
        "CUDA kernel implementations and BitsAndBytes NF4 quantizer state."
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def save(self, path: Union[str, Path]) -> None:
        """Persist reproducibility record as formatted JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)


class ReproducibilityManager:
    """Constructs complete reproducibility records from runtime components and configs."""

    @staticmethod
    def build_record(
        dataset_prov: DatasetProvenance,
        training_prov: TrainingProvenance,
        hardware_prov: HardwareProvenance,
        generation_config_hash: str = "",
    ) -> ReproducibilityRecord:
        """Assemble comprehensive reproducibility record."""
        import torch
        import transformers
        import peft
        import bitsandbytes

        env_versions = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "platform": platform.platform(),
        }

        train_hypers = {
            "optimizer": training_prov.optimizer,
            "scheduler": training_prov.scheduler,
            "learning_rate": training_prov.learning_rate,
            "warmup_ratio": training_prov.warmup_ratio,
            "per_device_train_batch_size": training_prov.per_device_train_batch_size,
            "gradient_accumulation_steps": training_prov.gradient_accumulation_steps,
            "effective_batch_size": training_prov.effective_batch_size,
            "num_train_epochs": training_prov.num_train_epochs,
            "max_length": training_prov.max_length,
            "assistant_only_loss": training_prov.assistant_only_loss,
        }

        quant_cfg = {
            "quant_type": training_prov.quantization_type,
            "use_double_quant": training_prov.use_double_quant,
            "load_in_4bit": True,
        }

        return ReproducibilityRecord(
            random_seed=training_prov.seed,
            dataset_version=dataset_prov.dataset_version,
            dataset_manifest_sha256=dataset_prov.manifest_sha256,
            training_config_hash=training_prov.config_hash,
            generation_config_hash=generation_config_hash,
            environment_versions=env_versions,
            hardware_specification=hardware_prov.to_dict(),
            training_hyperparameters=train_hypers,
            quantization_configuration=quant_cfg,
            lora_configuration=training_prov.qlora_params,
        )
