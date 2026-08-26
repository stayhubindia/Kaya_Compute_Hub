"""
Provenance Collection Engine (Phase 5.1).
Collects, validates, and records dataset provenance, training configuration provenance,
and hardware execution environment provenance for complete reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field

from src.training.utils import compute_file_sha256, detect_hardware_environment

logger = logging.getLogger(__name__)


class DatasetProvenance(BaseModel):
    """Immutable provenance record for the fine-tuning training dataset."""
    dataset_version: str = "dataset-v1.0"
    lifecycle_status: str = "FROZEN"
    manifest_path: str = ""
    manifest_sha256: str = ""
    target_count: int = 0
    actual_final_count: int = 0
    split_hashes: Dict[str, str] = Field(default_factory=dict)
    provenance_status: str = "UNVERIFIED"  # 'VERIFIED' or 'UNVERIFIED'

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class TrainingProvenance(BaseModel):
    """Exact hyperparameters and settings used to reproduce QLoRA training."""
    config_hash: str = ""
    config_path: str = "configs/training.yaml"
    base_model_name: str = "Qwen/Qwen3-4B-Base"
    qlora_params: Dict[str, Any] = Field(default_factory=dict)
    optimizer: str = "paged_adamw_8bit"
    scheduler: str = "cosine"
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = 8
    num_train_epochs: int = 3
    seed: int = 42
    max_length: int = 4096
    quantization_type: str = "nf4"
    use_double_quant: bool = True
    assistant_only_loss: bool = True
    checkpoint_strategy: Dict[str, Any] = Field(default_factory=dict)
    training_engine_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class HardwareProvenance(BaseModel):
    """Execution environment telemetry. Uses NOT_AVAILABLE if evaluated pre-training."""
    gpu_name: str = "NOT_AVAILABLE"
    gpu_count: int = 0
    gpu_vram_gb: float = 0.0
    cuda_version: Optional[str] = None
    pytorch_version: str = "NOT_AVAILABLE"
    transformers_version: str = "NOT_AVAILABLE"
    peft_version: str = "NOT_AVAILABLE"
    bitsandbytes_version: str = "NOT_AVAILABLE"
    status: str = "NOT_AVAILABLE"  # 'RECORDED' or 'NOT_AVAILABLE'

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ProvenanceCollector:
    """Orchestrates dataset, training configuration, and hardware provenance harvesting."""

    @staticmethod
    def collect_dataset_provenance(
        manifest_path: Union[str, Path] = "datasets/production/manifests/production_manifest.json",
        dataset_dir: Union[str, Path] = "datasets/production/processed",
    ) -> DatasetProvenance:
        """Read and verify the production dataset manifest and split files."""
        m_p = Path(manifest_path)
        d_p = Path(dataset_dir)

        if not m_p.exists():
            return DatasetProvenance(
                manifest_path=str(m_p),
                provenance_status="UNVERIFIED",
            )

        try:
            m_sha = compute_file_sha256(m_p)
            with open(m_p, "r", encoding="utf-8") as f:
                data = json.load(f)

            split_hashes = {}
            for split_name in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
                split_f = d_p / split_name
                if split_f.exists():
                    split_hashes[split_name] = compute_file_sha256(split_f)
                else:
                    split_hashes[split_name] = "MISSING"

            is_frozen = data.get("status") == "FROZEN"

            return DatasetProvenance(
                dataset_version=data.get("dataset_version", "dataset-v1.0"),
                lifecycle_status=data.get("status", "UNKNOWN"),
                manifest_path=str(m_p),
                manifest_sha256=m_sha,
                target_count=data.get("target_count", 0),
                actual_final_count=data.get("actual_final_count", 0),
                split_hashes=split_hashes,
                provenance_status="VERIFIED" if is_frozen else "UNVERIFIED",
            )
        except Exception as e:
            logger.error("Failed to collect dataset provenance: %s", e)
            return DatasetProvenance(
                manifest_path=str(m_p),
                provenance_status="UNVERIFIED",
            )

    @staticmethod
    def collect_training_provenance(
        config_path: Union[str, Path] = "configs/training.yaml"
    ) -> TrainingProvenance:
        """Parse training YAML and compute canonical configuration hash."""
        cfg_p = Path(config_path)
        if not cfg_p.exists():
            return TrainingProvenance(config_path=str(cfg_p))

        with open(cfg_p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        raw_str = json.dumps(raw, sort_keys=True)
        c_hash = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

        m_cfg = raw.get("model", {})
        l_cfg = raw.get("lora", {})
        t_cfg = raw.get("training", {})
        q_cfg = raw.get("quantization", {})
        opt_cfg = raw.get("optimizer", {})
        sch_cfg = raw.get("scheduler", {})
        seq_cfg = raw.get("sequence", {})
        ckpt_cfg = raw.get("checkpoint", {})

        per_dev = t_cfg.get("per_device_train_batch_size", 1)
        grad_acc = t_cfg.get("gradient_accumulation_steps", 8)

        return TrainingProvenance(
            config_hash=c_hash,
            config_path=str(cfg_p),
            base_model_name=m_cfg.get("name", "Qwen/Qwen3-4B-Base"),
            qlora_params=l_cfg,
            optimizer=opt_cfg.get("name", "paged_adamw_8bit"),
            scheduler=sch_cfg.get("type", "cosine"),
            learning_rate=t_cfg.get("learning_rate", 2.0e-4),
            warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
            per_device_train_batch_size=per_dev,
            gradient_accumulation_steps=grad_acc,
            effective_batch_size=per_dev * grad_acc,
            num_train_epochs=t_cfg.get("num_train_epochs", 3),
            seed=t_cfg.get("seed", 42),
            max_length=seq_cfg.get("max_length", 4096),
            quantization_type=q_cfg.get("quant_type", "nf4"),
            use_double_quant=q_cfg.get("use_double_quant", True),
            assistant_only_loss=t_cfg.get("assistant_only_loss", True),
            checkpoint_strategy=ckpt_cfg,
        )

    @staticmethod
    def collect_hardware_provenance(
        telemetry: Optional[Dict[str, Any]] = None
    ) -> HardwareProvenance:
        """Capture hardware telemetry. If telemetry is None or pre-training, records NOT_AVAILABLE."""
        if telemetry is not None and telemetry.get("gpu_name") and telemetry.get("gpu_name") != "NOT_AVAILABLE":
            return HardwareProvenance(
                gpu_name=str(telemetry.get("gpu_name")),
                gpu_count=int(telemetry.get("gpu_count", 1)),
                gpu_vram_gb=float(telemetry.get("gpu_vram_gb", 0.0)),
                cuda_version=telemetry.get("cuda_version"),
                pytorch_version=str(telemetry.get("pytorch_version", "")),
                transformers_version=str(telemetry.get("transformers_version", "")),
                peft_version=str(telemetry.get("peft_version", "")),
                bitsandbytes_version=str(telemetry.get("bitsandbytes_version", "")),
                status="RECORDED",
            )

        # Pre-training default: do not invent hardware values
        return HardwareProvenance(
            gpu_name="NOT_AVAILABLE",
            gpu_count=0,
            gpu_vram_gb=0.0,
            status="NOT_AVAILABLE",
        )
