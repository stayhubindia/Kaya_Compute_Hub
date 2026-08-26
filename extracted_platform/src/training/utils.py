"""
Training Utilities & Hardware Environment Inspector (Phase 4.1).
Provides runtime hardware detection, deterministic seeding, SHA-256 calculation,
training schedule estimation, and training manifest generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

from pydantic import BaseModel, Field


class HardwareEnvironmentInfo(BaseModel):
    """Runtime hardware and environment diagnostics."""
    cuda_available: bool = Field(description="Whether CUDA acceleration is available")
    device_count: int = Field(default=0, description="Number of detected CUDA devices")
    device_name: Optional[str] = Field(default=None, description="Primary CUDA device name")
    total_memory_gb: float = Field(default=0.0, description="Total GPU VRAM in gigabytes")
    compute_capability: Optional[str] = Field(default=None, description="CUDA compute capability (e.g. '7.5' for T4)")
    bfloat16_supported: bool = Field(default=False, description="Whether hardware supports native bfloat16")
    cpu_count: int = Field(default_factory=os.cpu_count, description="System CPU core count")
    system_memory_gb: float = Field(default=0.0, description="Host RAM in gigabytes")
    torch_version: str = Field(default="", description="Installed PyTorch version")
    cuda_version: Optional[str] = Field(default=None, description="PyTorch compiled CUDA version")
    transformers_version: str = Field(default="", description="Installed Transformers version")
    peft_version: str = Field(default="", description="Installed PEFT version")
    bitsandbytes_version: str = Field(default="", description="Installed BitsAndBytes version")
    trl_version: str = Field(default="", description="Installed TRL version")
    accelerate_version: str = Field(default="", description="Installed Accelerate version")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def detect_hardware_environment() -> HardwareEnvironmentInfo:
    """Inspect and report the runtime execution hardware and library environment."""
    cuda_available = torch.cuda.is_available() if torch is not None else False
    device_count = torch.cuda.device_count() if (torch is not None and cuda_available) else 0
    device_name = torch.cuda.get_device_name(0) if (torch is not None and cuda_available and device_count > 0) else None
    total_memory_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        if (torch is not None and cuda_available and device_count > 0)
        else 0.0
    )
    compute_capability = None
    bfloat16_supported = False
    if torch is not None and cuda_available and device_count > 0:
        cap = torch.cuda.get_device_capability(0)
        compute_capability = f"{cap[0]}.{cap[1]}"
        try:
            bfloat16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            bfloat16_supported = False

    # Host memory
    try:
        sys_mem = round(psutil.virtual_memory().total / (1024 ** 3), 2) if psutil is not None else 0.0
    except Exception:
        sys_mem = 0.0

    # Package versions
    def _get_ver(mod_name: str) -> str:
        try:
            mod = __import__(mod_name)
            return getattr(mod, "__version__", "installed")
        except ImportError:
            return "NOT_INSTALLED"

    return HardwareEnvironmentInfo(
        cuda_available=cuda_available,
        device_count=device_count,
        device_name=device_name,
        total_memory_gb=total_memory_gb,
        compute_capability=compute_capability,
        bfloat16_supported=bfloat16_supported,
        system_memory_gb=sys_mem,
        torch_version=torch.__version__ if torch is not None else "NOT_INSTALLED",
        cuda_version=torch.version.cuda if torch is not None else None,
        transformers_version=_get_ver("transformers"),
        peft_version=_get_ver("peft"),
        bitsandbytes_version=_get_ver("bitsandbytes"),
        trl_version=_get_ver("trl"),
        accelerate_version=_get_ver("accelerate"),
    )


def set_seed(seed: int = 42) -> None:
    """Establish deterministic seed across Python random, NumPy, and PyTorch."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


set_deterministic_seed = set_seed


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Calculate the cryptographic SHA-256 hash of a file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot compute hash: file does not exist at '{p}'")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Calculate deterministic SHA-256 hash of a training configuration."""
    canonical_json = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def estimate_training_schedule(
    record_count: int,
    total_tokens: int,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    epochs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute training schedule estimations: effective batch size, steps per epoch,
    and cumulative token exposure across specified epochs.
    """
    if epochs is None:
        epochs = [1, 2, 3]

    effective_batch_size = micro_batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(record_count / effective_batch_size) if effective_batch_size > 0 else 0
    avg_tokens_per_record = (total_tokens / record_count) if record_count > 0 else 0.0

    epoch_estimates = {}
    for ep in epochs:
        total_steps = steps_per_epoch * ep
        total_token_exposure = total_tokens * ep
        epoch_estimates[f"{ep}_epoch{'s' if ep > 1 else ''}"] = {
            "epochs": ep,
            "total_steps": total_steps,
            "total_tokens": total_token_exposure,
            "avg_tokens_per_step": round(total_token_exposure / total_steps, 2) if total_steps > 0 else 0.0,
        }

    return {
        "dataset_records": record_count,
        "dataset_tokens": total_tokens,
        "avg_tokens_per_record": round(avg_tokens_per_record, 2),
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "schedules": epoch_estimates,
    }


class TrainingManifest(BaseModel):
    """Immutable audit manifest for a fine-tuning run."""
    manifest_version: str = Field(default="1.0.0")
    dataset_version: str = Field(default="dataset-v1.0")
    dataset_sha256: str = Field(default="")
    train_sha256: str = Field(default="")
    validation_sha256: str = Field(default="")
    test_sha256: str = Field(default="")
    model_name: str = Field(default="Qwen/Qwen3-4B-Base")
    model_path: str = Field(default="")
    tokenizer_path: str = Field(default="")
    quantization_config: Dict[str, Any] = Field(default_factory=dict)
    lora_config: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    hardware_environment: Dict[str, Any] = Field(default_factory=dict)
    training_schedule: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def save_json(self, path: Union[str, Path]) -> None:
        """Persist manifest securely as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
