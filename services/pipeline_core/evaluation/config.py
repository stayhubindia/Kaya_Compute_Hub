"""
Evaluation Configuration Specification (Phase 4.4).
Provides strongly typed Pydantic models, validation rules, YAML loader,
and hash tracking for the evaluation and benchmarking subsystem.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class EvaluationModelConfig(BaseModel):
    """Configuration for evaluation model targets (Base vs LoRA Adapter)."""
    name: str = Field(default="Qwen/Qwen3-4B-Base", description="Model architecture identifier")
    model_type: Literal["base", "adapter"] = Field(
        default="base", description="Evaluation target: 'base' model or 'adapter' (Base + LoRA)"
    )
    base_model_path: str = Field(
        default="/content/drive/MyDrive/GoogleColab/AI/Qwen3/models/Qwen3-4B-Base",
        description="Path to base model directory",
    )
    adapter_path: Optional[str] = Field(
        default="/content/drive/MyDrive/GoogleColab/AI/Qwen3/training/dataset-v1.0/qlora-v1/checkpoints/best",
        description="Path to fine-tuned LoRA adapter checkpoint",
    )
    local_fallback_base_path: str = Field(
        default="models/Qwen3-4B-Base",
        description="Local fallback directory for base model",
    )
    local_fallback_adapter_path: str = Field(
        default="outputs/training/dataset-v1.0/qlora-v1/checkpoints/best",
        description="Local fallback directory for LoRA adapter",
    )
    fallback_pretrained_id: Optional[str] = Field(
        default=None,
        description="HuggingFace fallback model ID if local path is unavailable",
    )
    trust_remote_code: bool = Field(default=True)
    torch_dtype: str = Field(default="auto")
    device_map: str = Field(default="auto")
    load_in_4bit: bool = Field(default=True, description="Load in 4-bit NF4 if GPU available")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        if v not in ("base", "adapter"):
            raise ValueError(f"Invalid model_type '{v}'. Must be 'base' or 'adapter'.")
        return v


class EvaluationDatasetConfig(BaseModel):
    """Configuration for evaluation dataset loading and isolation."""
    version: str = Field(default="dataset-v1.0")
    lifecycle: str = Field(default="FROZEN")
    manifest_path: str = Field(default="datasets/production/manifests/production_manifest.json")
    train_file: str = Field(default="datasets/production/processed/train.jsonl")
    validation_file: str = Field(default="datasets/production/processed/validation.jsonl")
    test_file: str = Field(default="datasets/production/processed/test.jsonl")
    split: str = Field(default="test", description="Target split for evaluation (default: test)")
    require_frozen: bool = Field(default=True)
    validate_sha256: bool = Field(default=True)
    max_examples: Optional[int] = Field(default=None, ge=1)

    @field_validator("lifecycle")
    @classmethod
    def validate_lifecycle(cls, v: str) -> str:
        if v != "FROZEN":
            raise ValueError(f"Evaluation requires FROZEN lifecycle. Found '{v}'.")
        return v

    @field_validator("split")
    @classmethod
    def validate_split(cls, v: str) -> str:
        if v not in ("test", "validation", "train"):
            raise ValueError(f"Invalid evaluation split '{v}'. Allowed: ['test', 'validation', 'train']")
        return v


class GenerationConfig(BaseModel):
    """Configuration for model response generation during evaluation."""
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    do_sample: bool = Field(default=False, description="Deterministic greedy decoding for benchmark consistency")
    stop_strings: List[str] = Field(
        default_factory=lambda: ["<|im_end|>", "<|endoftext|>"]
    )


class MetricsConfig(BaseModel):
    """Configuration for metric calculation and analysis."""
    compute_deterministic: bool = Field(default=True)
    compute_repetition: bool = Field(default=True)
    compute_formatting: bool = Field(default=True)
    compute_task_matching: bool = Field(default=True)
    n_gram_sizes: List[int] = Field(default_factory=lambda: [2, 3, 4])


class RegressionConfig(BaseModel):
    """Configuration for baseline vs fine-tuned regression comparison."""
    tolerance_pct: float = Field(default=5.0, ge=0.0, le=100.0, description="Tolerance % for unchanged metric status")
    critical_metrics: List[str] = Field(
        default_factory=lambda: [
            "validity_rate",
            "empty_rate",
            "repetition_ratio",
            "formatting_score",
        ]
    )


class EvaluationConfig(BaseModel):
    """Master configuration for Phase 4.4 Evaluation and Benchmarking Engine."""
    model: EvaluationModelConfig = Field(default_factory=EvaluationModelConfig)
    dataset: EvaluationDatasetConfig = Field(default_factory=EvaluationDatasetConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    regression: RegressionConfig = Field(default_factory=RegressionConfig)
    output_dir: str = Field(default="outputs/evaluation")
    reports_dir: str = Field(default="reports")
    seed: int = Field(default=42)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> EvaluationConfig:
        """Load evaluation config from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Evaluation config file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save evaluation config to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of configuration payload."""
        data_str = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()
