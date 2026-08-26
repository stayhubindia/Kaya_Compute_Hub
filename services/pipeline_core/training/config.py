"""
Training Configuration Schema & Validation (Phase 4.1 & 4.2).
Provides strongly typed Pydantic models for QLoRA fine-tuning configuration,
hardware hyperparameters, dataset locking, quantization, and evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Configuration for base model loading."""
    name: str = Field(default="Qwen/Qwen3-4B-Base", description="Model name or identifier")
    path: str = Field(default="/content/drive/MyDrive/GoogleColab/AI/Qwen3/models/Qwen3-4B-Base", description="Local/Drive path to weights")
    fallback_pretrained_id: Optional[str] = Field(default=None, description="Hugging Face hub ID if local path not found")
    trust_remote_code: bool = Field(default=True, description="Whether to trust remote code")
    torch_dtype: str = Field(default="auto", description="Model torch dtype")


class DatasetConfig(BaseModel):
    """Configuration for training dataset source and validation."""
    version: str = Field(default="dataset-v1.0", description="Target dataset version")
    lifecycle: str = Field(default="FROZEN", description="Lifecycle state requirement")
    manifest_path: str = Field(default="datasets/production/manifests/production_manifest.json", description="Path to production manifest")
    candidate_file: Optional[str] = Field(default="datasets/production/processed/candidate_dataset.jsonl", description="Path to candidate JSONL")
    train_file: str = Field(default="datasets/production/processed/train.jsonl", description="Path to train JSONL")
    validation_file: str = Field(default="datasets/production/processed/validation.jsonl", description="Path to validation JSONL")
    test_file: str = Field(default="datasets/production/processed/test.jsonl", description="Path to test JSONL")
    require_frozen: bool = Field(default=True, description="Enforce dataset state is FROZEN")
    validate_sha256: bool = Field(default=True, description="Verify SHA-256 checksums against manifest")
    split_ratios: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.90, "validation": 0.05, "test": 0.05},
        description="Expected split proportions"
    )

    @field_validator("split_ratios")
    @classmethod
    def validate_split_sum(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
        return v


class TokenizerConfig(BaseModel):
    """Configuration for tokenizer loading and sequence length bounds."""
    model_path: str = Field(default="/content/drive/MyDrive/GoogleColab/AI/Qwen3/models/Qwen3-4B-Base", description="Path or HF ID")
    fallback_pretrained_id: Optional[str] = Field(default=None, description="Fallback Hugging Face tokenizer ID")
    max_seq_length: int = Field(default=4096, ge=64, le=32768, description="Maximum sequence length")
    truncation: bool = Field(default=True, description="Whether to truncate sequences exceeding max_seq_length")
    padding: bool = Field(default=False, description="Whether to pad during dataset mapping")
    padding_side: str = Field(default="right", description="Tokenizer padding side ('right' or 'left')")
    add_eos_token: bool = Field(default=False, description="Whether to append EOS token automatically")
    trust_remote_code: bool = Field(default=True, description="Trust remote code on tokenizer load")


class QuantizationConfig(BaseModel):
    """Configuration for 4-bit BitsAndBytes quantization."""
    enabled: bool = Field(default=True, description="Enable quantization")
    bits: int = Field(default=4, description="Quantization bitwidth")
    load_in_4bit: bool = Field(default=True, description="Enable 4-bit base model loading")
    quant_type: str = Field(default="nf4", description="Quantization type ('nf4' or 'fp4')")
    use_double_quant: bool = Field(default=True, description="Use nested quantization for memory savings")
    compute_dtype: str = Field(default="bfloat16", description="Compute dtype ('bfloat16' or 'float16')")
    fallback_compute_dtype: str = Field(default="float16", description="Fallback dtype if bfloat16 unsupported")

    @field_validator("quant_type")
    @classmethod
    def validate_quant_type(cls, v: str) -> str:
        if v not in ["nf4", "fp4"]:
            raise ValueError(f"quant_type must be 'nf4' or 'fp4', got '{v}'")
        return v


class LoraConfig(BaseModel):
    """Configuration for PEFT LoRA adapter injection."""
    enabled: bool = Field(default=True, description="Enable LoRA fine-tuning")
    r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, le=512, description="LoRA scaling alpha")
    alpha: Optional[int] = Field(default=None, description="Alias for lora_alpha")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout rate")
    dropout: Optional[float] = Field(default=None, description="Alias for lora_dropout")
    bias: str = Field(default="none", description="LoRA bias setting ('none', 'all', 'lora_only')")
    task_type: str = Field(default="CAUSAL_LM", description="PEFT task type")
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        description="Linear projection module names for LoRA adapters"
    )
    modules_to_save: Optional[List[str]] = Field(default=None, description="Modules to save in full precision")

    @model_validator(mode="after")
    def sync_aliases(self) -> "LoraConfig":
        if self.alpha is not None:
            self.lora_alpha = self.alpha
        if self.dropout is not None:
            self.lora_dropout = self.dropout
        return self


class TrainingHyperparameters(BaseModel):
    """Training schedule and optimizer hyperparameters."""
    output_dir: str = Field(
        default="/content/drive/MyDrive/GoogleColab/AI/Qwen3/training/dataset-v1.0/qlora-v1",
        description="Target output directory for checkpoints and adapters"
    )
    local_fallback_output_dir: str = Field(
        default="outputs/training/dataset-v1.0/qlora-v1",
        description="Fallback directory if target drive path is unwritable"
    )
    num_train_epochs: int = Field(default=3, ge=1, le=100, description="Total training epochs")
    per_device_train_batch_size: int = Field(default=1, ge=1, le=64, description="Micro-batch size per device")
    per_device_eval_batch_size: int = Field(default=1, ge=1, le=64, description="Evaluation micro-batch size")
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=256, description="Gradient accumulation steps")
    learning_rate: float = Field(default=2.0e-4, gt=0.0, le=1.0, description="Peak learning rate")
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0, description="Weight decay")
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0, description="Warmup ratio of total steps")
    max_grad_norm: float = Field(default=1.0, ge=0.0, description="Gradient clipping norm")
    lr_scheduler_type: str = Field(default="cosine", description="Learning rate scheduler")
    optimizer_name: str = Field(default="paged_adamw_8bit", description="Optimizer type ('paged_adamw_8bit', 'adamw_torch')")
    logging_steps: int = Field(default=5, ge=1, description="Telemetry logging interval in steps")
    save_steps: int = Field(default=25, ge=1, description="Checkpoint save interval in steps")
    eval_steps: int = Field(default=25, ge=1, description="Evaluation interval in steps")
    save_total_limit: int = Field(default=3, ge=1, description="Maximum stored checkpoints")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing for VRAM reduction")
    mixed_precision: str = Field(default="auto", description="Mixed precision mode ('auto', 'bf16', 'fp16', 'no')")
    seed: int = Field(default=42, description="Global random seed")
    dataloader_num_workers: int = Field(default=0, ge=0, description="Dataloader worker processes")
    assistant_only_loss: bool = Field(default=True, description="Mask non-assistant tokens with -100 for SFT loss")

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


class EvaluationConfig(BaseModel):
    """Configuration for validation and evaluation loops."""
    enabled: bool = Field(default=True, description="Enable evaluation during training")
    eval_strategy: str = Field(default="epoch", description="Evaluation strategy ('steps' or 'epoch')")
    strategy: Optional[str] = Field(default=None, description="Alias for eval_strategy")
    eval_steps: int = Field(default=25, ge=1, description="Evaluation step interval")
    per_device_eval_batch_size: int = Field(default=1, ge=1, description="Batch size for evaluation")
    metrics: List[str] = Field(default_factory=lambda: ["loss", "perplexity"], description="Evaluation metrics")
    domain_stratified: bool = Field(default=True, description="Compute evaluation loss per domain")
    difficulty_stratified: bool = Field(default=True, description="Compute evaluation loss per difficulty")

    @model_validator(mode="after")
    def sync_strategy(self) -> "EvaluationConfig":
        if self.strategy is not None:
            self.eval_strategy = self.strategy
        return self


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint management and recovery."""
    enabled: bool = Field(default=True, description="Enable checkpoint saving")
    save_strategy: str = Field(default="steps", description="Save strategy ('steps' or 'epoch')")
    save_steps: int = Field(default=25, ge=1, description="Save step interval")
    save_total_limit: int = Field(default=3, ge=1, description="Maximum retained checkpoints")
    resume_enabled: bool = Field(default=True, description="Whether checkpoint resuming is enabled")
    resume_from_checkpoint: Optional[str] = Field(default=None, description="Path to checkpoint directory to resume from")
    enforce_dataset_version_lock: bool = Field(default=True, description="Reject checkpoints from different dataset versions")


class TrainingConfig(BaseModel):
    """Root configuration model for QLoRA fine-tuning."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    checkpoint: Optional[CheckpointConfig] = Field(default=None, description="Alias for checkpointing")
    optimizer: Optional[Dict[str, Any]] = Field(default=None, description="Optimizer config dictionary")
    scheduler: Optional[Dict[str, Any]] = Field(default=None, description="Scheduler config dictionary")
    sequence: Optional[Dict[str, Any]] = Field(default=None, description="Sequence bounds config dictionary")

    @model_validator(mode="after")
    def sync_nested_configs(self) -> "TrainingConfig":
        if self.checkpoint is not None:
            self.checkpointing = self.checkpoint
        if self.optimizer and "name" in self.optimizer:
            self.training.optimizer_name = str(self.optimizer["name"])
        if self.scheduler and "type" in self.scheduler:
            self.training.lr_scheduler_type = str(self.scheduler["type"])
        if self.sequence and "max_length" in self.sequence:
            self.tokenizer.max_seq_length = int(self.sequence["max_length"])
        return self

    @classmethod
    def load_from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load and validate training configuration from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return cls(**raw_data)

    def save_to_yaml(self, path: Union[str, Path]) -> None:
        """Persist configuration to a YAML file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def validate_rules(self) -> List[str]:
        """Perform comprehensive sanity and bounds validation across all parameters."""
        errors: List[str] = []
        if self.training.learning_rate <= 0:
            errors.append(f"Learning rate must be > 0, got {self.training.learning_rate}")
        if self.training.num_train_epochs <= 0:
            errors.append(f"num_train_epochs must be > 0, got {self.training.num_train_epochs}")
        if self.training.per_device_train_batch_size <= 0:
            errors.append(f"per_device_train_batch_size must be > 0, got {self.training.per_device_train_batch_size}")
        if self.training.gradient_accumulation_steps <= 0:
            errors.append(f"gradient_accumulation_steps must be > 0, got {self.training.gradient_accumulation_steps}")
        if self.tokenizer.max_seq_length <= 0:
            errors.append(f"max_seq_length must be > 0, got {self.tokenizer.max_seq_length}")
        if self.lora.r <= 0:
            errors.append(f"LoRA rank r must be > 0, got {self.lora.r}")
        if self.lora.lora_alpha <= 0:
            errors.append(f"LoRA alpha must be > 0, got {self.lora.lora_alpha}")
        if not (0.0 <= self.lora.lora_dropout <= 1.0):
            errors.append(f"LoRA dropout must be between 0.0 and 1.0, got {self.lora.lora_dropout}")
        return errors
