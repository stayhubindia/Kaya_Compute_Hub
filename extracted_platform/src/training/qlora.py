"""
QLoRA Configuration & Parameter Analytics (Phase 4.1).
Provides 4-bit BitsAndBytes quantization setup, PEFT LoRA adapter configuration,
linear module discovery, and trainable parameter accounting for Qwen3-4B-Base.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
from pydantic import BaseModel, Field
from transformers import BitsAndBytesConfig
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.training.config import LoraConfig, QuantizationConfig, TrainingConfig
from src.training.utils import detect_hardware_environment


class ParameterAnalysisReport(BaseModel):
    """Report detailing total, trainable, and frozen model parameter counts."""
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_percentage: float
    target_modules: List[str]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class QLoRAConfigurator:
    """
    Manages 4-bit quantization and LoRA adapter configuration,
    ensuring compatibility with NVIDIA Tesla T4 and accurate parameter accounting.
    """

    def __init__(self, training_config: TrainingConfig):
        self.config = training_config
        self.quant_config = training_config.quantization
        self.lora_config = training_config.lora

    def resolve_compute_dtype(self) -> torch.dtype:
        """Select appropriate compute dtype based on hardware support."""
        hw = detect_hardware_environment()
        if self.quant_config.compute_dtype == "bfloat16":
            if hw.bfloat16_supported:
                return torch.bfloat16
            else:
                return torch.float16
        elif self.quant_config.compute_dtype == "float16":
            return torch.float16
        return torch.float32

    def get_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """Build BitsAndBytes 4-bit quantization configuration."""
        if not self.quant_config.load_in_4bit:
            return None

        compute_dtype = self.resolve_compute_dtype()
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.quant_config.quant_type,
            bnb_4bit_use_double_quant=self.quant_config.use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def get_peft_config(self) -> PeftLoraConfig:
        """Build PEFT LoRA adapter configuration."""
        return PeftLoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type,
            target_modules=self.lora_config.target_modules,
            modules_to_save=self.lora_config.modules_to_save,
        )

    @staticmethod
    def find_all_linear_names(model: torch.nn.Module) -> List[str]:
        """Discover all linear projection layer names in the base model."""
        import bitsandbytes as bnb
        linear_classes = (torch.nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
        lora_module_names: Set[str] = set()

        for name, module in model.named_modules():
            if isinstance(module, linear_classes):
                names = name.split(".")
                # Extract the leaf attribute name (e.g. 'q_proj', 'v_proj')
                lora_module_names.add(names[-1] if len(names) > 1 else names[0])

        # Remove lm_head if present to avoid unquantized output logits overhead
        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")

        return sorted(list(lora_module_names))

    @staticmethod
    def analyze_parameters(model: torch.nn.Module, lora_cfg: Optional[LoraConfig] = None) -> ParameterAnalysisReport:
        """Calculate total, trainable, and frozen parameters for a model."""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = total_params - trainable_params
        trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

        r = lora_cfg.r if lora_cfg else 16
        alpha = lora_cfg.lora_alpha if lora_cfg else 32
        dropout = lora_cfg.lora_dropout if lora_cfg else 0.05
        targets = lora_cfg.target_modules if lora_cfg else []

        return ParameterAnalysisReport(
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            frozen_parameters=frozen_params,
            trainable_percentage=round(trainable_pct, 4),
            target_modules=targets,
            lora_rank=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
        )

    @staticmethod
    def estimate_qwen_qlora_parameters(
        num_layers: int = 36,
        hidden_size: int = 2560,
        intermediate_size: int = 6912,
        r: int = 16,
        target_modules: Optional[List[str]] = None,
    ) -> ParameterAnalysisReport:
        """
        Analytical estimation of Qwen3-4B-Base LoRA parameter metrics
        for preflight checks when model weights are not loaded into VRAM.
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        # Approximate base model params for ~4B architecture
        # Self-attention: Q (h*h), K (h*kv_h), V (h*kv_h), O (h*h)
        # MLP: Gate (h*inter), Up (h*inter), Down (inter*h)
        total_base_params = 4_000_000_000

        # LoRA params per layer:
        # For each target module (in_features -> out_features):
        # LoRA params = r * (in_features + out_features)
        lora_params_per_layer = 0
        for mod in target_modules:
            if mod in ["q_proj", "o_proj"]:
                lora_params_per_layer += r * (hidden_size + hidden_size)
            elif mod in ["k_proj", "v_proj"]:
                # Grouped query attention kv_dim (typically hidden_size / 4 or hidden_size)
                lora_params_per_layer += r * (hidden_size + hidden_size // 4)
            elif mod in ["gate_proj", "up_proj"]:
                lora_params_per_layer += r * (hidden_size + intermediate_size)
            elif mod in ["down_proj"]:
                lora_params_per_layer += r * (intermediate_size + hidden_size)

        total_trainable = num_layers * lora_params_per_layer
        trainable_pct = (total_trainable / (total_base_params + total_trainable)) * 100.0

        return ParameterAnalysisReport(
            total_parameters=total_base_params + total_trainable,
            trainable_parameters=total_trainable,
            frozen_parameters=total_base_params,
            trainable_percentage=round(trainable_pct, 4),
            target_modules=target_modules,
            lora_rank=r,
            lora_alpha=r * 2,
            lora_dropout=0.05,
        )
