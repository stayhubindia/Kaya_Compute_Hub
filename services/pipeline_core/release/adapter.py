"""
Adapter Artifact Validation Engine (Phase 5.1).
Validates PEFT/LoRA adapter directories, config schema parameters,
weight file existence, and tokenizer assets against training specifications.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AdapterConfigSchema(BaseModel):
    """Schema representing HuggingFace PEFT / LoRA adapter_config.json."""
    peft_type: str = "LORA"
    task_type: str = "CAUSAL_LM"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    bias: str = "none"
    base_model_name_or_path: Optional[str] = None
    modules_to_save: Optional[List[str]] = None


class AdapterValidationResult(BaseModel):
    """Detailed outcome of adapter directory audit."""
    is_valid: bool = False
    status: str = "ARTIFACT_NOT_AVAILABLE"  # 'VALID_ARTIFACT', 'ARTIFACT_NOT_AVAILABLE', 'INVALID_FORMAT'
    adapter_config_path: Optional[str] = None
    weights_path: Optional[str] = None
    weights_format: Optional[str] = None  # 'safetensors' or 'bin'
    tokenizer_present: bool = False
    config_data: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class AdapterValidator:
    """Validates real PEFT/LoRA adapter artifacts against expected training configurations."""

    def __init__(
        self,
        expected_r: int = 16,
        expected_alpha: int = 32,
        expected_dropout: float = 0.05,
        expected_target_modules: Optional[List[str]] = None,
        expected_base_model: Optional[str] = "Qwen/Qwen3-4B-Base",
    ):
        self.expected_r = expected_r
        self.expected_alpha = expected_alpha
        self.expected_dropout = expected_dropout
        self.expected_target_modules = set(
            expected_target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        self.expected_base_model = expected_base_model

    def validate_directory(self, adapter_dir: Union[str, Path]) -> AdapterValidationResult:
        """Audit an adapter folder and verify config, weights, and tokenizer files."""
        p = Path(adapter_dir)
        result = AdapterValidationResult()

        if not p.exists() or not p.is_dir():
            result.status = "ARTIFACT_NOT_AVAILABLE"
            result.errors.append(f"Adapter directory does not exist: {p}")
            return result

        # 1. Check adapter_config.json
        cfg_file = p / "adapter_config.json"
        if not cfg_file.exists():
            result.status = "ARTIFACT_NOT_AVAILABLE"
            result.errors.append(f"adapter_config.json not found in {p}")
            return result

        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
            result.config_data = raw_cfg
            result.adapter_config_path = str(cfg_file)
        except Exception as e:
            result.status = "INVALID_FORMAT"
            result.errors.append(f"Failed to parse adapter_config.json: {e}")
            return result

        # 2. Validate config schema & parameters
        try:
            parsed_cfg = AdapterConfigSchema(**raw_cfg)
        except Exception as e:
            result.status = "INVALID_FORMAT"
            result.errors.append(f"adapter_config.json schema validation failed: {e}")
            return result

        if parsed_cfg.peft_type.upper() != "LORA":
            result.errors.append(f"Expected peft_type 'LORA', got '{parsed_cfg.peft_type}'")

        if parsed_cfg.task_type.upper() != "CAUSAL_LM":
            result.errors.append(f"Expected task_type 'CAUSAL_LM', got '{parsed_cfg.task_type}'")

        if parsed_cfg.r != self.expected_r:
            result.errors.append(f"Expected LoRA rank {self.expected_r}, got {parsed_cfg.r}")

        if parsed_cfg.lora_alpha != self.expected_alpha:
            result.errors.append(f"Expected LoRA alpha {self.expected_alpha}, got {parsed_cfg.lora_alpha}")

        if abs(parsed_cfg.lora_dropout - self.expected_dropout) > 1e-4:
            result.errors.append(
                f"Expected LoRA dropout {self.expected_dropout}, got {parsed_cfg.lora_dropout}"
            )

        found_targets = set(parsed_cfg.target_modules)
        missing_targets = self.expected_target_modules - found_targets
        if missing_targets:
            result.errors.append(f"Missing expected target modules: {sorted(missing_targets)}")

        if self.expected_base_model and parsed_cfg.base_model_name_or_path:
            base_ref = parsed_cfg.base_model_name_or_path
            if self.expected_base_model not in base_ref and "Qwen3-4B-Base" not in base_ref:
                result.warnings.append(
                    f"base_model_name_or_path '{base_ref}' differs from expected '{self.expected_base_model}'"
                )

        # 3. Check Weights file
        weights_safetensors = p / "adapter_model.safetensors"
        weights_bin = p / "adapter_model.bin"

        if weights_safetensors.exists() and weights_safetensors.stat().st_size > 0:
            result.weights_path = str(weights_safetensors)
            result.weights_format = "safetensors"
        elif weights_bin.exists() and weights_bin.stat().st_size > 0:
            result.weights_path = str(weights_bin)
            result.weights_format = "bin"
        else:
            result.status = "ARTIFACT_NOT_AVAILABLE"
            result.errors.append("No non-empty adapter weight file found (adapter_model.safetensors or adapter_model.bin)")
            return result

        # 4. Check Tokenizer
        tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "vocab.json"]
        tok_dir = p / "tokenizer" if (p / "tokenizer").is_dir() else p
        has_tok = any((tok_dir / tf).exists() for tf in tokenizer_files)
        result.tokenizer_present = has_tok
        if not has_tok:
            result.warnings.append("No tokenizer configuration detected inside adapter directory")

        # 5. Determine validity
        if len(result.errors) == 0:
            result.is_valid = True
            result.status = "VALID_ARTIFACT"
        else:
            result.is_valid = False
            result.status = "INVALID_FORMAT"

        return result
