"""
Evaluation Inference Engine (Phase 4.4).
Supports loading Base model and LoRA Adapter targets, ChatML prompt serialization,
latency and throughput profiling, and hardware-aware inference execution.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pydantic import BaseModel, Field

from src.dataset.schema import Message, Role
from src.evaluation.config import EvaluationConfig, EvaluationModelConfig, GenerationConfig
from src.evaluation.dataset import EvaluationExample
from src.training.tokenizer import MockQwenTokenizer, TrainingTokenizerWrapper
from src.training.utils import detect_hardware_environment


class AdapterCompatibilityError(ValueError):
    """Raised when LoRA adapter metadata or architecture is incompatible with base model."""
    pass


class EvaluationInferenceResult(BaseModel):
    """Telemetry and response text from single-example inference."""
    record_id: str
    domain: str
    topic: str
    task_type: str
    difficulty: str
    prompt: str
    generated_text: str
    reference_text: str
    latency_seconds: float
    tokens_generated: int
    tokens_per_second: float
    vram_allocated_mb: float = 0.0
    is_mock: bool = False
    model_type: str  # 'base' or 'adapter'
    model_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationInferenceEngine:
    """
    Executes inference for Base Qwen3-4B model or Fine-Tuned LoRA Adapter.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model_cfg = config.model
        self.gen_cfg = config.generation
        self.hardware = detect_hardware_environment()

        self.tokenizer = None
        self.model = None
        self.is_mock = False

    def load_tokenizer(self) -> Any:
        """Load tokenizer or mock tokenizer."""
        from src.training.config import TokenizerConfig
        tok_cfg = TokenizerConfig(
            name=self.model_cfg.name,
            path=self.model_cfg.base_model_path,
            fallback_pretrained_id=self.model_cfg.fallback_pretrained_id,
            trust_remote_code=self.model_cfg.trust_remote_code,
        )
        wrapper = TrainingTokenizerWrapper(tok_cfg)
        self.tokenizer = wrapper.load()
        return self.tokenizer

    def format_chatml_prompt(self, messages: List[Message]) -> str:
        """Format message history into ChatML prompt ending with assistant turn header."""
        prompt_parts: List[str] = []
        for msg in messages:
            role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            prompt_parts.append(f"<|im_start|>{role_str}\n{msg.content}<|im_end|>\n")
        # Append assistant header
        prompt_parts.append("<|im_start|>assistant\n")
        return "".join(prompt_parts)

    def validate_adapter_compatibility(self, adapter_path: Path) -> Dict[str, Any]:
        """Verify that adapter exists and is compatible with the base model."""
        if not adapter_path.exists():
            raise AdapterCompatibilityError(f"Adapter checkpoint directory not found: {adapter_path}")

        adapter_config_file = adapter_path / "adapter_config.json"
        if not adapter_config_file.exists():
            # Check parent or metadata
            meta_file = adapter_path / "checkpoint_metadata.json"
            if not meta_file.exists():
                raise AdapterCompatibilityError(
                    f"Invalid adapter checkpoint: missing adapter_config.json or checkpoint_metadata.json in {adapter_path}"
                )

        metadata: Dict[str, Any] = {}
        if (adapter_path / "checkpoint_metadata.json").exists():
            import json
            with open(adapter_path / "checkpoint_metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)

            expected_dataset_ver = self.config.dataset.version
            if metadata.get("dataset_version") and metadata["dataset_version"] != expected_dataset_ver:
                raise AdapterCompatibilityError(
                    f"Adapter dataset version mismatch: adapter was trained on '{metadata.get('dataset_version')}', "
                    f"evaluation expects '{expected_dataset_ver}'"
                )

        return metadata

    def load_model(self) -> Any:
        """
        Load Base model or LoRA Adapter depending on configuration.
        """
        if self.tokenizer is None:
            self.load_tokenizer()

        base_path = Path(self.model_cfg.base_model_path)
        if not base_path.exists():
            base_path = Path(self.model_cfg.local_fallback_base_path)

        adapter_path = Path(self.model_cfg.adapter_path or "")
        if not adapter_path.exists():
            adapter_path = Path(self.model_cfg.local_fallback_adapter_path)

        # 1. Real CUDA Hardware Loading
        if torch.cuda.is_available() and base_path.exists():
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            bnb_config = None
            if self.model_cfg.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

            model = AutoModelForCausalLM.from_pretrained(
                str(base_path),
                quantization_config=bnb_config,
                device_map=self.model_cfg.device_map,
                trust_remote_code=self.model_cfg.trust_remote_code,
                torch_dtype=torch.float16 if bnb_config else torch.float32,
            )

            if self.model_cfg.model_type == "adapter":
                self.validate_adapter_compatibility(adapter_path)
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter_path))

            model.eval()
            self.model = model
            self.is_mock = False
            return self.model

        # 2. Offline / Mock Loading (Hermetic test execution mode)
        self.is_mock = True
        if self.model_cfg.model_type == "adapter":
            # In mock mode, validate adapter path if it exists
            if adapter_path.exists():
                self.validate_adapter_compatibility(adapter_path)

        # Mock Model for test execution
        class MockEvaluationModel:
            def __init__(self, model_type: str):
                self.model_type = model_type

            def generate_text(self, prompt: str, reference: str, gen_cfg: GenerationConfig) -> str:
                # Deterministic simulation of completion
                if self.model_type == "adapter":
                    # Fine-tuned simulation: fluent, properly formatted response
                    return f"```python\n# Solution for {reference[:30]}...\ndef execute():\n    return True\n```"
                else:
                    # Baseline simulation: raw completion
                    return f"Response to the query: {reference[:40]}"

        self.model = MockEvaluationModel(self.model_cfg.model_type)
        return self.model

    def generate(self, example: EvaluationExample) -> EvaluationInferenceResult:
        """
        Generate model response for an evaluation example.
        """
        if self.model is None:
            self.load_model()

        prompt_str = self.format_chatml_prompt(example.prompt_messages)
        start_time = time.time()
        vram_mb = 0.0

        if not self.is_mock and hasattr(self.model, "generate"):
            # Real generation
            inputs = self.tokenizer(prompt_str, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.gen_cfg.max_new_tokens,
                    temperature=self.gen_cfg.temperature if self.gen_cfg.do_sample else None,
                    top_p=self.gen_cfg.top_p if self.gen_cfg.do_sample else None,
                    top_k=self.gen_cfg.top_k if self.gen_cfg.do_sample else None,
                    repetition_penalty=self.gen_cfg.repetition_penalty,
                    do_sample=self.gen_cfg.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            # Strip prompt tokens
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][input_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            tokens_count = len(generated_ids)

        else:
            # Mock / CPU Simulation
            generated_text = self.model.generate_text(prompt_str, example.reference_completion, self.gen_cfg)
            tokens_count = len(generated_text.split())

        elapsed = time.time() - start_time
        tps = round(tokens_count / elapsed, 2) if elapsed > 0 else 0.0

        return EvaluationInferenceResult(
            record_id=example.record_id,
            domain=example.domain,
            topic=example.topic,
            task_type=example.task_type,
            difficulty=example.difficulty,
            prompt=prompt_str,
            generated_text=generated_text,
            reference_text=example.reference_completion,
            latency_seconds=round(elapsed, 4),
            tokens_generated=tokens_count,
            tokens_per_second=tps,
            vram_allocated_mb=round(vram_mb, 2),
            is_mock=self.is_mock,
            model_type=self.model_cfg.model_type,
            model_name=self.model_cfg.name,
            metadata=example.metadata,
        )
