"""
Production Evaluation Execution Engine (Phase 4.7).
Orchestrates hardware-gated benchmark execution for baseline (Qwen3-4B-Base)
and fine-tuned LoRA adapters with atomic persistence and resume capability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import yaml
from pydantic import BaseModel, Field

from src.dataset.schema import Role
from src.evaluation.benchmark_cases import BenchmarkCase
from src.evaluation.benchmark_dataset import BenchmarkDatasetManager
from src.evaluation.config import MetricsConfig
from src.evaluation.inference import EvaluationInferenceResult
from src.evaluation.metrics import MetricCalculator, SampleMetrics
from src.training.utils import compute_file_sha256, detect_hardware_environment

logger = logging.getLogger(__name__)


class GenerationConfig(BaseModel):
    """Locked generation configuration ensuring identical sampling parameters across models."""
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 512
    repetition_penalty: float = 1.05
    do_sample: bool = False
    seed: int = 42

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of generation parameters."""
        data = self.model_dump()
        canonical_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    @classmethod
    def load_from_yaml(cls, path: Union[str, Path]) -> GenerationConfig:
        """Load generation parameters from YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Generation config not found at: {p}")
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        gen_data = raw.get("generation", raw)
        return cls(**gen_data)

    def save_to_yaml(self, path: Union[str, Path]) -> None:
        """Save generation configuration to YAML."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump({"generation": self.model_dump()}, f, default_flow_style=False)


class CaseExecutionResult(BaseModel):
    """Case-level execution result recording telemetry, response, and calculated metrics."""
    experiment_id: str
    benchmark_id: str
    model: str  # 'base' or 'adapter'
    domain: str
    topic: str
    difficulty: str
    task_type: str
    evaluation_type: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    latency_seconds: float = 0.0
    tokens_per_second: float = 0.0
    response: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    status: str = "COMPLETED"  # 'COMPLETED', 'FAILED', 'SKIPPED'
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class GPUReadinessGate:
    """Validates CUDA, GPU detection, and VRAM availability."""

    @staticmethod
    def check() -> Tuple[bool, str]:
        hw = detect_hardware_environment()
        if not hw.cuda_available:
            return False, "MODEL INFERENCE BLOCKED — GPU UNAVAILABLE (CUDA not available)"
        if hw.device_count <= 0:
            return False, "MODEL INFERENCE BLOCKED — GPU UNAVAILABLE (0 GPUs detected)"
        return True, f"GPU Ready: {hw.device_name} ({hw.total_memory_gb:.2f} GB VRAM)"


class EvaluationExecutionEngine:
    """Master engine for executing benchmark evaluations with strict gating and atomic persistence."""

    def __init__(
        self,
        model_type: str,
        benchmark_dir: Union[str, Path],
        generation_config_path: Union[str, Path] = "configs/generation.yaml",
        output_dir: Union[str, Path] = "experiments",
        base_model_path: str = "/content/drive/MyDrive/GoogleColab/AI/Qwen3/models/Qwen3-4B-Base",
        adapter_path: Optional[str] = None,
        dry_run: bool = False,
        resume: bool = False,
    ):
        self.model_type = model_type.lower()
        if self.model_type not in ("base", "adapter"):
            raise ValueError(f"Invalid model_type '{model_type}'. Must be 'base' or 'adapter'.")

        self.benchmark_dir = Path(benchmark_dir)
        self.gen_config = GenerationConfig.load_from_yaml(generation_config_path)
        self.gen_config_hash = self.gen_config.compute_hash()
        self.output_dir = Path(output_dir)
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path or "/content/drive/MyDrive/GoogleColab/AI/Qwen3/models/Qwen3-4B-Base-LoRA"
        self.dry_run = dry_run
        self.resume = resume
        self.metric_calculator = MetricCalculator(MetricsConfig())

    def preflight(self) -> Tuple[bool, List[str]]:
        """Verify benchmark, model weights, adapter configuration, and directory readiness."""
        issues = []

        # 1. Benchmark directory
        if not self.benchmark_dir.exists():
            issues.append(f"Benchmark directory not found: {self.benchmark_dir}")
        else:
            manifest_file = self.benchmark_dir / "manifest.json"
            if not manifest_file.exists():
                issues.append(f"Benchmark manifest missing: {manifest_file}")
            bench_file = self.benchmark_dir / "benchmark.jsonl"
            if not bench_file.exists():
                issues.append(f"Benchmark cases file missing: {bench_file}")

        # 2. Output directory
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Output directory not writable: {e}")

        # 3. Model path & Adapter path
        if not self.dry_run:
            base_p = Path(self.base_model_path)
            if not base_p.exists():
                issues.append(f"Base model path does not exist: {base_p}")

            if self.model_type == "adapter":
                adapt_p = Path(self.adapter_path)
                if not adapt_p.exists():
                    issues.append(f"ADAPTER EVALUATION BLOCKED — ADAPTER NOT AVAILABLE at {adapt_p}")

        return len(issues) == 0, issues

    def format_chat_prompt(self, case: BenchmarkCase) -> str:
        """Format benchmark messages into native Qwen3 chat template string."""
        formatted_turns = []
        for msg in case.get_prompt_messages():
            role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            formatted_turns.append(f"<|im_start|>{role_str}\n{msg.content}<|im_end|>")
        formatted_turns.append("<|im_start|>assistant\n")
        return "\n".join(formatted_turns)

    def execute_dry_run(self, cases: List[BenchmarkCase], manifest_sha: str) -> Dict[str, Any]:
        """Execute non-destructive dry run validating the pipeline without model inference."""
        return {
            "dry_run": True,
            "model_type": self.model_type,
            "benchmark_cases_count": len(cases),
            "benchmark_sha256": manifest_sha,
            "generation_config_hash": self.gen_config_hash,
            "status": "VALIDATED",
            "message": "Dry-run validation successful. No model inference was executed.",
        }
