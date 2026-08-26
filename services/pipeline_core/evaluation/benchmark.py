"""
Domain, Difficulty, & Task Benchmark Engine (Phase 4.4).
Implements multi-dimensional metric stratification across all 13 domains,
4 difficulty levels, and task types without artificial score equalizations.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.evaluation.config import EvaluationConfig
from src.evaluation.inference import EvaluationInferenceResult
from src.evaluation.metrics import AggregatedMetrics, MetricCalculator, SampleMetrics


DOMAINS_13 = [
    "programming",
    "software_engineering",
    "cybersecurity",
    "linux_systems",
    "networking",
    "ai_ml",
    "mathematics",
    "science",
    "psychology",
    "human_behavior",
    "reasoning",
    "technology",
    "general_knowledge",
]

DIFFICULTIES_4 = ["beginner", "intermediate", "advanced", "expert"]


class OverallBenchmarkReport(BaseModel):
    """Master benchmark report capturing multi-dimensional evaluation results."""
    model_name: str
    model_type: str  # 'base' or 'adapter'
    dataset_version: str
    dataset_sha256: str = ""
    config_hash: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hardware_device: str = "CPU"
    is_mock: bool = False
    sample_count: int = 0
    overall_metrics: AggregatedMetrics = Field(default_factory=AggregatedMetrics)
    domain_metrics: Dict[str, AggregatedMetrics] = Field(default_factory=dict)
    difficulty_metrics: Dict[str, AggregatedMetrics] = Field(default_factory=dict)
    task_metrics: Dict[str, AggregatedMetrics] = Field(default_factory=dict)
    sample_results: List[EvaluationInferenceResult] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class BenchmarkEngine:
    """Executes stratified multi-dimensional benchmark analysis."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.metric_calculator = MetricCalculator(self.config.metrics)

    def compute_benchmark(
        self,
        inference_results: List[EvaluationInferenceResult],
        dataset_sha256: str = "",
        hardware_device: str = "CPU",
    ) -> OverallBenchmarkReport:
        """
        Compute full benchmark report from raw inference results.
        """
        if not inference_results:
            return OverallBenchmarkReport(
                model_name=self.config.model.name,
                model_type=self.config.model.model_type,
                dataset_version=self.config.dataset.version,
                dataset_sha256=dataset_sha256,
                config_hash=self.config.compute_hash(),
                hardware_device=hardware_device,
            )

        # 1. Compute per-sample metrics
        sample_metrics: List[SampleMetrics] = []
        for r in inference_results:
            sm = self.metric_calculator.calculate_sample_metrics(r)
            sample_metrics.append(sm)

        # 2. Overall Aggregation
        overall = self.metric_calculator.aggregate_metrics(sample_metrics, inference_results)

        # 3. Domain Grouping
        domain_samples: Dict[str, List[SampleMetrics]] = defaultdict(list)
        domain_inferences: Dict[str, List[EvaluationInferenceResult]] = defaultdict(list)
        for sm, inf in zip(sample_metrics, inference_results):
            domain_samples[inf.domain].append(sm)
            domain_inferences[inf.domain].append(inf)

        domain_metrics: Dict[str, AggregatedMetrics] = {}
        for dom, s_list in domain_samples.items():
            domain_metrics[dom] = self.metric_calculator.aggregate_metrics(s_list, domain_inferences[dom])

        # 4. Difficulty Grouping
        diff_samples: Dict[str, List[SampleMetrics]] = defaultdict(list)
        diff_inferences: Dict[str, List[EvaluationInferenceResult]] = defaultdict(list)
        for sm, inf in zip(sample_metrics, inference_results):
            diff_samples[inf.difficulty].append(sm)
            diff_inferences[inf.difficulty].append(inf)

        difficulty_metrics: Dict[str, AggregatedMetrics] = {}
        for diff, s_list in diff_samples.items():
            difficulty_metrics[diff] = self.metric_calculator.aggregate_metrics(s_list, diff_inferences[diff])

        # 5. Task-Type Grouping
        task_samples: Dict[str, List[SampleMetrics]] = defaultdict(list)
        task_inferences: Dict[str, List[EvaluationInferenceResult]] = defaultdict(list)
        for sm, inf in zip(sample_metrics, inference_results):
            task_samples[inf.task_type].append(sm)
            task_inferences[inf.task_type].append(inf)

        task_metrics: Dict[str, AggregatedMetrics] = {}
        for task, s_list in task_samples.items():
            task_metrics[task] = self.metric_calculator.aggregate_metrics(s_list, task_inferences[task])

        first_inf = inference_results[0]
        return OverallBenchmarkReport(
            model_name=first_inf.model_name,
            model_type=first_inf.model_type,
            dataset_version=self.config.dataset.version,
            dataset_sha256=dataset_sha256,
            config_hash=self.config.compute_hash(),
            hardware_device=hardware_device,
            is_mock=first_inf.is_mock,
            sample_count=len(inference_results),
            overall_metrics=overall,
            domain_metrics=domain_metrics,
            difficulty_metrics=difficulty_metrics,
            task_metrics=task_metrics,
            sample_results=inference_results,
        )
