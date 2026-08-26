"""
Regression Analysis Framework (Phase 4.4).
Implements multi-metric comparative regression testing between Baseline model
and Fine-Tuned LoRA Adapter with delta tracking and classification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from src.evaluation.benchmark import OverallBenchmarkReport
from src.evaluation.config import RegressionConfig
from src.evaluation.metrics import AggregatedMetrics


ChangeStatus = Literal["IMPROVED", "REGRESSED", "UNCHANGED"]

# Definition of metric optimization directions
HIGHER_IS_BETTER = {
    "validity_rate",
    "avg_formatting_score",
    "exact_match_rate",
    "avg_keyword_overlap",
    "avg_unique_word_ratio",
    "avg_tokens_per_second",
}

LOWER_IS_BETTER = {
    "empty_rate",
    "avg_repetition_ratio",
    "repeated_lines_rate",
    "truncation_rate",
    "avg_latency_seconds",
}


class MetricDelta(BaseModel):
    """Comparative measurement between baseline and adapter for a single metric."""
    metric_name: str
    baseline_value: float
    adapter_value: float
    absolute_delta: float  # adapter - baseline
    percent_change: float
    status: ChangeStatus
    direction: str  # 'higher_is_better', 'lower_is_better', 'neutral'


class GroupRegressionDelta(BaseModel):
    """Regression comparison for a domain, difficulty, or task group."""
    group_name: str
    group_type: str  # 'domain', 'difficulty', 'task'
    metrics: Dict[str, MetricDelta] = Field(default_factory=dict)
    summary_status: str = "UNCHANGED"


class RegressionReport(BaseModel):
    """Comprehensive regression scorecard comparing Baseline vs Adapter."""
    baseline_model: str
    adapter_model: str
    dataset_version: str
    tolerance_pct: float
    overall_deltas: Dict[str, MetricDelta] = Field(default_factory=dict)
    domain_deltas: Dict[str, GroupRegressionDelta] = Field(default_factory=dict)
    difficulty_deltas: Dict[str, GroupRegressionDelta] = Field(default_factory=dict)
    task_deltas: Dict[str, GroupRegressionDelta] = Field(default_factory=dict)
    total_improvements: int = 0
    total_regressions: int = 0
    total_unchanged: int = 0
    verdict: str = "INCONCLUSIVE"  # 'IMPROVED', 'REGRESSED', 'NEUTRAL', 'INCONCLUSIVE'
    executive_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class RegressionAnalyzer:
    """Computes regression analysis comparing Baseline vs Adapter benchmark reports."""

    def __init__(self, config: Optional[RegressionConfig] = None):
        self.config = config or RegressionConfig()

    def _compare_single_metric(self, name: str, base_val: float, adapt_val: float) -> MetricDelta:
        """Compute delta, percent change, and classification for a metric."""
        abs_delta = round(adapt_val - base_val, 4)
        pct_change = round(((adapt_val - base_val) / base_val * 100), 2) if base_val != 0 else 0.0

        tol = self.config.tolerance_pct
        status: ChangeStatus = "UNCHANGED"

        if name in HIGHER_IS_BETTER:
            direction = "higher_is_better"
            if pct_change > tol or (base_val == 0 and adapt_val > 0):
                status = "IMPROVED"
            elif pct_change < -tol or (base_val > 0 and adapt_val == 0):
                status = "REGRESSED"
        elif name in LOWER_IS_BETTER:
            direction = "lower_is_better"
            if pct_change < -tol or (base_val > 0 and adapt_val == 0):
                status = "IMPROVED"
            elif pct_change > tol or (base_val == 0 and adapt_val > 0):
                status = "REGRESSED"
        else:
            direction = "neutral"
            if abs(pct_change) > tol:
                status = "IMPROVED" if abs_delta > 0 else "REGRESSED"

        return MetricDelta(
            metric_name=name,
            baseline_value=round(base_val, 4),
            adapter_value=round(adapt_val, 4),
            absolute_delta=abs_delta,
            percent_change=pct_change,
            status=status,
            direction=direction,
        )

    def _compare_aggregated_metrics(
        self, base_m: AggregatedMetrics, adapt_m: AggregatedMetrics
    ) -> Dict[str, MetricDelta]:
        """Compare all fields in two AggregatedMetrics objects."""
        deltas: Dict[str, MetricDelta] = {}
        base_dict = base_m.to_dict()
        adapt_dict = adapt_m.to_dict()

        for k, base_v in base_dict.items():
            if isinstance(base_v, (int, float)) and k not in ("total_samples", "valid_responses", "empty_responses"):
                adapt_v = adapt_dict.get(k, 0.0)
                deltas[k] = self._compare_single_metric(k, float(base_v), float(adapt_v))

        return deltas

    def compare_benchmarks(
        self, baseline: OverallBenchmarkReport, adapter: OverallBenchmarkReport
    ) -> RegressionReport:
        """
        Execute comprehensive regression comparison between Baseline and Adapter.
        """
        # Overall Deltas
        overall_deltas = self._compare_aggregated_metrics(
            baseline.overall_metrics, adapter.overall_metrics
        )

        improvements = 0
        regressions = 0
        unchanged = 0

        for d in overall_deltas.values():
            if d.status == "IMPROVED":
                improvements += 1
            elif d.status == "REGRESSED":
                regressions += 1
            else:
                unchanged += 1

        # Domain Deltas
        domain_deltas: Dict[str, GroupRegressionDelta] = {}
        all_domains = set(baseline.domain_metrics.keys()).union(adapter.domain_metrics.keys())
        for dom in sorted(all_domains):
            base_dm = baseline.domain_metrics.get(dom, AggregatedMetrics())
            adapt_dm = adapter.domain_metrics.get(dom, AggregatedMetrics())
            dm_deltas = self._compare_aggregated_metrics(base_dm, adapt_dm)
            domain_deltas[dom] = GroupRegressionDelta(
                group_name=dom,
                group_type="domain",
                metrics=dm_deltas,
            )

        # Difficulty Deltas
        difficulty_deltas: Dict[str, GroupRegressionDelta] = {}
        all_diffs = set(baseline.difficulty_metrics.keys()).union(adapter.difficulty_metrics.keys())
        for diff in sorted(all_diffs):
            base_dfm = baseline.difficulty_metrics.get(diff, AggregatedMetrics())
            adapt_dfm = adapter.difficulty_metrics.get(diff, AggregatedMetrics())
            dfm_deltas = self._compare_aggregated_metrics(base_dfm, adapt_dfm)
            difficulty_deltas[diff] = GroupRegressionDelta(
                group_name=diff,
                group_type="difficulty",
                metrics=dfm_deltas,
            )

        # Task Deltas
        task_deltas: Dict[str, GroupRegressionDelta] = {}
        all_tasks = set(baseline.task_metrics.keys()).union(adapter.task_metrics.keys())
        for task in sorted(all_tasks):
            base_tm = baseline.task_metrics.get(task, AggregatedMetrics())
            adapt_tm = adapter.task_metrics.get(task, AggregatedMetrics())
            tm_deltas = self._compare_aggregated_metrics(base_tm, adapt_tm)
            task_deltas[task] = GroupRegressionDelta(
                group_name=task,
                group_type="task",
                metrics=tm_deltas,
            )

        # Multi-metric Scorecard Verdict
        if improvements > regressions and regressions == 0:
            verdict = "IMPROVED"
            exec_summary = f"Fine-tuned adapter demonstrated clear improvements across {improvements} metrics with 0 regressions."
        elif improvements > regressions:
            verdict = "IMPROVED_WITH_TRADE_OFFS"
            exec_summary = f"Fine-tuned adapter improved {improvements} metrics but showed regressions in {regressions} metrics."
        elif regressions > improvements:
            verdict = "REGRESSED"
            exec_summary = f"Fine-tuned adapter exhibited regressions across {regressions} metrics compared to baseline."
        else:
            verdict = "NEUTRAL"
            exec_summary = f"No significant delta observed between baseline and adapter within {self.config.tolerance_pct}% tolerance."

        return RegressionReport(
            baseline_model=f"{baseline.model_name} ({baseline.model_type})",
            adapter_model=f"{adapter.model_name} ({adapter.model_type})",
            dataset_version=baseline.dataset_version,
            tolerance_pct=self.config.tolerance_pct,
            overall_deltas=overall_deltas,
            domain_deltas=domain_deltas,
            difficulty_deltas=difficulty_deltas,
            task_deltas=task_deltas,
            total_improvements=improvements,
            total_regressions=regressions,
            total_unchanged=unchanged,
            verdict=verdict,
            executive_summary=exec_summary,
        )
