"""
Unit tests for Regression Framework & Delta Analysis (Phase 4.4).
"""

from src.evaluation.benchmark import OverallBenchmarkReport
from src.evaluation.config import RegressionConfig
from src.evaluation.metrics import AggregatedMetrics
from src.evaluation.regression import RegressionAnalyzer


def test_regression_analyzer_improvements_and_regressions():
    cfg = RegressionConfig(tolerance_pct=5.0)
    analyzer = RegressionAnalyzer(cfg)

    base_overall = AggregatedMetrics(
        total_samples=10,
        valid_responses=8,
        validity_rate=0.80,
        empty_responses=2,
        empty_rate=0.20,
        avg_formatting_score=0.70,
        avg_repetition_ratio=0.15,
        avg_token_length=50.0,
    )

    # Adapter improves formatting and reduces repetition, but is slightly slower
    adapt_overall = AggregatedMetrics(
        total_samples=10,
        valid_responses=10,
        validity_rate=1.00,
        empty_responses=0,
        empty_rate=0.00,
        avg_formatting_score=0.95,
        avg_repetition_ratio=0.02,
        avg_token_length=55.0,
    )

    base_rep = OverallBenchmarkReport(
        model_name="Qwen3-4B-Base",
        model_type="base",
        dataset_version="dataset-v1.0",
        sample_count=10,
        overall_metrics=base_overall,
    )

    adapt_rep = OverallBenchmarkReport(
        model_name="Qwen3-4B-Base",
        model_type="adapter",
        dataset_version="dataset-v1.0",
        sample_count=10,
        overall_metrics=adapt_overall,
    )

    regression = analyzer.compare_benchmarks(base_rep, adapt_rep)
    assert regression.total_improvements >= 4
    assert regression.verdict in ("IMPROVED", "IMPROVED_WITH_TRADE_OFFS")

    # Check validity rate delta
    val_delta = regression.overall_deltas["validity_rate"]
    assert val_delta.baseline_value == 0.80
    assert val_delta.adapter_value == 1.00
    assert val_delta.absolute_delta == 0.20
    assert val_delta.status == "IMPROVED"

    # Check repetition ratio delta (lower is better)
    rep_delta = regression.overall_deltas["avg_repetition_ratio"]
    assert rep_delta.status == "IMPROVED"
