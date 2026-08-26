"""
Unit tests for Domain, Difficulty, and Task Benchmark Engine (Phase 4.4).
"""

from src.evaluation.benchmark import BenchmarkEngine, DOMAINS_13, DIFFICULTIES_4
from src.evaluation.config import EvaluationConfig
from src.evaluation.inference import EvaluationInferenceResult


def test_benchmark_stratification():
    config = EvaluationConfig()
    engine = BenchmarkEngine(config)

    # Synthetic inference sample spanning multiple domains & difficulties
    inferences = [
        EvaluationInferenceResult(
            record_id="rec_p1",
            domain="programming",
            topic="algorithms",
            task_type="code_generation",
            difficulty="beginner",
            prompt="p1",
            generated_text="def foo(): return 1",
            reference_text="def foo(): return 1",
            latency_seconds=0.1,
            tokens_generated=8,
            tokens_per_second=80.0,
            model_type="base",
            model_name="Qwen/Qwen3-4B-Base",
        ),
        EvaluationInferenceResult(
            record_id="rec_m1",
            domain="mathematics",
            topic="calculus",
            task_type="calculation",
            difficulty="advanced",
            prompt="p2",
            generated_text="x = 42",
            reference_text="x = 42",
            latency_seconds=0.2,
            tokens_generated=5,
            tokens_per_second=25.0,
            model_type="base",
            model_name="Qwen/Qwen3-4B-Base",
        ),
    ]

    report = engine.compute_benchmark(inferences, dataset_sha256="testsha123", hardware_device="CPU")
    assert report.sample_count == 2
    assert report.overall_metrics.validity_rate == 1.0
    assert "programming" in report.domain_metrics
    assert "mathematics" in report.domain_metrics
    assert "beginner" in report.difficulty_metrics
    assert "advanced" in report.difficulty_metrics
    assert "code_generation" in report.task_metrics
    assert "calculation" in report.task_metrics
    assert report.domain_metrics["programming"].total_samples == 1
    assert report.domain_metrics["mathematics"].total_samples == 1
