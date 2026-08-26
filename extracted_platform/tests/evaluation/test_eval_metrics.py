"""
Unit tests for Metric Calculation (Phase 4.4).
"""

from src.evaluation.inference import EvaluationInferenceResult
from src.evaluation.metrics import MetricCalculator


def test_metric_calculator_valid_sample():
    calc = MetricCalculator()
    res = EvaluationInferenceResult(
        record_id="rec_001",
        domain="programming",
        topic="python",
        task_type="code_generation",
        difficulty="beginner",
        prompt="Write hello world",
        generated_text="Here is the python code:\n```python\nprint('Hello World')\n```",
        reference_text="```python\nprint('Hello World')\n```",
        latency_seconds=0.12,
        tokens_generated=15,
        tokens_per_second=125.0,
        model_type="base",
        model_name="Qwen/Qwen3-4B-Base",
    )

    metrics = calc.calculate_sample_metrics(res)
    assert metrics.is_valid is True
    assert metrics.is_empty is False
    assert metrics.char_length > 0
    assert metrics.word_length > 0
    assert metrics.repetition_ratio == 0.0
    assert metrics.has_repeated_lines is False
    assert metrics.formatting_score == 1.0  # Balanced code blocks
    assert metrics.keyword_overlap > 0.0


def test_metric_calculator_empty_and_repetition():
    calc = MetricCalculator()
    # Empty
    empty_res = EvaluationInferenceResult(
        record_id="rec_empty",
        domain="ai_ml",
        topic="transformers",
        task_type="qa",
        difficulty="intermediate",
        prompt="Explain attention",
        generated_text="   ",
        reference_text="Attention is all you need",
        latency_seconds=0.01,
        tokens_generated=0,
        tokens_per_second=0.0,
        model_type="base",
        model_name="Qwen/Qwen3-4B-Base",
    )
    empty_m = calc.calculate_sample_metrics(empty_res)
    assert empty_m.is_valid is False
    assert empty_m.is_empty is True

    # Degenerate repetition
    rep_text = "loop line\nloop line\nloop line\nloop line\n"
    rep_res = EvaluationInferenceResult(
        record_id="rec_rep",
        domain="ai_ml",
        topic="transformers",
        task_type="qa",
        difficulty="intermediate",
        prompt="Loop test",
        generated_text=rep_text,
        reference_text="Normal text",
        latency_seconds=0.05,
        tokens_generated=8,
        tokens_per_second=160.0,
        model_type="base",
        model_name="Qwen/Qwen3-4B-Base",
    )
    rep_m = calc.calculate_sample_metrics(rep_res)
    assert rep_m.has_repeated_lines is True
    assert rep_m.repetition_ratio > 0.0


def test_metric_aggregation():
    calc = MetricCalculator()
    s1 = EvaluationInferenceResult(
        record_id="rec_1",
        domain="programming",
        topic="python",
        task_type="coding",
        difficulty="beginner",
        prompt="p1",
        generated_text="valid text one.",
        reference_text="valid text one.",
        latency_seconds=0.1,
        tokens_generated=10,
        tokens_per_second=100.0,
        model_type="base",
        model_name="Qwen/Qwen3-4B-Base",
    )
    s2 = EvaluationInferenceResult(
        record_id="rec_2",
        domain="programming",
        topic="python",
        task_type="coding",
        difficulty="beginner",
        prompt="p2",
        generated_text="",
        reference_text="valid text two.",
        latency_seconds=0.05,
        tokens_generated=0,
        tokens_per_second=0.0,
        model_type="base",
        model_name="Qwen/Qwen3-4B-Base",
    )

    m1 = calc.calculate_sample_metrics(s1)
    m2 = calc.calculate_sample_metrics(s2)

    agg = calc.aggregate_metrics([m1, m2], [s1, s2])
    assert agg.total_samples == 2
    assert agg.valid_responses == 1
    assert agg.validity_rate == 0.5
    assert agg.empty_responses == 1
    assert agg.empty_rate == 0.5
    assert agg.exact_match_rate == 0.5
