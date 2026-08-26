"""
Evaluation and Benchmarking Architecture (Phase 4.4).
Exposes config, dataset loading, inference, metrics, benchmarking, regression, and reporting modules.
"""

from src.evaluation.benchmark import (
    DIFFICULTIES_4,
    DOMAINS_13,
    BenchmarkEngine,
    OverallBenchmarkReport,
)
from src.evaluation.comparison import ModelComparator
from src.evaluation.config import (
    EvaluationConfig,
    EvaluationDatasetConfig,
    EvaluationModelConfig,
    GenerationConfig,
    MetricsConfig,
    RegressionConfig,
)
from src.evaluation.dataset import (
    EvaluationDatasetError,
    EvaluationDatasetLoader,
    EvaluationExample,
)
from src.evaluation.inference import (
    AdapterCompatibilityError,
    EvaluationInferenceEngine,
    EvaluationInferenceResult,
)
from src.evaluation.metrics import (
    AggregatedMetrics,
    MetricCalculator,
    SampleMetrics,
)
from src.evaluation.regression import (
    GroupRegressionDelta,
    MetricDelta,
    RegressionAnalyzer,
    RegressionReport,
)
from src.evaluation.reports import (
    EvaluationManifest,
    EvaluationReportManager,
)
from src.evaluation.runner import (
    EvaluationPreflightResult,
    EvaluationRunner,
)

__all__ = [
    "EvaluationConfig",
    "EvaluationModelConfig",
    "EvaluationDatasetConfig",
    "GenerationConfig",
    "MetricsConfig",
    "RegressionConfig",
    "EvaluationDatasetError",
    "EvaluationDatasetLoader",
    "EvaluationExample",
    "AdapterCompatibilityError",
    "EvaluationInferenceEngine",
    "EvaluationInferenceResult",
    "SampleMetrics",
    "AggregatedMetrics",
    "MetricCalculator",
    "DOMAINS_13",
    "DIFFICULTIES_4",
    "BenchmarkEngine",
    "OverallBenchmarkReport",
    "MetricDelta",
    "GroupRegressionDelta",
    "RegressionReport",
    "RegressionAnalyzer",
    "ModelComparator",
    "EvaluationManifest",
    "EvaluationReportManager",
    "EvaluationPreflightResult",
    "EvaluationRunner",
]
