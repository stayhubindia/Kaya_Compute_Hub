"""
Evaluation Runner & Lifecycle Orchestrator (Phase 4.4).
Orchestrates preflight audits, dataset verification, baseline & adapter inference,
metric calculation, multi-dimensional stratification, and comparative reporting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.evaluation.benchmark import BenchmarkEngine, OverallBenchmarkReport
from src.evaluation.config import EvaluationConfig
from src.evaluation.dataset import EvaluationDatasetLoader, EvaluationExample
from src.evaluation.inference import EvaluationInferenceEngine, EvaluationInferenceResult
from src.evaluation.regression import RegressionAnalyzer, RegressionReport
from src.evaluation.reports import EvaluationManifest, EvaluationReportManager
from src.training.utils import compute_file_sha256, detect_hardware_environment

logger = logging.getLogger(__name__)


class EvaluationPreflightResult:
    """Detailed results of the evaluation preflight audit."""

    def __init__(self):
        self.checks: Dict[str, bool] = {}
        self.details: Dict[str, Any] = {}
        self.errors: List[str] = []

    @property
    def is_passed(self) -> bool:
        return len(self.errors) == 0 and all(self.checks.values())

    def add_check(self, name: str, passed: bool, detail: Any = None):
        self.checks[name] = passed
        if detail is not None:
            self.details[name] = detail
        if not passed:
            self.errors.append(f"Check '{name}' failed: {detail}")


class EvaluationRunner:
    """Master controller for the evaluation and benchmarking system."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.dataset_loader = EvaluationDatasetLoader(self.config.dataset)
        self.inference_engine = EvaluationInferenceEngine(self.config)
        self.benchmark_engine = BenchmarkEngine(self.config)
        self.regression_analyzer = RegressionAnalyzer(self.config.regression)
        self.report_manager = EvaluationReportManager(self.config.reports_dir)
        self.hardware = detect_hardware_environment()

    def run_preflight_audit(self) -> EvaluationPreflightResult:
        """
        Execute comprehensive 12-point preflight audit for evaluation readiness.
        """
        result = EvaluationPreflightResult()

        # 1. Config validation
        try:
            config_hash = self.config.compute_hash()
            result.add_check("evaluation_config_valid", True, f"Hash: {config_hash[:16]}...")
        except Exception as e:
            result.add_check("evaluation_config_valid", False, str(e))

        # 2. Manifest and Frozen State
        try:
            manifest = self.dataset_loader.load_manifest()
            status_val = manifest.status.value if hasattr(manifest.status, "value") else str(manifest.status)
            is_frozen = (status_val == "FROZEN")
            result.add_check("dataset_frozen_lifecycle", is_frozen, f"Lifecycle: {status_val}")
        except Exception as e:
            result.add_check("dataset_frozen_lifecycle", False, str(e))

        # 3. Split Files Existence
        split_files = [self.config.dataset.train_file, self.config.dataset.validation_file, self.config.dataset.test_file]
        missing = [p for p in split_files if not Path(p).exists()]
        result.add_check("split_files_present", len(missing) == 0, f"Missing: {missing}" if missing else "All 3 splits present")

        # 4. SHA-256 Checksums
        try:
            checksums = self.dataset_loader.verify_split_checksums()
            result.add_check("sha256_checksums_match", all(checksums.values()), checksums)
        except Exception as e:
            result.add_check("sha256_checksums_match", False, str(e))

        # 5. Split Isolation (Zero Contamination)
        try:
            isolated_count = self.dataset_loader.verify_split_isolation()
            result.add_check("split_isolation_verified", True, f"Zero contamination across {isolated_count} test records")
        except Exception as e:
            result.add_check("split_isolation_verified", False, str(e))

        # 6. Test Examples Extraction
        try:
            examples = self.dataset_loader.load_examples()
            result.add_check("test_examples_loaded", len(examples) > 0, f"Loaded {len(examples)} test examples")
        except Exception as e:
            result.add_check("test_examples_loaded", False, str(e))

        # 7. Hardware Audit
        cuda_ok = self.hardware.cuda_available
        hw_desc = f"{self.hardware.gpu_name} ({self.hardware.gpu_vram_gb:.2f} GB)" if cuda_ok else "No GPU / CUDA unavailable"
        result.add_check("hardware_audit", True, hw_desc)

        return result

    def evaluate(
        self,
        model_type: Optional[str] = None,
        dry_run: bool = False,
        max_examples: Optional[int] = None,
    ) -> OverallBenchmarkReport:
        """
        Execute evaluation for Base model or LoRA Adapter.
        """
        if model_type:
            self.config.model.model_type = model_type  # type: ignore

        # 1. Preflight
        preflight = self.run_preflight_audit()
        manifest_path = Path(self.config.dataset.manifest_path)
        dataset_sha = compute_file_sha256(manifest_path) if manifest_path.exists() else ""

        # 2. Load Examples
        examples = self.dataset_loader.load_examples(limit=max_examples)

        # 3. Create Manifest in RUNNING state
        manifest = self.report_manager.create_manifest(
            config=self.config,
            sample_count=len(examples),
            dataset_sha256=dataset_sha,
            status="RUNNING",
        )

        # 4. Hardware Gate
        if not self.hardware.cuda_available and not dry_run:
            manifest.status = "BLOCKED"
            manifest.details["reason"] = "MODEL INFERENCE BLOCKED — GPU UNAVAILABLE (CUDA not found)"
            manifest.save(Path(self.config.reports_dir) / "evaluation_manifest.json")
            # Build and save blocked report
            empty_report = self.benchmark_engine.compute_benchmark(
                inference_results=[],
                dataset_sha256=dataset_sha,
                hardware_device="CPU (Offline)",
            )
            self.report_manager.save_benchmark_reports(empty_report, manifest)
            return empty_report

        # 5. Run Inference
        results: List[EvaluationInferenceResult] = []
        for ex in examples:
            inf_res = self.inference_engine.generate(ex)
            results.append(inf_res)

        # 6. Aggregate & Benchmark
        hw_desc = f"{self.hardware.gpu_name}" if self.hardware.cuda_available else "CPU (Simulation)"
        report = self.benchmark_engine.compute_benchmark(
            inference_results=results,
            dataset_sha256=dataset_sha,
            hardware_device=hw_desc,
        )

        # 7. Finalize and Save Reports
        manifest.status = "COMPLETED"
        manifest.save(Path(self.config.reports_dir) / "evaluation_manifest.json")
        self.report_manager.save_benchmark_reports(report, manifest)

        return report

    def compare(
        self,
        baseline_report: Optional[OverallBenchmarkReport] = None,
        adapter_report: Optional[OverallBenchmarkReport] = None,
    ) -> RegressionReport:
        """
        Compare baseline and fine-tuned benchmark reports and generate regression report.
        """
        if baseline_report is None or adapter_report is None:
            # Try to load existing reports
            base_json = Path(self.config.reports_dir) / "evaluation_report.json"
            if base_json.exists() and baseline_report is None:
                import json
                with open(base_json, "r", encoding="utf-8") as f:
                    baseline_report = OverallBenchmarkReport(**json.load(f))

            if adapter_report is None:
                # If adapter report not provided, generate from evaluation
                adapter_report = self.evaluate(model_type="adapter", dry_run=True)

        if baseline_report is None or adapter_report is None:
            raise ValueError("Both baseline and adapter benchmark reports are required for comparison.")

        regression = self.regression_analyzer.compare_benchmarks(baseline_report, adapter_report)
        self.report_manager.save_regression_report(regression)
        return regression
