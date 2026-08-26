"""
Experiment Lifecycle, Manifest & Model Comparison Engine (Phase 4.7).
Manages experiment manifests, atomic persistence, resume capability,
and multi-dimensional regression comparison between Baseline and Adapter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from src.evaluation.execution import CaseExecutionResult, GenerationConfig
from src.training.utils import compute_file_sha256, detect_hardware_environment

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class CaseChangeStatus(str, Enum):
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    UNCHANGED = "UNCHANGED"


class EvaluationManifest(BaseModel):
    """Immutable experiment manifest tracking full provenance, hashes, and execution telemetry."""
    experiment_id: str
    model: str  # 'base' or 'adapter'
    adapter_name: Optional[str] = None
    benchmark_version: str = "benchmark-v1.0"
    benchmark_sha256: str = ""
    generation_config: Dict[str, Any] = Field(default_factory=dict)
    generation_config_sha256: str = ""
    seed: int = 42
    hardware: Dict[str, Any] = Field(default_factory=dict)
    software_versions: Dict[str, str] = Field(default_factory=dict)
    start_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    case_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    artifact_hashes: Dict[str, str] = Field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.PLANNED
    status_reason: Optional[str] = None

    def save_atomic(self, path: Union[str, Path]) -> None:
        """Atomically persist manifest to disk via temp file replacement."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = p.with_suffix(".tmp")
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
        tmp_p.replace(p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> EvaluationManifest:
        """Load manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


class ExperimentManager:
    """Manages experiment directory allocation, atomic output writing, and resume states."""

    def __init__(self, base_dir: Union[str, Path] = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment_id(self, benchmark_version: str, model_type: str) -> str:
        """Generate unique, structured experiment ID."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"eval-{benchmark_version}-{model_type}-{ts}"

    def init_manifest(
        self,
        experiment_id: str,
        model_type: str,
        benchmark_version: str,
        benchmark_sha256: str,
        gen_config: GenerationConfig,
        case_count: int,
        adapter_name: Optional[str] = None,
    ) -> EvaluationManifest:
        """Initialize evaluation manifest with system, software, and hardware telemetry."""
        import torch
        import transformers

        hw = detect_hardware_environment()
        hw_info = {
            "gpu_name": hw.device_name or "None",
            "gpu_count": hw.device_count,
            "gpu_vram_gb": round(hw.total_memory_gb, 2),
            "cuda_available": hw.cuda_available,
            "platform": platform.platform(),
        }
        sw_info = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }

        return EvaluationManifest(
            experiment_id=experiment_id,
            model=model_type,
            adapter_name=adapter_name,
            benchmark_version=benchmark_version,
            benchmark_sha256=benchmark_sha256,
            generation_config=gen_config.model_dump(),
            generation_config_sha256=gen_config.compute_hash(),
            seed=gen_config.seed,
            hardware=hw_info,
            software_versions=sw_info,
            case_count=case_count,
            status=ExperimentStatus.PLANNED,
        )

    @staticmethod
    def load_completed_case_ids(results_file: Union[str, Path]) -> Set[str]:
        """Read completed benchmark case IDs for seamless resumption."""
        p = Path(results_file)
        if not p.exists():
            return set()
        completed = set()
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get("status") == "COMPLETED":
                            completed.add(record["benchmark_id"])
                    except Exception:
                        continue
        return completed

    @staticmethod
    def append_case_result(results_file: Union[str, Path], result: CaseExecutionResult) -> None:
        """Append case execution result atomically to results file."""
        p = Path(results_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")


class CaseComparison(BaseModel):
    """Detailed side-by-side comparison for a single benchmark case."""
    benchmark_id: str
    domain: str
    difficulty: str
    task_type: str
    baseline_status: str
    adapter_status: str
    baseline_keyword_overlap: float = 0.0
    adapter_keyword_overlap: float = 0.0
    baseline_formatting_score: float = 1.0
    adapter_formatting_score: float = 1.0
    classification: CaseChangeStatus = CaseChangeStatus.UNCHANGED
    notes: List[str] = Field(default_factory=list)


class ModelComparisonReport(BaseModel):
    """Aggregated model comparison report between Baseline and LoRA Adapter."""
    baseline_experiment_id: str
    adapter_experiment_id: str
    benchmark_version: str
    benchmark_sha256: str
    generation_config_sha256: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cases_total: int = 0
    cases_improved: int = 0
    cases_regressed: int = 0
    cases_unchanged: int = 0
    cases_failed_baseline: int = 0
    cases_failed_adapter: int = 0
    domain_deltas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    difficulty_deltas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    task_deltas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    strongest_improvement_domain: Optional[str] = None
    largest_regression_domain: Optional[str] = None
    performance_comparison: Dict[str, Any] = Field(default_factory=dict)
    overall_verdict: str = "NEUTRAL"

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ModelComparisonEngine:
    """Compares baseline and adapter evaluation runs under strictly identical configurations."""

    @staticmethod
    def compare_experiments(
        baseline_manifest: EvaluationManifest,
        adapter_manifest: EvaluationManifest,
        baseline_results: List[CaseExecutionResult],
        adapter_results: List[CaseExecutionResult],
    ) -> Tuple[ModelComparisonReport, List[CaseComparison]]:
        """Perform multi-dimensional comparative regression analysis."""
        # 1. Strict Configuration Verification
        if baseline_manifest.benchmark_version != adapter_manifest.benchmark_version:
            raise ValueError(
                f"Benchmark version mismatch: baseline='{baseline_manifest.benchmark_version}' vs adapter='{adapter_manifest.benchmark_version}'"
            )
        if baseline_manifest.benchmark_sha256 != adapter_manifest.benchmark_sha256:
            raise ValueError(
                f"Benchmark SHA-256 mismatch: baseline='{baseline_manifest.benchmark_sha256}' vs adapter='{adapter_manifest.benchmark_sha256}'"
            )
        if baseline_manifest.generation_config_sha256 != adapter_manifest.generation_config_sha256:
            raise ValueError(
                f"Generation config hash mismatch: baseline='{baseline_manifest.generation_config_sha256}' vs adapter='{adapter_manifest.generation_config_sha256}'"
            )

        # 2. Map Results by Benchmark ID
        base_map = {r.benchmark_id: r for r in baseline_results}
        adapt_map = {r.benchmark_id: r for r in adapter_results}
        all_ids = sorted(set(base_map.keys()).union(adapt_map.keys()))

        comparisons: List[CaseComparison] = []
        improved_cnt, regressed_cnt, unchanged_cnt = 0, 0, 0
        failed_base, failed_adapt = 0, 0

        domain_groups: Dict[str, List[CaseComparison]] = {}
        diff_groups: Dict[str, List[CaseComparison]] = {}
        task_groups: Dict[str, List[CaseComparison]] = {}

        for b_id in all_ids:
            base_r = base_map.get(b_id)
            adapt_r = adapt_map.get(b_id)

            if not base_r or not adapt_r:
                continue

            domain = base_r.domain
            diff = base_r.difficulty
            task = base_r.task_type

            if base_r.status != "COMPLETED":
                failed_base += 1
            if adapt_r.status != "COMPLETED":
                failed_adapt += 1

            base_kw = float(base_r.metrics.get("keyword_overlap", 0.0))
            adapt_kw = float(adapt_r.metrics.get("keyword_overlap", 0.0))
            base_fmt = float(base_r.metrics.get("formatting_score", 1.0))
            adapt_fmt = float(adapt_r.metrics.get("formatting_score", 1.0))

            # Direction-aware case classification
            kw_delta = adapt_kw - base_kw
            fmt_delta = adapt_fmt - base_fmt
            composite_delta = kw_delta * 0.7 + fmt_delta * 0.3

            if composite_delta > 0.05 or (base_r.status == "FAILED" and adapt_r.status == "COMPLETED"):
                classification = CaseChangeStatus.IMPROVED
                improved_cnt += 1
            elif composite_delta < -0.05 or (base_r.status == "COMPLETED" and adapt_r.status == "FAILED"):
                classification = CaseChangeStatus.REGRESSED
                regressed_cnt += 1
            else:
                classification = CaseChangeStatus.UNCHANGED
                unchanged_cnt += 1

            cmp = CaseComparison(
                benchmark_id=b_id,
                domain=domain,
                difficulty=diff,
                task_type=task,
                baseline_status=base_r.status,
                adapter_status=adapt_r.status,
                baseline_keyword_overlap=round(base_kw, 4),
                adapter_keyword_overlap=round(adapt_kw, 4),
                baseline_formatting_score=round(base_fmt, 4),
                adapter_formatting_score=round(adapt_fmt, 4),
                classification=classification,
            )
            comparisons.append(cmp)
            domain_groups.setdefault(domain, []).append(cmp)
            diff_groups.setdefault(diff, []).append(cmp)
            task_groups.setdefault(task, []).append(cmp)

        # 3. Stratified Deltas
        domain_deltas: Dict[str, Dict[str, Any]] = {}
        for dom, cmps in sorted(domain_groups.items()):
            imp = sum(1 for c in cmps if c.classification == CaseChangeStatus.IMPROVED)
            reg = sum(1 for c in cmps if c.classification == CaseChangeStatus.REGRESSED)
            unc = sum(1 for c in cmps if c.classification == CaseChangeStatus.UNCHANGED)
            net = imp - reg
            domain_deltas[dom] = {
                "total": len(cmps),
                "improved": imp,
                "regressed": reg,
                "unchanged": unc,
                "net_change": net,
            }

        diff_deltas: Dict[str, Dict[str, Any]] = {}
        for diff, cmps in sorted(diff_groups.items()):
            imp = sum(1 for c in cmps if c.classification == CaseChangeStatus.IMPROVED)
            reg = sum(1 for c in cmps if c.classification == CaseChangeStatus.REGRESSED)
            unc = sum(1 for c in cmps if c.classification == CaseChangeStatus.UNCHANGED)
            diff_deltas[diff] = {
                "total": len(cmps),
                "improved": imp,
                "regressed": reg,
                "unchanged": unc,
                "net_change": imp - reg,
            }

        task_deltas: Dict[str, Dict[str, Any]] = {}
        for task, cmps in sorted(task_groups.items()):
            imp = sum(1 for c in cmps if c.classification == CaseChangeStatus.IMPROVED)
            reg = sum(1 for c in cmps if c.classification == CaseChangeStatus.REGRESSED)
            unc = sum(1 for c in cmps if c.classification == CaseChangeStatus.UNCHANGED)
            task_deltas[task] = {
                "total": len(cmps),
                "improved": imp,
                "regressed": reg,
                "unchanged": unc,
                "net_change": imp - reg,
            }

        # 4. Domain Extremes
        strongest_dom = max(domain_deltas.items(), key=lambda x: x[1]["net_change"])[0] if domain_deltas else None
        largest_reg_dom = min(domain_deltas.items(), key=lambda x: x[1]["net_change"])[0] if domain_deltas else None

        # 5. Performance Comparison
        hw_match = baseline_manifest.hardware.get("gpu_name") == adapter_manifest.hardware.get("gpu_name")
        perf = {
            "hardware_match": hw_match,
            "status": "VALID_COMPARISON" if hw_match else "HARDWARE_MISMATCH",
            "baseline_gpu": baseline_manifest.hardware.get("gpu_name", "Unknown"),
            "adapter_gpu": adapter_manifest.hardware.get("gpu_name", "Unknown"),
        }

        # 6. Overall Verdict
        if improved_cnt > regressed_cnt and regressed_cnt == 0:
            verdict = "IMPROVED"
        elif improved_cnt > regressed_cnt:
            verdict = "IMPROVED_WITH_TRADE_OFFS"
        elif regressed_cnt > improved_cnt:
            verdict = "REGRESSED"
        else:
            verdict = "NEUTRAL"

        report = ModelComparisonReport(
            baseline_experiment_id=baseline_manifest.experiment_id,
            adapter_experiment_id=adapter_manifest.experiment_id,
            benchmark_version=baseline_manifest.benchmark_version,
            benchmark_sha256=baseline_manifest.benchmark_sha256,
            generation_config_sha256=baseline_manifest.generation_config_sha256,
            cases_total=len(comparisons),
            cases_improved=improved_cnt,
            cases_regressed=regressed_cnt,
            cases_unchanged=unchanged_cnt,
            cases_failed_baseline=failed_base,
            cases_failed_adapter=failed_adapt,
            domain_deltas=domain_deltas,
            difficulty_deltas=diff_deltas,
            task_deltas=task_deltas,
            strongest_improvement_domain=strongest_dom,
            largest_regression_domain=largest_reg_dom,
            performance_comparison=perf,
            overall_verdict=verdict,
        )

        return report, comparisons

    @classmethod
    def generate_markdown_report(cls, report: ModelComparisonReport) -> str:
        """Render formatted comparison report in markdown."""
        lines = [
            "# Benchmark Evaluation Model Comparison Report (Phase 4.7)",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Baseline Experiment:** `{report.baseline_experiment_id}`",
            f"- **Adapter Experiment:** `{report.adapter_experiment_id}`",
            f"- **Benchmark Version:** `{report.benchmark_version}`",
            f"- **Benchmark SHA-256:** `{report.benchmark_sha256}`",
            f"- **Generation Config SHA:** `{report.generation_config_sha256}`",
            f"- **Overall Verdict:** **`{report.overall_verdict}`**",
            "",
            "## 2. Per-Case Regression Summary",
            "",
            "| Metric | Count | Percentage |",
            "| :--- | :--- | :--- |",
            f"| **Total Cases Evaluated** | {report.cases_total} | 100.0% |",
            f"| **Cases Improved** | {report.cases_improved} | {report.cases_improved / max(report.cases_total, 1) * 100:.1f}% |",
            f"| **Cases Regressed** | {report.cases_regressed} | {report.cases_regressed / max(report.cases_total, 1) * 100:.1f}% |",
            f"| **Cases Unchanged** | {report.cases_unchanged} | {report.cases_unchanged / max(report.cases_total, 1) * 100:.1f}% |",
            f"| **Baseline Failures** | {report.cases_failed_baseline} | - |",
            f"| **Adapter Failures** | {report.cases_failed_adapter} | - |",
            "",
            "## 3. Domain-Level Breakdown",
            "",
            "| Domain | Total | Improved | Regressed | Unchanged | Net Change |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for dom, stats in sorted(report.domain_deltas.items()):
            lines.append(
                f"| `{dom}` | {stats['total']} | {stats['improved']} | {stats['regressed']} | {stats['unchanged']} | **{stats['net_change']:+d}** |"
            )

        lines.extend([
            "",
            f"- **Strongest Improvement Domain:** `{report.strongest_improvement_domain}`",
            f"- **Largest Regression Domain:** `{report.largest_regression_domain}`",
            "",
            "## 4. Difficulty Breakdown",
            "",
            "| Difficulty Tier | Total | Improved | Regressed | Unchanged | Net Change |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for diff, stats in sorted(report.difficulty_deltas.items()):
            lines.append(
                f"| `{diff}` | {stats['total']} | {stats['improved']} | {stats['regressed']} | {stats['unchanged']} | **{stats['net_change']:+d}** |"
            )

        lines.extend([
            "",
            "## 5. Performance & Hardware Verification",
            "",
            f"- **Hardware Match:** `{report.performance_comparison.get('hardware_match')}`",
            f"- **Status:** `{report.performance_comparison.get('status')}`",
            f"- **Baseline GPU:** `{report.performance_comparison.get('baseline_gpu')}`",
            f"- **Adapter GPU:** `{report.performance_comparison.get('adapter_gpu')}`",
        ])

        return "\n".join(lines)

    @classmethod
    def save_comparison_reports(
        cls,
        report: ModelComparisonReport,
        output_dir: Union[str, Path] = "reports",
    ) -> None:
        """Save comparison report in JSON and Markdown formats."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "model_comparison.json", "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        with open(out / "model_comparison.md", "w", encoding="utf-8") as f:
            f.write(cls.generate_markdown_report(report))
