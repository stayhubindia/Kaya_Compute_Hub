"""
Evaluation Reports & Manifest Manager (Phase 4.4).
Manages serialization of structured JSON and Markdown evaluation reports,
domain scorecards, difficulty matrices, regression comparisons, and evaluation manifests.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import transformers
from pydantic import BaseModel, Field

from src.evaluation.benchmark import OverallBenchmarkReport
from src.evaluation.comparison import ModelComparator
from src.evaluation.config import EvaluationConfig
from src.evaluation.regression import RegressionReport
from src.training.utils import detect_hardware_environment


class EvaluationManifest(BaseModel):
    """Manifest tracking evaluation run metadata, hardware state, and lifecycle status."""
    evaluation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_name: str
    model_type: str
    adapter_path: Optional[str] = None
    dataset_version: str
    dataset_sha256: str
    config_hash: str
    hardware: str
    cuda_available: bool
    pytorch_version: str
    transformers_version: str
    seed: int
    sample_count: int
    status: str = "PLANNED"  # 'PLANNED', 'RUNNING', 'COMPLETED', 'BLOCKED', 'FAILED'
    details: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> EvaluationManifest:
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


class EvaluationReportManager:
    """Orchestrates writing and loading of all evaluation reports and manifest artifacts."""

    def __init__(self, reports_dir: Union[str, Path] = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def create_manifest(
        self,
        config: EvaluationConfig,
        sample_count: int,
        dataset_sha256: str,
        evaluation_id: Optional[str] = None,
        status: str = "PLANNED",
    ) -> EvaluationManifest:
        """Create and persist an EvaluationManifest."""
        hw = detect_hardware_environment()
        hw_desc = f"{hw.gpu_name} ({hw.gpu_count}x)" if hw.cuda_available else "CPU (Offline Mode)"

        eval_id = evaluation_id or f"eval-{config.model.model_type}-{int(datetime.now(timezone.utc).timestamp())}"
        manifest = EvaluationManifest(
            evaluation_id=eval_id,
            model_name=config.model.name,
            model_type=config.model.model_type,
            adapter_path=config.model.adapter_path if config.model.model_type == "adapter" else None,
            dataset_version=config.dataset.version,
            dataset_sha256=dataset_sha256,
            config_hash=config.compute_hash(),
            hardware=hw_desc,
            cuda_available=hw.cuda_available,
            pytorch_version=torch.__version__,
            transformers_version=transformers.__version__,
            seed=config.seed,
            sample_count=sample_count,
            status=status,
        )

        manifest_path = self.reports_dir / "evaluation_manifest.json"
        manifest.save(manifest_path)
        return manifest

    def save_benchmark_reports(self, report: OverallBenchmarkReport, manifest: Optional[EvaluationManifest] = None) -> None:
        """Save overall, domain, difficulty, and task evaluation reports."""
        # 1. Master JSON
        eval_json_path = self.reports_dir / "evaluation_report.json"
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        # 2. Domain JSON
        domain_json_path = self.reports_dir / "domain_report.json"
        with open(domain_json_path, "w", encoding="utf-8") as f:
            dom_data = {k: v.to_dict() for k, v in report.domain_metrics.items()}
            json.dump(dom_data, f, indent=2)

        # 3. Difficulty JSON
        diff_json_path = self.reports_dir / "difficulty_report.json"
        with open(diff_json_path, "w", encoding="utf-8") as f:
            diff_data = {k: v.to_dict() for k, v in report.difficulty_metrics.items()}
            json.dump(diff_data, f, indent=2)

        # 4. Task JSON
        task_json_path = self.reports_dir / "task_report.json"
        with open(task_json_path, "w", encoding="utf-8") as f:
            task_data = {k: v.to_dict() for k, v in report.task_metrics.items()}
            json.dump(task_data, f, indent=2)

        # 5. Master Markdown Report
        eval_md_path = self.reports_dir / "evaluation_report.md"
        with open(eval_md_path, "w", encoding="utf-8") as f:
            f.write(self._render_evaluation_markdown(report, manifest))

    def save_regression_report(self, regression: RegressionReport) -> None:
        """Save regression report in JSON and Markdown."""
        reg_json_path = self.reports_dir / "regression_report.json"
        with open(reg_json_path, "w", encoding="utf-8") as f:
            json.dump(regression.to_dict(), f, indent=2)

        reg_md_path = self.reports_dir / "regression_report.md"
        with open(reg_md_path, "w", encoding="utf-8") as f:
            f.write(ModelComparator.generate_full_comparison_report(regression))

    def _render_evaluation_markdown(
        self, report: OverallBenchmarkReport, manifest: Optional[EvaluationManifest] = None
    ) -> str:
        """Render comprehensive markdown report for a single evaluation benchmark."""
        m = report.overall_metrics
        status_banner = "COMPLETED" if report.sample_count > 0 and not report.is_mock else "EVALUATION INFRASTRUCTURE READY (MOCK / OFFLINE)"

        lines = [
            f"# Model Evaluation & Benchmark Report — {status_banner}",
            "",
            f"**Model:** `{report.model_name}` (`{report.model_type}`)  ",
            f"**Dataset Version:** `{report.dataset_version}` (`FROZEN`)  ",
            f"**Dataset SHA-256:** `{report.dataset_sha256}`  ",
            f"**Evaluation Timestamp:** `{report.timestamp}`  ",
            f"**Hardware Device:** `{report.hardware_device}`  ",
            f"**Mock Mode:** `{report.is_mock}`  ",
            f"**Total Samples Evaluated:** `{report.sample_count}`  ",
            "",
            "## 1. Overall Aggregated Metrics",
            "",
            "| Metric | Value | Metric | Value |",
            "| :--- | :--- | :--- | :--- |",
            f"| `Validity Rate` | {m.validity_rate:.2%} | `Empty Rate` | {m.empty_rate:.2%} |",
            f"| `Avg Formatting Score` | {m.avg_formatting_score:.4f} | `Avg Repetition Ratio` | {m.avg_repetition_ratio:.4f} |",
            f"| `Truncation Rate` | {m.truncation_rate:.2%} | `Repeated Lines Rate` | {m.repeated_lines_rate:.2%} |",
            f"| `Avg Token Length` | {m.avg_token_length:.1f} | `Avg Char Length` | {m.avg_char_length:.1f} |",
            f"| `Exact Match Rate` | {m.exact_match_rate:.2%} | `Keyword Overlap` | {m.avg_keyword_overlap:.4f} |",
            f"| `Avg Latency (s)` | {m.avg_latency_seconds:.4f}s | `Tokens / Second` | {m.avg_tokens_per_second:.2f} |",
            "",
            "## 2. Sequence Length Distribution",
            "",
            f"- **Mean Token Length:** `{m.avg_token_length:.2f}`",
            f"- **P50 / P90 / P95 / P99:** `{m.p50_token_length:.1f}` / `{m.p90_token_length:.1f}` / `{m.p95_token_length:.1f}` / `{m.p99_token_length:.1f}`",
            "",
            "## 3. Domain Performance Breakdown",
            "",
            "| Domain | Total | Validity | Repetition | Formatting | Avg Tokens | Latency (s) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for dom, dm in sorted(report.domain_metrics.items()):
            lines.append(
                f"| `{dom}` | {dm.total_samples} | {dm.validity_rate:.2%} | {dm.avg_repetition_ratio:.4f} | "
                f"{dm.avg_formatting_score:.2f} | {dm.avg_token_length:.1f} | {dm.avg_latency_seconds:.4f}s |"
            )

        lines.extend([
            "",
            "## 4. Difficulty Breakdown",
            "",
            "| Difficulty | Total | Validity | Repetition | Formatting | Avg Tokens |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for diff, dfm in sorted(report.difficulty_metrics.items()):
            lines.append(
                f"| `{diff}` | {dfm.total_samples} | {dfm.validity_rate:.2%} | {dfm.avg_repetition_ratio:.4f} | "
                f"{dfm.avg_formatting_score:.2f} | {dfm.avg_token_length:.1f} |"
            )

        return "\n".join(lines)
