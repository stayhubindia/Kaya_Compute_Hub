"""
Model Comparison Formatter & Visualizer (Phase 4.4).
Generates side-by-side comparison tables and markdown representations
contrasting Baseline model results with Fine-Tuned LoRA Adapter results.
"""

from __future__ import annotations

from typing import Any, Dict, List
from src.evaluation.benchmark import OverallBenchmarkReport
from src.evaluation.regression import RegressionReport


class ModelComparator:
    """Formats side-by-side comparison matrices for reporting."""

    @staticmethod
    def generate_overall_comparison_markdown(regression: RegressionReport) -> str:
        """Render markdown table of overall metric comparisons."""
        lines = [
            "## 1. Overall Performance Comparison: Baseline vs Fine-Tuned",
            "",
            f"**Baseline:** `{regression.baseline_model}`  ",
            f"**Fine-Tuned Adapter:** `{regression.adapter_model}`  ",
            f"**Dataset Version:** `{regression.dataset_version}`  ",
            f"**Verdict:** `{regression.verdict}`  ",
            f"**Executive Summary:** {regression.executive_summary}",
            "",
            "| Metric | Baseline | Fine-Tuned | Delta (Y-X) | Change (%) | Status |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        status_icons = {
            "IMPROVED": "🟢 IMPROVED",
            "REGRESSED": "🔴 REGRESSED",
            "UNCHANGED": "⚪ UNCHANGED",
        }

        for metric_name, d in regression.overall_deltas.items():
            icon = status_icons.get(d.status, d.status)
            lines.append(
                f"| `{metric_name}` | {d.baseline_value} | {d.adapter_value} | "
                f"{d.absolute_delta:+.4f} | {d.percent_change:+.2f}% | {icon} |"
            )

        lines.extend([
            "",
            f"- **Total Improvements:** `{regression.total_improvements}`",
            f"- **Total Regressions:** `{regression.total_regressions}`",
            f"- **Unchanged Metrics:** `{regression.total_unchanged}` (within ±{regression.tolerance_pct}%)",
        ])
        return "\n".join(lines)

    @staticmethod
    def generate_domain_comparison_markdown(regression: RegressionReport) -> str:
        """Render per-domain side-by-side comparison table."""
        lines = [
            "## 2. Per-Domain Performance Comparison",
            "",
            "| Domain | Baseline Validity | Adapter Validity | Baseline Repetition | Adapter Repetition | Baseline Formatting | Adapter Formatting |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for dom, grp in sorted(regression.domain_deltas.items()):
            m = grp.metrics
            b_val = m.get("validity_rate").baseline_value if "validity_rate" in m else 0.0
            a_val = m.get("validity_rate").adapter_value if "validity_rate" in m else 0.0

            b_rep = m.get("avg_repetition_ratio").baseline_value if "avg_repetition_ratio" in m else 0.0
            a_rep = m.get("avg_repetition_ratio").adapter_value if "avg_repetition_ratio" in m else 0.0

            b_fmt = m.get("avg_formatting_score").baseline_value if "avg_formatting_score" in m else 0.0
            a_fmt = m.get("avg_formatting_score").adapter_value if "avg_formatting_score" in m else 0.0

            lines.append(
                f"| `{dom}` | {b_val:.2f} | {a_val:.2f} | {b_rep:.4f} | {a_rep:.4f} | {b_fmt:.2f} | {a_fmt:.2f} |"
            )

        return "\n".join(lines)

    @staticmethod
    def generate_difficulty_comparison_markdown(regression: RegressionReport) -> str:
        """Render per-difficulty side-by-side comparison table."""
        lines = [
            "## 3. Per-Difficulty Performance Breakdown",
            "",
            "| Difficulty | Baseline Validity | Adapter Validity | Baseline Repetition | Adapter Repetition | Baseline Token Length | Adapter Token Length |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for diff, grp in sorted(regression.difficulty_deltas.items()):
            m = grp.metrics
            b_val = m.get("validity_rate").baseline_value if "validity_rate" in m else 0.0
            a_val = m.get("validity_rate").adapter_value if "validity_rate" in m else 0.0

            b_rep = m.get("avg_repetition_ratio").baseline_value if "avg_repetition_ratio" in m else 0.0
            a_rep = m.get("avg_repetition_ratio").adapter_value if "avg_repetition_ratio" in m else 0.0

            b_tok = m.get("avg_token_length").baseline_value if "avg_token_length" in m else 0.0
            a_tok = m.get("avg_token_length").adapter_value if "avg_token_length" in m else 0.0

            lines.append(
                f"| `{diff}` | {b_val:.2f} | {a_val:.2f} | {b_rep:.4f} | {a_rep:.4f} | {b_tok:.1f} | {a_tok:.1f} |"
            )

        return "\n".join(lines)

    @classmethod
    def generate_full_comparison_report(cls, regression: RegressionReport) -> str:
        """Assemble complete markdown comparison document."""
        return "\n\n".join([
            "# Model Evaluation Benchmark & Regression Comparison Report",
            cls.generate_overall_comparison_markdown(regression),
            cls.generate_domain_comparison_markdown(regression),
            cls.generate_difficulty_comparison_markdown(regression),
        ])
