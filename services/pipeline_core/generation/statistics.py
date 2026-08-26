"""
Instruction Generation Statistics and Report Builder (Phase 3.4).
Aggregates telemetry across knowledge units, generated candidates, acceptance rates,
domain/task distributions, quality scores, and produces comprehensive Markdown & JSON reports.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.dataset.deduplicator import DeduplicationReport
from src.generation.models import CandidateRecord, KnowledgeUnit


class GenerationStatisticsAggregator:
    """Aggregates telemetry and writes comprehensive reports and manifests."""

    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self.total_units_evaluated: int = 0
        self.total_units_selected: int = 0
        self.total_candidates_generated: int = 0
        self.total_candidates_accepted: int = 0
        self.total_candidates_rejected: int = 0

        self.domain_counts: Counter = Counter()
        self.topic_counts: Counter = Counter()
        self.task_type_counts: Counter = Counter()
        self.difficulty_counts: Counter = Counter()
        self.content_type_counts: Counter = Counter()
        self.source_counts: Counter = Counter()
        self.license_counts: Counter = Counter()

        self.quality_scores: List[float] = []
        self.grounding_scores: List[float] = []
        self.rejection_reasons: Counter = Counter()
        self.rejected_records_sample: List[Dict[str, Any]] = []

    def record_unit_analyzed(self, unit: KnowledgeUnit, selected: bool) -> None:
        self.total_units_evaluated += 1
        if selected:
            self.total_units_selected += 1

    def record_candidate(self, candidate: CandidateRecord) -> None:
        self.total_candidates_generated += 1
        if candidate.is_accepted:
            self.total_candidates_accepted += 1
            rec = candidate.record
            self.domain_counts[rec.metadata.domain] += 1
            self.topic_counts[rec.metadata.topic] += 1
            self.task_type_counts[rec.metadata.task_type] += 1
            self.difficulty_counts[rec.metadata.difficulty] += 1
            self.source_counts[rec.metadata.source] += 1
            if rec.metadata.license:
                self.license_counts[rec.metadata.license] += 1
            else:
                self.license_counts["UNKNOWN"] += 1

            for ct in candidate.knowledge_unit.content_types:
                self.content_type_counts[ct.value] += 1

            self.quality_scores.append(candidate.quality_score)
            self.grounding_scores.append(candidate.grounding.grounding_score)
        else:
            self.total_candidates_rejected += 1
            for reason in candidate.rejection_reasons:
                self.rejection_reasons[reason] += 1
            if len(self.rejected_records_sample) < 100:
                self.rejected_records_sample.append(candidate.to_dict())

    def get_summary_dict(self, dedup_report: Optional[DeduplicationReport] = None) -> Dict[str, Any]:
        avg_q = sum(self.quality_scores) / max(1, len(self.quality_scores))
        avg_g = sum(self.grounding_scores) / max(1, len(self.grounding_scores))
        preferred_q = sum(1 for q in self.quality_scores if q >= 0.90)

        return {
            "version": self.version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_units_evaluated": self.total_units_evaluated,
            "total_units_selected": self.total_units_selected,
            "total_candidates_generated": self.total_candidates_generated,
            "total_candidates_accepted": self.total_candidates_accepted,
            "total_candidates_rejected": self.total_candidates_rejected,
            "acceptance_rate": round(self.total_candidates_accepted / max(1, self.total_candidates_generated), 4),
            "average_quality_score": round(avg_q, 4),
            "average_grounding_score": round(avg_g, 4),
            "preferred_quality_count": preferred_q,
            "domain_distribution": dict(self.domain_counts),
            "topic_distribution": dict(self.topic_counts),
            "task_type_distribution": dict(self.task_type_counts),
            "difficulty_distribution": dict(self.difficulty_counts),
            "content_type_distribution": dict(self.content_type_counts),
            "source_distribution": dict(self.source_counts),
            "license_distribution": dict(self.license_counts),
            "rejection_reasons": dict(self.rejection_reasons),
            "deduplication": dedup_report.to_dict() if dedup_report else None,
        }

    def write_reports(
        self,
        output_dir: Path,
        dedup_report: Optional[DeduplicationReport] = None,
        manifest_files: Optional[Dict[str, str]] = None,
    ) -> None:
        """Writes all reports, statistics, and manifests to the output reports/ directory."""
        reports_dir = output_dir / "reports"
        manifests_dir = output_dir / "manifests"
        reports_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        summary = self.get_summary_dict(dedup_report)

        # 1. statistics.json
        with open(reports_dir / "statistics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 2. generation_report.json & generation_report.md
        with open(reports_dir / "generation_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        md_content = self._generate_markdown_report(summary)
        with open(reports_dir / "generation_report.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        # 3. quality_report.json & quality_report.md
        quality_summary = {
            "average_quality_score": summary["average_quality_score"],
            "average_grounding_score": summary["average_grounding_score"],
            "preferred_quality_count": summary["preferred_quality_count"],
            "total_evaluated": summary["total_candidates_generated"],
            "passed_count": summary["total_candidates_accepted"],
            "failed_count": summary["total_candidates_rejected"],
            "rejection_breakdown": summary["rejection_reasons"],
        }
        with open(reports_dir / "quality_report.json", "w", encoding="utf-8") as f:
            json.dump(quality_summary, f, indent=2)

        quality_md = self._generate_quality_markdown(quality_summary)
        with open(reports_dir / "quality_report.md", "w", encoding="utf-8") as f:
            f.write(quality_md)

        # 4. provenance_report.json
        provenance_summary = {
            "source_distribution": summary["source_distribution"],
            "license_distribution": summary["license_distribution"],
            "total_accepted_records": summary["total_candidates_accepted"],
            "rights_verification_required_count": summary["license_distribution"].get("UNKNOWN", 0),
        }
        with open(reports_dir / "provenance_report.json", "w", encoding="utf-8") as f:
            json.dump(provenance_summary, f, indent=2)

        # 5. rejection_report.json
        with open(reports_dir / "rejection_report.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_rejected": self.total_candidates_rejected,
                    "rejection_reasons": summary["rejection_reasons"],
                    "samples": self.rejected_records_sample,
                },
                f,
                indent=2,
            )

        # 6. manifests/generation_manifest.json
        manifest_data = {
            "generator_version": self.version,
            "generated_at": summary["generated_at"],
            "total_records": summary["total_candidates_accepted"],
            "files": manifest_files or {},
            "summary": summary,
        }
        with open(manifests_dir / "generation_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        lines = [
            "# Scientific Instruction Dataset Generation Report",
            f"\n**Generated At**: {summary['generated_at']}  ",
            f"**Engine Version**: {summary['version']}  ",
            f"**Total Units Evaluated**: {summary['total_units_evaluated']}  ",
            f"**Total Candidates Generated**: {summary['total_candidates_generated']}  ",
            f"**Accepted Records**: {summary['total_candidates_accepted']} ({summary['acceptance_rate']*100:.1f}%)  ",
            f"**Rejected Records**: {summary['total_candidates_rejected']}  ",
            f"**Average Quality Score**: {summary['average_quality_score']} (Preferred >=0.90: {summary['preferred_quality_count']})  ",
            f"**Average Grounding Score**: {summary['average_grounding_score']}  \n",
            "## 1. Task Type Distribution",
            "| Task Type | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ]
        total_accepted = max(1, summary["total_candidates_accepted"])
        for task, count in summary["task_type_distribution"].items():
            lines.append(f"| `{task}` | {count} | {count/total_accepted*100:.1f}% |")

        lines.extend([
            "\n## 2. Difficulty Distribution",
            "| Difficulty | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ])
        for diff, count in summary["difficulty_distribution"].items():
            lines.append(f"| `{diff}` | {count} | {count/total_accepted*100:.1f}% |")

        lines.extend([
            "\n## 3. Scientific Content Types",
            "| Content Type | Occurrences |",
            "| :--- | :--- |",
        ])
        for ct, count in summary["content_type_distribution"].items():
            lines.append(f"| `{ct}` | {count} |")

        lines.extend([
            "\n## 4. Source & Rights Distribution",
            "| Source / License | Count |",
            "| :--- | :--- |",
        ])
        for src, count in summary["source_distribution"].items():
            lines.append(f"| Source: `{src}` | {count} |")
        for lic, count in summary["license_distribution"].items():
            lines.append(f"| License: `{lic}` | {count} |")

        return "\n".join(lines) + "\n"

    def _generate_quality_markdown(self, q: Dict[str, Any]) -> str:
        lines = [
            "# Instruction Quality & Grounding Audit Report",
            f"\n- **Total Evaluated**: {q['total_evaluated']}",
            f"- **Passed Quality Gate (>= 0.85)**: {q['passed_count']}",
            f"- **Rejected Count**: {q['failed_count']}",
            f"- **Average Quality Score**: {q['average_quality_score']}",
            f"- **Average Grounding Score**: {q['average_grounding_score']}",
            f"- **Preferred Tier (>= 0.90)**: {q['preferred_quality_count']}\n",
            "## Rejection Breakdown",
            "| Reason | Occurrences |",
            "| :--- | :--- |",
        ]
        for reason, count in q["rejection_breakdown"].items():
            lines.append(f"| {reason} | {count} |")
        return "\n".join(lines) + "\n"
