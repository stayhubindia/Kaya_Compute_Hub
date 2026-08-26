"""
Dataset Statistics & Reporting Engine.
Computes comprehensive dataset metrics, distributions, quality summaries, provenance metrics,
and generates both machine-readable (dataset_report.json) and human-readable (dataset_report.md) reports.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.dataset.cleaner import CleaningReport
from src.dataset.deduplicator import DeduplicationReport
from src.dataset.quality import QualityValidationReport
from src.dataset.schema import DatasetRecord
from src.dataset.splitter import SplitResult


class DatasetStatistics:
    """Computes distributions, summaries, provenance telemetry, and writes formatted dataset reports."""

    def __init__(
        self,
        domain_targets: Optional[Dict[str, float]] = None,
        difficulty_targets: Optional[Dict[str, float]] = None,
    ):
        self.domain_targets = domain_targets or {}
        self.difficulty_targets = difficulty_targets or {}

    def compute_metrics(
        self,
        raw_total: int,
        accepted_records: List[DatasetRecord],
        cleaning_report: Optional[CleaningReport] = None,
        dedup_report: Optional[DeduplicationReport] = None,
        quality_report: Optional[QualityValidationReport] = None,
        split_result: Optional[SplitResult] = None,
    ) -> Dict[str, Any]:
        total_accepted = len(accepted_records)

        # 1. Domain Distribution
        domain_counts = Counter(r.metadata.domain for r in accepted_records)
        domain_dist: Dict[str, Dict[str, Any]] = {}
        for dom, count in domain_counts.most_common():
            actual_pct = (count / total_accepted) if total_accepted else 0.0
            target_pct = self.domain_targets.get(dom, 0.0)
            domain_dist[dom] = {
                "count": count,
                "percentage": round(actual_pct * 100, 2),
                "target_percentage": round(target_pct * 100, 2),
                "delta": round((actual_pct - target_pct) * 100, 2),
            }

        # 2. Task Type Distribution
        task_counts = Counter(r.metadata.task_type for r in accepted_records)
        task_dist: Dict[str, Dict[str, Any]] = {}
        for ttype, count in task_counts.most_common():
            task_dist[ttype] = {
                "count": count,
                "percentage": round((count / total_accepted * 100) if total_accepted else 0.0, 2),
            }

        # 3. Difficulty Distribution
        diff_counts = Counter(r.metadata.difficulty for r in accepted_records)
        diff_dist: Dict[str, Dict[str, Any]] = {}
        for diff, count in diff_counts.items():
            actual_pct = (count / total_accepted) if total_accepted else 0.0
            target_pct = self.difficulty_targets.get(diff, 0.0)
            diff_dist[diff] = {
                "count": count,
                "percentage": round(actual_pct * 100, 2),
                "target_percentage": round(target_pct * 100, 2),
                "delta": round((actual_pct - target_pct) * 100, 2),
            }

        # 4. Message & Turn Statistics
        char_lengths = [r.total_chars() for r in accepted_records]
        word_lengths = [r.total_words() for r in accepted_records]
        turn_counts = [r.turn_count() for r in accepted_records]
        single_turn_count = sum(1 for r in accepted_records if r.is_single_turn())
        multi_turn_count = total_accepted - single_turn_count

        length_stats = {
            "characters": {
                "min": min(char_lengths) if char_lengths else 0,
                "max": max(char_lengths) if char_lengths else 0,
                "mean": round(statistics.mean(char_lengths), 2) if char_lengths else 0,
                "median": round(statistics.median(char_lengths), 2) if char_lengths else 0,
            },
            "words": {
                "min": min(word_lengths) if word_lengths else 0,
                "max": max(word_lengths) if word_lengths else 0,
                "mean": round(statistics.mean(word_lengths), 2) if word_lengths else 0,
                "median": round(statistics.median(word_lengths), 2) if word_lengths else 0,
            },
            "turns": {
                "single_turn_examples": single_turn_count,
                "multi_turn_examples": multi_turn_count,
                "single_turn_percentage": round((single_turn_count / total_accepted * 100) if total_accepted else 0, 2),
                "multi_turn_percentage": round((multi_turn_count / total_accepted * 100) if total_accepted else 0, 2),
                "mean_turns": round(statistics.mean(turn_counts), 2) if turn_counts else 0,
            },
        }

        # 5. Quality Statistics
        quality_scores = [r.metadata.quality_score for r in accepted_records if r.metadata.quality_score is not None]
        quality_stats = {
            "evaluated_count": len(quality_scores),
            "unscored_count": total_accepted - len(quality_scores),
            "mean_score": round(statistics.mean(quality_scores), 4) if quality_scores else None,
            "min_score": round(min(quality_scores), 4) if quality_scores else None,
            "max_score": round(max(quality_scores), 4) if quality_scores else None,
            "preferred_score_count": sum(1 for s in quality_scores if s >= 0.90),
        }

        # 6. Source & Provenance Statistics (Phase 2.3.1)
        source_type_counts = Counter(
            (r.metadata.provenance.source_type if r.metadata.provenance else r.metadata.source_type)
            for r in accepted_records
        )
        source_counts = Counter(
            (r.metadata.provenance.source if r.metadata.provenance else r.metadata.source)
            for r in accepted_records
        )
        generator_counts = Counter(
            (r.metadata.provenance.generator if r.metadata.provenance else r.metadata.generator)
            for r in accepted_records
            if (r.metadata.provenance and r.metadata.provenance.generator) or r.metadata.generator
        )
        known_license_count = sum(
            1 for r in accepted_records
            if (r.metadata.provenance and r.metadata.provenance.license) or r.metadata.license
        )
        source_id_counts = Counter(
            (r.metadata.provenance.source_id if r.metadata.provenance else r.metadata.source_id)
            for r in accepted_records
            if (r.metadata.provenance and r.metadata.provenance.source_id) or r.metadata.source_id
        )

        source_stats = {
            "source_type_distribution": {
                stype: {
                    "count": cnt,
                    "percentage": round((cnt / total_accepted * 100) if total_accepted else 0.0, 2),
                }
                for stype, cnt in source_type_counts.most_common()
            },
            "source_distribution": {
                src: {
                    "count": cnt,
                    "percentage": round((cnt / total_accepted * 100) if total_accepted else 0.0, 2),
                }
                for src, cnt in source_counts.most_common()
            },
            "generator_distribution": {
                gen: {
                    "count": cnt,
                    "percentage": round((cnt / total_accepted * 100) if total_accepted else 0.0, 2),
                }
                for gen, cnt in generator_counts.most_common()
            },
            "license_availability": {
                "known_license_count": known_license_count,
                "unspecified_count": total_accepted - known_license_count,
                "known_percentage": round((known_license_count / total_accepted * 100) if total_accepted else 0.0, 2),
            },
            "records_per_source_id": dict(source_id_counts.most_common()),
        }

        # 7. Split Summary
        split_stats = split_result.to_dict() if split_result else {}

        # 8. Aggregate Metrics
        return {
            "summary": {
                "total_raw_inputs": raw_total,
                "accepted_examples": total_accepted,
                "rejected_examples": cleaning_report.rejected_count if cleaning_report else 0,
                "exact_duplicates_removed": dedup_report.exact_duplicates if dedup_report else 0,
                "near_duplicates_removed": dedup_report.near_duplicates if dedup_report else 0,
                "acceptance_rate_percent": round((total_accepted / raw_total * 100) if raw_total else 0, 2),
            },
            "domain_distribution": domain_dist,
            "task_type_distribution": task_dist,
            "difficulty_distribution": diff_dist,
            "length_statistics": length_stats,
            "quality_statistics": quality_stats,
            "source_statistics": source_stats,
            "split_statistics": split_stats,
            "cleaning_details": cleaning_report.to_dict() if cleaning_report else {},
            "deduplication_details": dedup_report.to_dict() if dedup_report else {},
        }

    def generate_markdown_report(self, metrics: Dict[str, Any]) -> str:
        s = metrics["summary"]
        doms = metrics["domain_distribution"]
        tasks = metrics["task_type_distribution"]
        diffs = metrics["difficulty_distribution"]
        lens = metrics["length_statistics"]
        qual = metrics["quality_statistics"]
        srcs = metrics.get("source_statistics", {})
        split = metrics.get("split_statistics", {})

        md = []
        md.append("# Dataset Engineering & Quality Report\n")
        md.append("## Executive Summary\n")
        md.append("| Metric | Count | Percentage |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| **Total Raw Ingested** | `{s['total_raw_inputs']}` | 100.0% |")
        md.append(f"| **Cleaned & Accepted** | `{s['accepted_examples']}` | {s['acceptance_rate_percent']}% |")
        md.append(f"| **Rejected (Cleaning)** | `{s['rejected_examples']}` | - |")
        md.append(f"| **Exact Duplicates Removed** | `{s['exact_duplicates_removed']}` | - |")
        md.append(f"| **Near Duplicates Removed** | `{s['near_duplicates_removed']}` | - |\n")

        md.append("## Dataset Sources & Provenance\n")
        if srcs:
            md.append("### Source Type Distribution")
            md.append("| Source Type | Records | Percentage |")
            md.append("| :--- | :--- | :--- |")
            for stype, data in srcs.get("source_type_distribution", {}).items():
                md.append(f"| `{stype}` | {data['count']} | {data['percentage']}% |")
            md.append("")

            md.append("### Source Distribution")
            md.append("| Source | Records | Percentage |")
            md.append("| :--- | :--- | :--- |")
            for src_name, data in srcs.get("source_distribution", {}).items():
                md.append(f"| `{src_name}` | {data['count']} | {data['percentage']}% |")
            md.append("")

            gen_dist = srcs.get("generator_distribution", {})
            if gen_dist:
                md.append("### Synthetic Generator Distribution")
                md.append("| Generator | Records | Percentage |")
                md.append("| :--- | :--- | :--- |")
                for gen_name, data in gen_dist.items():
                    md.append(f"| `{gen_name}` | {data['count']} | {data['percentage']}% |")
                md.append("")

            lic = srcs.get("license_availability", {})
            md.append(f"- **Known Licenses**: {lic.get('known_license_count', 0)} ({lic.get('known_percentage', 0)}%)")
            md.append(f"- **Unspecified / Explicitly None**: {lic.get('unspecified_count', 0)}\n")

        md.append("## Domain Distribution\n")
        md.append("| Domain | Count | Actual % | Target % | Delta |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        for dom, data in sorted(doms.items(), key=lambda x: x[1]["count"], reverse=True):
            delta_str = f"+{data['delta']}%" if data['delta'] > 0 else f"{data['delta']}%"
            md.append(f"| `{dom}` | {data['count']} | {data['percentage']}% | {data['target_percentage']}% | {delta_str} |")
        md.append("")

        md.append("## Task Type Distribution\n")
        md.append("| Task Type | Count | Percentage |")
        md.append("| :--- | :--- | :--- |")
        for task, data in sorted(tasks.items(), key=lambda x: x[1]["count"], reverse=True):
            md.append(f"| `{task}` | {data['count']} | {data['percentage']}% |")
        md.append("")

        md.append("## Difficulty Distribution\n")
        md.append("| Level | Count | Actual % | Target % | Delta |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        for diff, data in diffs.items():
            delta_str = f"+{data['delta']}%" if data['delta'] > 0 else f"{data['delta']}%"
            md.append(f"| `{diff}` | {data['count']} | {data['percentage']}% | {data['target_percentage']}% | {delta_str} |")
        md.append("")

        md.append("## Conversation & Length Metrics\n")
        md.append(f"- **Single-Turn Examples**: {lens['turns']['single_turn_examples']} ({lens['turns']['single_turn_percentage']}%)")
        md.append(f"- **Multi-Turn Examples**: {lens['turns']['multi_turn_examples']} ({lens['turns']['multi_turn_percentage']}%)")
        md.append(f"- **Mean Turn Count**: {lens['turns']['mean_turns']}")
        md.append(f"- **Characters / Example**: Min: {lens['characters']['min']} | Mean: {lens['characters']['mean']} | Median: {lens['characters']['median']} | Max: {lens['characters']['max']}")
        md.append(f"- **Words / Example**: Min: {lens['words']['min']} | Mean: {lens['words']['mean']} | Median: {lens['words']['median']} | Max: {lens['words']['max']}\n")

        md.append("## Quality Metrics\n")
        md.append(f"- **Evaluated Records**: {qual['evaluated_count']}")
        md.append(f"- **Unscored Records**: {qual['unscored_count']}")
        md.append(f"- **Mean Quality Score**: {qual['mean_score'] if qual['mean_score'] is not None else 'N/A'}")
        md.append(f"- **Preferred Score Count (≥ 0.90)**: {qual['preferred_score_count']}\n")

        if split:
            md.append("## Dataset Split Summary\n")
            md.append("| Split Partition | Examples | Percentage |")
            md.append("| :--- | :--- | :--- |")
            md.append(f"| **Train** | `{split.get('train_count', 0)}` | {split.get('split_summary', {}).get('train_percent', 0)}% |")
            md.append(f"| **Validation** | `{split.get('validation_count', 0)}` | {split.get('split_summary', {}).get('validation_percent', 0)}% |")
            md.append(f"| **Test (Isolated)** | `{split.get('test_count', 0)}` | {split.get('split_summary', {}).get('test_percent', 0)}% |")
            md.append(f"\n*Leakage Detected*: **{split.get('split_summary', {}).get('leakage_detected', False)}**\n")

        return "\n".join(md)
