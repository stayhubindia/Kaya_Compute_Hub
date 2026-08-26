"""
Pilot Dataset Assembly & Validation Engine (Phase 2.3.5).
Orchestrates candidate generation across all 13 domains and 4 difficulty tiers,
multi-source ingestion, cleaning, deduplication, quality evaluation, stratified mixing,
train/val/test splitting, cross-split leakage verification, and readiness evaluation.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from src.dataset.cleaner import DatasetCleaner, RejectedRecord
from src.dataset.deduplicator import DatasetDeduplicator
from src.dataset.generator import GenerationRequest, GenerationResult, SampleSyntheticGenerator
from src.dataset.loader import DatasetLoader
from src.dataset.metadata import MetadataEnricher
from src.dataset.mixer import DatasetMixer, MixingRequest, MixingResult
from src.dataset.normalizer import DatasetNormalizer
from src.dataset.quality import QualityValidator
from src.dataset.schema import (
    DatasetRecord,
    DifficultyLevel,
    ProvenanceInfo,
    RecordMetadata,
    Role,
    SourceType,
    TaskType,
)
from src.dataset.source_registry import SourceRegistry
from src.dataset.splitter import DatasetSplitter, SplitResult
from src.dataset.statistics import DatasetStatistics
from src.dataset.template_registry import TaskTemplate, TemplateRegistry


# ============================================================================
# 1. DATA MODELS & MANIFESTS
# ============================================================================

class PilotManifest(BaseModel):
    """Manifest capturing full metadata and state for a pilot dataset build."""
    pilot_version: str = "pilot-v1"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    seed: int = 42
    target_count: int = 1000
    actual_count: int = 0
    candidate_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    exact_duplicate_count: int = 0
    near_duplicate_count: int = 0
    train_count: int = 0
    validation_count: int = 0
    test_count: int = 0
    input_sources: List[str] = Field(default_factory=list)
    source_versions: Dict[str, str] = Field(default_factory=dict)
    generator_versions: Dict[str, str] = Field(default_factory=dict)
    template_manifest_version: str = "1.0.0"
    dataset_config_version: str = "1.0.0"
    pipeline_version: str = "1.0.0"
    mixing_strategy: str = "proportional"

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def save_json(self, path: Union[str, Path]) -> Path:
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return out_path


class ReadinessDimensionScore(BaseModel):
    """Evaluation score and status for a single measurable readiness dimension."""
    dimension: str
    status: str = "PASS"  # PASS | WARN | FAIL
    score: Optional[float] = None
    threshold: Optional[str] = None
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class PilotReadinessReport(BaseModel):
    """Comprehensive readiness assessment report across all quality dimensions."""
    overall_status: str = "PASS"  # PASS | WARN | FAIL
    target_count: int = 1000
    candidate_count: int = 0
    final_count: int = 0
    dimensions: List[ReadinessDimensionScore] = Field(default_factory=list)
    domain_distribution: Dict[str, Any] = Field(default_factory=dict)
    difficulty_distribution: Dict[str, Any] = Field(default_factory=dict)
    task_distribution: Dict[str, Any] = Field(default_factory=dict)
    source_distribution: Dict[str, Any] = Field(default_factory=dict)
    quality_summary: Dict[str, Any] = Field(default_factory=dict)
    deduplication_summary: Dict[str, Any] = Field(default_factory=dict)
    split_summary: Dict[str, Any] = Field(default_factory=dict)
    leakage_detected: bool = False
    provenance_summary: Dict[str, Any] = Field(default_factory=dict)
    shortages: List[Dict[str, Any]] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def generate_markdown(self) -> str:
        lines = [
            "# Pilot Dataset Assembly & Readiness Audit Report",
            "",
            "## 1. Executive Summary & Readiness Status",
            "",
            f"**Overall Readiness Assessment**: `{self.overall_status}`",
            "",
            "| Dimension | Status | Measured Score | Threshold Criteria | Details |",
            "| :--- | :---: | :--- | :--- | :--- |",
        ]

        for dim in self.dimensions:
            score_str = f"{dim.score:.4f}" if dim.score is not None else "N/A"
            thresh_str = dim.threshold or "N/A"
            status_icon = "✅ PASS" if dim.status == "PASS" else ("⚠️ WARN" if dim.status == "WARN" else "❌ FAIL")
            lines.append(f"| **{dim.dimension}** | {status_icon} | `{score_str}` | `{thresh_str}` | {dim.details} |")

        lines.extend([
            "",
            "---",
            "",
            "## 2. Dataset Size & Flow",
            "",
            "| Stage | Count | Notes |",
            "| :--- | :--- | :--- |",
            f"| **Target Request** | `{self.target_count}` | Configured pilot goal |",
            f"| **Candidate Pool** | `{self.candidate_count}` | Ingested & synthesized candidate pool |",
            f"| **Final Unified Dataset** | `{self.final_count}` | Selected balanced dataset |",
            f"| **Train Split (90%)** | `{self.split_summary.get('train_count', 0)}` | Training examples |",
            f"| **Validation Split (5%)** | `{self.split_summary.get('validation_count', 0)}` | Validation examples |",
            f"| **Test Split (5%)** | `{self.split_summary.get('test_count', 0)}` | Test evaluation examples |",
            "",
            "---",
            "",
            "## 3. Domain Distribution & Target Compliance",
            "",
            "| Domain | Target % | Actual Count | Actual % | Delta |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for dom, stats in sorted(self.domain_distribution.items()):
            t_pct = stats.get("target_percentage", 0.0)
            a_cnt = stats.get("count", 0)
            a_pct = stats.get("percentage", 0.0)
            delta = stats.get("delta", 0.0)
            lines.append(f"| `{dom}` | {t_pct:.1f}% | `{a_cnt}` | {a_pct:.1f}% | {delta:+.2f}% |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Difficulty Distribution",
            "",
            "| Difficulty Tier | Target % | Actual Count | Actual % | Delta |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for diff, stats in sorted(self.difficulty_distribution.items()):
            t_pct = stats.get("target_percentage", 0.0)
            a_cnt = stats.get("count", 0)
            a_pct = stats.get("percentage", 0.0)
            delta = stats.get("delta", 0.0)
            lines.append(f"| `{diff}` | {t_pct:.1f}% | `{a_cnt}` | {a_pct:.1f}% | {delta:+.2f}% |")

        lines.extend([
            "",
            "---",
            "",
            "## 5. Quality Metrics",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Mean Quality Score** | `{self.quality_summary.get('mean_score', 'N/A')}` |",
            f"| **Median Quality Score** | `{self.quality_summary.get('median_score', 'N/A')}` |",
            f"| **Min Quality Score** | `{self.quality_summary.get('min_score', 'N/A')}` |",
            f"| **Max Quality Score** | `{self.quality_summary.get('max_score', 'N/A')}` |",
            f"| **Percentage >= 0.85** | `{self.quality_summary.get('pct_ge_085', 0.0):.1f}%` |",
            f"| **Percentage >= 0.90** | `{self.quality_summary.get('pct_ge_090', 0.0):.1f}%` |",
            f"| **Rejected Percentage** | `{self.quality_summary.get('rejected_pct', 0.0):.1f}%` |",
            f"| **Unscored Percentage** | `{self.quality_summary.get('unscored_pct', 0.0):.1f}%` |",
            "",
            "---",
            "",
            "## 6. Deduplication & Cross-Split Leakage",
            "",
            "| Metric | Value | Status |",
            "| :--- | :--- | :--- |",
            f"| **Exact Duplicates** | `{self.deduplication_summary.get('exact_duplicates', 0)}` | - |",
            f"| **Near Duplicates** | `{self.deduplication_summary.get('near_duplicates', 0)}` | - |",
            f"| **Duplicate Rate** | `{self.deduplication_summary.get('duplicate_rate', 0.0):.2%}` | {'✅ PASS' if self.deduplication_summary.get('duplicate_rate', 0.0) <= 0.05 else '⚠️ WARN'} |",
            f"| **Cross-Split Leakage** | `{'DETECTED' if self.leakage_detected else 'NONE DETECTED'}` | {'❌ FAIL' if self.leakage_detected else '✅ PASS'} |",
            "",
            "---",
            "",
            "## 7. Provenance & Source Diversity",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Records with Provenance** | `{self.provenance_summary.get('with_provenance', 0)}` ({self.provenance_summary.get('provenance_rate', 0.0):.1f}%) |",
            f"| **Records without Provenance** | `{self.provenance_summary.get('without_provenance', 0)}` |",
            f"| **Active Sources** | `{', '.join(self.source_distribution.keys())}` |",
            "",
        ])

        if self.shortages:
            lines.extend([
                "---",
                "",
                "## 8. Data Shortage Telemetry",
                "",
                "| Stratum Category | Dimension | Requested | Available | Shortage Deficit |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for s in self.shortages:
                lines.append(
                    f"| `{s.get('category')}` | `{s.get('dimension')}` | `{s.get('requested')}` | `{s.get('available')}` | **`{s.get('shortage')}`** |"
                )
            lines.append("")

        return "\n".join(lines)

    def save_reports(self, output_dir: Union[str, Path]) -> Tuple[Path, Path]:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "pilot_readiness_report.json"
        md_path = out_dir / "pilot_readiness_report.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())

        return json_path, md_path


class PilotResult(BaseModel):
    """Container for complete output of a pilot assembly execution."""
    manifest: PilotManifest
    readiness_report: PilotReadinessReport
    train_records: List[DatasetRecord] = Field(default_factory=list)
    validation_records: List[DatasetRecord] = Field(default_factory=list)
    test_records: List[DatasetRecord] = Field(default_factory=list)
    candidate_records: List[DatasetRecord] = Field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.readiness_report.overall_status in ["PASS", "WARN"] and not self.readiness_report.leakage_detected


# ============================================================================
# 2. PILOT ASSEMBLER & VALIDATION ENGINE
# ============================================================================

class PilotAssembler:
    """
    Main Pilot Dataset Assembly and Validation Engine.
    Coordinates candidate generation across 13 domains, multi-source ingestion,
    cleaning, deduplication, quality scoring, stratified mixing, 90/5/5 splitting,
    cross-split leakage verification, and readiness assessment.
    """

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/dataset.yaml",
        templates_path: Union[str, Path] = "configs/domain_templates.yaml",
        sources_path: Union[str, Path] = "configs/sources.yaml",
    ):
        self.config_path = Path(config_path).resolve()
        self.templates_path = Path(templates_path).resolve()
        self.sources_path = Path(sources_path).resolve()

        self.config = self._load_yaml(self.config_path)
        self.pilot_cfg: Dict[str, Any] = self.config.get("pilot", {})
        self.domain_targets: Dict[str, float] = self.config.get("domain_targets", {})
        self.difficulty_targets: Dict[str, float] = self.config.get("difficulty", {}).get("targets", {
            "beginner": 0.25,
            "intermediate": 0.40,
            "advanced": 0.25,
            "expert": 0.10,
        })

        # Load registries & backends
        self.template_registry = TemplateRegistry.from_yaml(self.templates_path)
        self.generator = SampleSyntheticGenerator()
        self.mixer = DatasetMixer(config_path=self.config_path)
        self.cleaner = DatasetCleaner()
        self.deduplicator = DatasetDeduplicator(enable_near_dedup=True)
        self.quality_validator = QualityValidator(minimum_score=0.85, preferred_score=0.90)
        self.enricher = MetadataEnricher(pipeline_version=self.config.get("pipeline", {}).get("version", "1.0.0"))
        self.splitter = DatasetSplitter(
            train_ratio=0.90,
            validation_ratio=0.05,
            test_ratio=0.05,
            random_seed=self.pilot_cfg.get("seed", 42),
            stratify_by_domain=True,
        )

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"YAML configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def generate_candidate_pool(
        self,
        target_candidate_count: int,
        seed: int = 42,
        include_fixtures: bool = True,
    ) -> List[DatasetRecord]:
        """
        Synthesizes a rich, multi-domain, multi-difficulty candidate pool using
        TemplateRegistry templates and available multi-source fixtures.
        """
        candidates: List[DatasetRecord] = []
        rng = random.Random(seed)

        # 1. Ingest existing multi-source fixtures if requested
        if include_fixtures:
            fixture_paths = [
                Path("datasets/fixtures/synthetic.jsonl"),
                Path("datasets/fixtures/human.jsonl"),
                Path("datasets/fixtures/documentation.jsonl"),
                Path("datasets/fixtures/existing_dataset.jsonl"),
            ]
            for fix_p in fixture_paths:
                if fix_p.is_file():
                    try:
                        fix_records = self.mixer.ingest_sources([fix_p])
                        candidates.extend(fix_records)
                    except Exception as e:
                        pass

        # 2. Determine synthetic target count to reach target_candidate_count
        needed_synthetic = max(0, target_candidate_count - len(candidates))
        if needed_synthetic <= 0:
            return candidates

        # 3. Apportion synthetic quota across domains according to domain_targets
        domain_quotas = self.mixer.strategies["proportional"]._allocate_quotas(
            needed_synthetic, self.domain_targets
        )

        all_templates = self.template_registry.list_templates()
        templates_by_domain: Dict[str, List[TaskTemplate]] = defaultdict(list)
        for tmpl in all_templates:
            templates_by_domain[tmpl.domain].append(tmpl)

        # 4. Generate candidates per domain across templates & difficulty tiers
        difficulties = ["beginner", "intermediate", "advanced", "expert"]

        for dom, quota in sorted(domain_quotas.items()):
            if quota <= 0:
                continue

            dom_templates = templates_by_domain.get(dom, [])
            if not dom_templates:
                # If no specific template exists, use generic domain synthesis
                dom_seed = seed + hash(dom) % 100000
                req = GenerationRequest(
                    domain=dom,
                    topic="general",
                    task_type="explanation",
                    difficulty="intermediate",
                    number_of_examples=quota,
                    seed=dom_seed,
                )
                res = self.generator.generate_batch(req)
                candidates.extend(res.records)
                continue

            # Sub-allocate domain quota to difficulty levels
            diff_quotas = self.mixer.strategies["proportional"]._allocate_quotas(
                quota, self.difficulty_targets
            )

            for diff, diff_quota in sorted(diff_quotas.items()):
                if diff_quota <= 0:
                    continue

                # Filter templates supporting this difficulty
                matching_templates = [t for t in dom_templates if diff in t.supported_difficulties]
                if not matching_templates:
                    matching_templates = dom_templates

                for k in range(diff_quota):
                    tmpl = matching_templates[k % len(matching_templates)]
                    item_seed = seed + (len(candidates) * 31) + k
                    try:
                        records = self.generator.generate_from_template(
                            template=tmpl,
                            number_of_examples=1,
                            difficulty=diff,
                            seed=item_seed,
                            batch_id=f"pilot_gen_{dom}_{diff}",
                        )
                        candidates.extend(records)
                    except Exception:
                        # Fallback direct generation
                        req = GenerationRequest(
                            domain=dom,
                            topic=tmpl.topic,
                            task_type=tmpl.task_type,
                            difficulty=diff,
                            number_of_examples=1,
                            seed=item_seed,
                        )
                        res = self.generator.generate_batch(req)
                        candidates.extend(res.records)

        # Shuffle candidates deterministically
        rng.shuffle(candidates)
        return candidates

    def assemble(
        self,
        target_count: Optional[int] = None,
        seed: Optional[int] = None,
        version: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        candidate_multiplier: Optional[float] = None,
        save_outputs: bool = True,
    ) -> PilotResult:
        """
        Executes full end-to-end Pilot Dataset Assembly and Validation.
        """
        effective_target = target_count or self.pilot_cfg.get("target_count", 1000)
        effective_seed = seed if seed is not None else self.pilot_cfg.get("seed", 42)
        effective_version = version or self.pilot_cfg.get("version", "pilot-v1")
        effective_mult = candidate_multiplier or self.pilot_cfg.get("candidate_multiplier", 1.2)
        target_candidate_count = int(math.ceil(effective_target * effective_mult))

        base_out_dir = Path(output_dir or self.pilot_cfg.get("output_dir", "datasets/pilot/v1")).resolve()
        raw_dir = base_out_dir / "raw"
        processed_dir = base_out_dir / "processed"
        reports_dir = base_out_dir / "reports"
        manifests_dir = base_out_dir / "manifests"

        # 1. Candidate Pool Generation
        candidates = self.generate_candidate_pool(
            target_candidate_count=target_candidate_count,
            seed=effective_seed,
            include_fixtures=True,
        )
        total_candidates_generated = len(candidates)

        # 2. Cleaning & Schema Validation
        # Convert records to dicts for cleaning
        candidate_dicts = [r.to_dict() for r in candidates]
        cleaned_records, cleaning_report = self.cleaner.clean_records(candidate_dicts)
        accepted_cleaned_count = len(cleaned_records)
        rejected_cleaned_count = len(cleaning_report.rejected_records)

        # 3. Deduplication
        deduped_records, dedup_report = self.deduplicator.deduplicate(cleaned_records)
        exact_dups = dedup_report.exact_duplicates
        near_dups = dedup_report.near_duplicates
        dup_rate = dedup_report.duplicate_rate

        # 4. Quality Evaluation
        quality_records, quality_report = self.quality_validator.validate_records(deduped_records)

        # 5. Stratified Mixing & Balancing
        mix_request = MixingRequest(
            target_count=effective_target,
            strategy=self.config.get("mixing", {}).get("strategy", "proportional"),
            seed=effective_seed,
            allow_oversampling=False,
            allow_undersampling=True,
            batch_id=f"pilot_{effective_version}_s{effective_seed}",
        )
        mix_result = self.mixer.mix(mix_request, candidate_records=quality_records)
        selected_records = mix_result.records


        # 6. Train / Validation / Test Splitting (90% / 5% / 5%)
        self.splitter.random_seed = effective_seed
        split_result = self.splitter.split(selected_records)

        # 7. Cross-Split Leakage Check
        train_hashes = {r.canonical_content_hash() for r in split_result.train}
        val_hashes = {r.canonical_content_hash() for r in split_result.validation}
        test_hashes = {r.canonical_content_hash() for r in split_result.test}

        leakage_train_val = len(train_hashes.intersection(val_hashes))
        leakage_train_test = len(train_hashes.intersection(test_hashes))
        leakage_val_test = len(val_hashes.intersection(test_hashes))
        total_leakage = leakage_train_val + leakage_train_test + leakage_val_test
        leakage_detected = total_leakage > 0

        # 8. Compute Statistics & Distributions
        stats_engine = DatasetStatistics(
            domain_targets=self.domain_targets,
            difficulty_targets=self.difficulty_targets,
        )
        dataset_stats = stats_engine.compute_metrics(
            raw_total=total_candidates_generated,
            accepted_records=selected_records,
            cleaning_report=cleaning_report,
            dedup_report=dedup_report,
            quality_report=quality_report,
            split_result=split_result,
        )

        # 9. Compute Quality Summary
        quality_scores = [
            r.metadata.quality_score for r in selected_records if r.metadata.quality_score is not None
        ]
        unscored_records = [r for r in selected_records if r.metadata.quality_score is None]
        unscored_pct = (len(unscored_records) / len(selected_records) * 100) if selected_records else 0.0

        if quality_scores:
            mean_qual = round(statistics.mean(quality_scores), 4)
            median_qual = round(statistics.median(quality_scores), 4)
            min_qual = round(min(quality_scores), 4)
            max_qual = round(max(quality_scores), 4)
            pct_ge_085 = round((sum(1 for s in quality_scores if s >= 0.85) / len(quality_scores)) * 100, 2)
            pct_ge_090 = round((sum(1 for s in quality_scores if s >= 0.90) / len(quality_scores)) * 100, 2)
        else:
            mean_qual = None
            median_qual = None
            min_qual = None
            max_qual = None
            pct_ge_085 = 0.0
            pct_ge_090 = 0.0

        quality_summary = {
            "mean_score": mean_qual,
            "median_score": median_qual,
            "min_score": min_qual,
            "max_score": max_qual,
            "pct_ge_085": pct_ge_085,
            "pct_ge_090": pct_ge_090,
            "rejected_pct": round((rejected_cleaned_count / total_candidates_generated * 100), 2) if total_candidates_generated else 0.0,
            "unscored_pct": round(unscored_pct, 2),
        }

        # 10. Provenance Telemetry
        with_prov = sum(1 for r in selected_records if r.metadata.provenance is not None)
        without_prov = len(selected_records) - with_prov
        prov_rate = (with_prov / len(selected_records) * 100) if selected_records else 0.0

        provenance_summary = {
            "with_provenance": with_prov,
            "without_provenance": without_prov,
            "provenance_rate": prov_rate,
            "generator_coverage": list({r.metadata.provenance.generator for r in selected_records if r.metadata.provenance and r.metadata.provenance.generator}),
        }

        # 11. Evaluate Readiness Dimensions
        dimensions: List[ReadinessDimensionScore] = []

        # (1) Schema Validity
        schema_valid = (rejected_cleaned_count == 0)
        dimensions.append(
            ReadinessDimensionScore(
                dimension="schema_validity",
                status="PASS" if schema_valid else "WARN",
                score=1.0 if schema_valid else round(accepted_cleaned_count / total_candidates_generated, 4),
                threshold="0 rejection errors",
                details=f"{accepted_cleaned_count} accepted, {rejected_cleaned_count} rejected",
            )
        )

        # (2) Domain Coverage
        active_domains_count = len(dataset_stats.get("domain_distribution", {}))
        dom_coverage_status = "PASS" if active_domains_count >= 13 else ("WARN" if active_domains_count >= 10 else "FAIL")
        dimensions.append(
            ReadinessDimensionScore(
                dimension="domain_coverage",
                status=dom_coverage_status,
                score=active_domains_count / 13.0,
                threshold="13 / 13 domains present",
                details=f"{active_domains_count} of 13 configured domains present",
            )
        )

        # (3) Difficulty Coverage
        active_diff_count = len(dataset_stats.get("difficulty_distribution", {}))
        diff_status = "PASS" if active_diff_count >= 4 else "WARN"
        dimensions.append(
            ReadinessDimensionScore(
                dimension="difficulty_coverage",
                status=diff_status,
                score=active_diff_count / 4.0,
                threshold="4 / 4 difficulty tiers",
                details=f"{active_diff_count} of 4 difficulty levels represented",
            )
        )

        # (4) Quality Assessment
        qual_status = "PASS" if (mean_qual is not None and mean_qual >= 0.85) else "WARN"
        dimensions.append(
            ReadinessDimensionScore(
                dimension="quality",
                status=qual_status,
                score=mean_qual,
                threshold="mean >= 0.85, prefer >= 0.90",
                details=f"mean score: {mean_qual}, >=0.85: {pct_ge_085}%, >=0.90: {pct_ge_090}%",
            )
        )

        # (5) Deduplication
        dedup_status = "PASS" if dup_rate <= 0.05 else "WARN"
        dimensions.append(
            ReadinessDimensionScore(
                dimension="deduplication",
                status=dedup_status,
                score=dup_rate,
                threshold="duplicate_rate <= 0.05 (5%)",
                details=f"exact: {exact_dups}, near: {near_dups}, rate: {dup_rate:.2%}",
            )
        )

        # (6) Provenance Completeness
        prov_status = "PASS" if prov_rate >= 99.0 else "WARN"
        dimensions.append(
            ReadinessDimensionScore(
                dimension="provenance",
                status=prov_status,
                score=prov_rate / 100.0,
                threshold="100% records with provenance",
                details=f"{with_prov}/{len(selected_records)} records with immutable provenance ({prov_rate:.1f}%)",
            )
        )

        # (7) Split Integrity
        train_ratio_actual = len(split_result.train) / len(selected_records) if selected_records else 0.0
        split_status = "PASS" if abs(train_ratio_actual - 0.90) <= 0.05 else "WARN"
        dimensions.append(
            ReadinessDimensionScore(
                dimension="split_integrity",
                status=split_status,
                score=train_ratio_actual,
                threshold="train: ~90%, val: ~5%, test: ~5%",
                details=f"train: {len(split_result.train)}, val: {len(split_result.validation)}, test: {len(split_result.test)}",
            )
        )

        # (8) Leakage Prevention
        dimensions.append(
            ReadinessDimensionScore(
                dimension="leakage",
                status="FAIL" if leakage_detected else "PASS",
                score=float(total_leakage),
                threshold="0 cross-split leakage records",
                details=f"train-val: {leakage_train_val}, train-test: {leakage_train_test}, val-test: {leakage_val_test}",
            )
        )

        # (9) Source Diversity
        source_type_dist = dataset_stats.get("source_statistics", {}).get("source_type_distribution", {})
        source_count = len(source_type_dist)
        source_status = "PASS" if source_count >= 2 else "WARN"
        dimensions.append(
            ReadinessDimensionScore(
                dimension="source_diversity",
                status=source_status,
                score=float(source_count),
                threshold=">= 2 source types present",
                details=f"{source_count} source types active ({', '.join(source_type_dist.keys())})",
            )
        )

        # Determine Overall Status
        if any(d.status == "FAIL" for d in dimensions):
            overall_status = "FAIL"
        elif any(d.status == "WARN" for d in dimensions):
            overall_status = "WARN"
        else:
            overall_status = "PASS"

        # 12. Build Pilot Readiness Report
        readiness_report = PilotReadinessReport(
            overall_status=overall_status,
            target_count=effective_target,
            candidate_count=total_candidates_generated,
            final_count=len(selected_records),
            dimensions=dimensions,
            domain_distribution=dataset_stats.get("domain_distribution", {}),
            difficulty_distribution=dataset_stats.get("difficulty_distribution", {}),
            task_distribution=dataset_stats.get("task_type_distribution", {}),
            source_distribution=source_type_dist,
            quality_summary=quality_summary,
            deduplication_summary={
                "exact_duplicates": exact_dups,
                "near_duplicates": near_dups,
                "duplicate_rate": dup_rate,
            },
            split_summary={
                "train_count": len(split_result.train),
                "validation_count": len(split_result.validation),
                "test_count": len(split_result.test),
                "total_count": len(selected_records),
            },
            leakage_detected=leakage_detected,
            provenance_summary=provenance_summary,
            shortages=[s.to_dict() for s in mix_result.shortages],
        )

        # 13. Build Pilot Manifest
        manifest = PilotManifest(
            pilot_version=effective_version,
            seed=effective_seed,
            target_count=effective_target,
            actual_count=len(selected_records),
            candidate_count=total_candidates_generated,
            accepted_count=accepted_cleaned_count,
            rejected_count=rejected_cleaned_count,
            exact_duplicate_count=exact_dups,
            near_duplicate_count=near_dups,
            train_count=len(split_result.train),
            validation_count=len(split_result.validation),
            test_count=len(split_result.test),
            input_sources=["synthetic_generator", "datasets/fixtures/"],
            source_versions={"sample_test_generator": "1.0.0"},
            generator_versions={"sample_test_generator": "1.0.0"},
            template_manifest_version="1.0.0",
            dataset_config_version=self.config.get("dataset", {}).get("version", "1.0.0"),
            pipeline_version=self.config.get("pipeline", {}).get("version", "1.0.0"),
            mixing_strategy=mix_result.strategy,
        )

        # 14. Persist Outputs if requested
        if save_outputs:
            raw_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)
            manifests_dir.mkdir(parents=True, exist_ok=True)

            # Raw candidates
            candidates_path = raw_dir / "pilot_candidates.jsonl"
            with open(candidates_path, "w", encoding="utf-8") as f:
                for r in candidates:
                    f.write(r.to_json() + "\n")

            # Processed Splits
            self._save_jsonl(processed_dir / "train.jsonl", split_result.train)
            self._save_jsonl(processed_dir / "validation.jsonl", split_result.validation)
            self._save_jsonl(processed_dir / "test.jsonl", split_result.test)

            # Manifest
            manifest.save_json(manifests_dir / "pilot_manifest.json")

            # Readiness Reports
            readiness_report.save_reports(reports_dir)

            # Mixing Reports
            mix_result.save_reports(reports_dir)

            # Standard Dataset & Source Reports
            report_json_file = reports_dir / "dataset_report.json"
            report_md_file = reports_dir / "dataset_report.md"
            source_json_file = reports_dir / "source_report.json"

            with open(report_json_file, "w", encoding="utf-8") as f:
                json.dump(dataset_stats, f, indent=2, ensure_ascii=False)

            with open(source_json_file, "w", encoding="utf-8") as f:
                json.dump(dataset_stats.get("source_statistics", {}), f, indent=2, ensure_ascii=False)

            md_report = stats_engine.generate_markdown_report(dataset_stats)
            with open(report_md_file, "w", encoding="utf-8") as f:
                f.write(md_report)

            # Rejection Report
            rej_path = reports_dir / "rejection_report.json"
            with open(rej_path, "w", encoding="utf-8") as f:
                json.dump(cleaning_report.to_dict(), f, indent=2, ensure_ascii=False)


        return PilotResult(
            manifest=manifest,
            readiness_report=readiness_report,
            train_records=split_result.train,
            validation_records=split_result.validation,
            test_records=split_result.test,
            candidate_records=candidates,
        )

    @staticmethod
    def _save_jsonl(path: Path, records: List[DatasetRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(r.to_json() + "\n")
