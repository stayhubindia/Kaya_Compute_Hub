"""
End-to-End Dataset Engineering Pipeline.
Orchestrates raw data ingestion, normalization, cleaning, deduplication, quality validation,
metadata enrichment, splitting, source registry tracking, and metric reporting driven by configs/dataset.yaml.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.dataset.cleaner import CleaningReport, DatasetCleaner, RejectedRecord, RejectionReason
from src.dataset.deduplicator import DeduplicationReport, DatasetDeduplicator
from src.dataset.loader import DatasetLoader, LoadingError, RawRecord
from src.dataset.metadata import MetadataEnricher
from src.dataset.normalizer import DatasetNormalizer
from src.dataset.quality import QualityValidationReport, QualityValidator
from src.dataset.schema import DatasetRecord
from src.dataset.source_registry import SourceRegistry
from src.dataset.splitter import DatasetSplitter, SplitResult
from src.dataset.statistics import DatasetStatistics


@dataclass
class PipelineResult:
    total_raw: int
    accepted_count: int
    rejected_count: int
    exact_duplicates: int
    near_duplicates: int
    split_result: SplitResult
    metrics: Dict[str, Any]
    markdown_report: str
    output_files: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_raw": self.total_raw,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "split_summary": self.split_result.to_dict(),
            "output_files": self.output_files,
        }


class DatasetPipeline:
    """Production dataset engineering pipeline for Qwen conversational datasets."""

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/dataset.yaml",
        sources_path: Optional[Union[str, Path]] = "configs/sources.yaml",
    ):
        self.config_path = Path(config_path).resolve()
        self.config = self._load_config()

        # Load Source Registry if sources.yaml is present
        self.source_registry = SourceRegistry()
        if sources_path:
            s_path = Path(sources_path).resolve()
            if s_path.is_file():
                self.source_registry.load_manifest(s_path)

        # Extract configurations
        self.model_cfg = self.config.get("model", {})
        self.domain_targets = self.config.get("domain_targets", {})
        self.difficulty_targets = self.config.get("difficulty", {}).get("targets", {})
        self.quality_cfg = self.config.get("quality", {})
        self.split_cfg = self.config.get("split", {})
        self.paths_cfg = self.config.get("paths", {})
        self.pipe_cfg = self.config.get("pipeline", {})

        # Taxonomy sets
        self.allowed_domains = set(self.config.get("domains", {}).keys())
        self.allowed_task_types = set(self.config.get("task_types", []))

        # Initialize subcomponents
        self.loader = DatasetLoader(continue_on_error=True)
        self.normalizer = DatasetNormalizer()
        
        cleaning_opts = self.pipe_cfg.get("cleaning", {})
        self.cleaner = DatasetCleaner(
            min_message_chars=cleaning_opts.get("min_message_chars", 10),
            max_message_chars=cleaning_opts.get("max_message_chars", 65536),
            allowed_domains=self.allowed_domains if self.allowed_domains else None,
            allowed_task_types=self.allowed_task_types if self.allowed_task_types else None,
        )

        dedup_opts = self.pipe_cfg.get("deduplication", {})
        self.deduplicator = DatasetDeduplicator(
            enable_near_dedup=True,
            near_duplicate_threshold=dedup_opts.get("near_duplicate_threshold", 0.85),
            ngram_size=dedup_opts.get("ngram_size", 3),
        )

        qual_opts = self.pipe_cfg.get("quality", {})
        self.quality_validator = QualityValidator(
            minimum_score=self.quality_cfg.get("minimum_score", 0.85),
            preferred_score=self.quality_cfg.get("preferred_score", 0.90),
            enforce_threshold=qual_opts.get("enforce_threshold", True),
            allow_unscored=True,
        )

        self.enricher = MetadataEnricher(
            pipeline_version=self.pipe_cfg.get("version", "1.0.0"),
            default_source_type="raw",
        )

        self.splitter = DatasetSplitter(
            train_ratio=self.split_cfg.get("train", 0.90),
            validation_ratio=self.split_cfg.get("validation", 0.05),
            test_ratio=self.split_cfg.get("test", 0.05),
            random_seed=self.pipe_cfg.get("random_seed", 42),
            stratify_by_domain=True,
        )

        self.statistics_engine = DatasetStatistics(
            domain_targets=self.domain_targets,
            difficulty_targets=self.difficulty_targets,
        )

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_outputs: bool = True,
    ) -> PipelineResult:
        """Executes full pipeline deterministically."""
        # 1. Ingest raw records
        raw_records, loading_errors = self.loader.load_path(input_path)
        total_raw = len(raw_records)

        # 2. Normalize
        normalized_records = [self.normalizer.normalize_record(r) for r in raw_records]

        # 3. Clean & Validate Schema
        cleaned_records, cleaning_report = self.cleaner.clean_records(normalized_records)

        # Add loading errors to rejection tracking
        for le in loading_errors:
            cleaning_report.add_rejection(
                RejectedRecord(
                    reason=RejectionReason.SCHEMA_VALIDATION_ERROR,
                    details=f"Loader error: {le.error_message}",
                    source_file=le.source_file,
                    line_number=le.line_number,
                    raw_preview=le.raw_line,
                )
            )

        # 4. Deduplicate (Exact & Near)
        unique_records, dedup_report = self.deduplicator.deduplicate(cleaned_records)

        # 5. Quality Validation
        quality_accepted, quality_report = self.quality_validator.validate_records(unique_records)

        # 6. Metadata Enrichment
        enriched_records = self.enricher.enrich_records(quality_accepted)

        # 7. Dataset Splitting
        split_result = self.splitter.split(enriched_records)

        # 8. Compute Statistics & Metrics
        metrics = self.statistics_engine.compute_metrics(
            raw_total=total_raw,
            accepted_records=enriched_records,
            cleaning_report=cleaning_report,
            dedup_report=dedup_report,
            quality_report=quality_report,
            split_result=split_result,
        )
        md_report = self.statistics_engine.generate_markdown_report(metrics)

        # 9. Save Artifacts (if requested)
        output_files: Dict[str, str] = {}
        if save_outputs:
            out_path = Path(output_dir) if output_dir else Path(self.paths_cfg.get("processed", "datasets/processed"))
            out_path.mkdir(parents=True, exist_ok=True)

            # Write train.jsonl, validation.jsonl, test.jsonl
            train_file = out_path / "train.jsonl"
            val_file = out_path / "validation.jsonl"
            test_file = out_path / "test.jsonl"

            self._write_jsonl(train_file, split_result.train)
            self._write_jsonl(val_file, split_result.validation)
            self._write_jsonl(test_file, split_result.test)

            # Write reports
            report_json_file = out_path / "dataset_report.json"
            report_md_file = out_path / "dataset_report.md"
            source_json_file = out_path / "source_report.json"
            rejections_file = out_path / "rejection_report.json"

            with open(report_json_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            with open(source_json_file, "w", encoding="utf-8") as f:
                json.dump(metrics.get("source_statistics", {}), f, indent=2, ensure_ascii=False)

            with open(report_md_file, "w", encoding="utf-8") as f:
                f.write(md_report)

            with open(rejections_file, "w", encoding="utf-8") as f:
                json.dump(cleaning_report.to_dict(), f, indent=2, ensure_ascii=False)

            output_files = {
                "train": str(train_file),
                "validation": str(val_file),
                "test": str(test_file),
                "report_json": str(report_json_file),
                "report_md": str(report_md_file),
                "source_json": str(source_json_file),
                "rejections": str(rejections_file),
            }

        return PipelineResult(
            total_raw=total_raw,
            accepted_count=len(enriched_records),
            rejected_count=cleaning_report.rejected_count,
            exact_duplicates=dedup_report.exact_duplicates,
            near_duplicates=dedup_report.near_duplicates,
            split_result=split_result,
            metrics=metrics,
            markdown_report=md_report,
            output_files=output_files,
        )

    def _write_jsonl(self, path: Path, records: List[DatasetRecord]):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(r.to_json() + "\n")
