"""
Production Dataset Generation Engine (Phase 3.2).
Provides batch-based synthetic candidate generation, atomic persistence, inline cleaning,
batch-local & global deduplication, quality evaluation, checkpointing, resumable recovery,
and global stratified balancing for large-scale production dataset scaling.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import statistics
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field

from src.dataset.cleaner import CleaningReport, DatasetCleaner, RejectedRecord, RejectionReason
from src.dataset.deduplicator import DeduplicationReport, DatasetDeduplicator
from src.dataset.generator import (
    GenerationRequest,
    GenerationResult,
    SampleSyntheticGenerator,
    SyntheticGeneratorInterface,
)
from src.dataset.mixer import DatasetMixer, MixingRequest, MixingResult, ShortageDetail
from src.dataset.production import (
    BatchCheckpoint,
    BatchPlan,
    BatchStatus,
    DatasetFreezeState,
    ProductionCheckpointManager,
    ProductionManifest,
    ProductionPlan,
    ProductionPlanner,
    derive_batch_seed,
)
from src.dataset.quality import QualityValidationReport, QualityValidator
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
from src.dataset.statistics import DatasetStatistics
from src.dataset.template_registry import TaskTemplate, TemplateRegistry


# ============================================================================
# 1. DATA MODELS & TELEMETRY
# ============================================================================

class BatchYieldMetrics(BaseModel):
    """Yield efficiency metrics across the generation and filtering pipeline for a batch."""
    requested_count: int
    generated_count: int
    cleaned_count: int
    deduped_count: int
    quality_accepted_count: int
    clean_rejected_count: int
    exact_duplicates: int
    near_duplicates: int
    quality_rejected_count: int
    cleaning_yield_pct: float
    dedup_yield_pct: float
    quality_yield_pct: float
    final_batch_yield_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class TemplateUsageStats(BaseModel):
    """Usage and acceptance tracking for a specific template within generation."""
    template_id: str
    domain: str
    difficulty: str
    task_type: str
    requested: int = 0
    generated: int = 0
    accepted: int = 0
    rejected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class BatchGenerationResult(BaseModel):
    """Complete outcome and telemetry of a single production batch execution."""
    batch_id: str
    batch_index: int
    seed: int
    status: str
    raw_file: Optional[str] = None
    raw_sha256: Optional[str] = None
    processed_file: Optional[str] = None
    processed_sha256: Optional[str] = None
    generated_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    yield_metrics: Optional[BatchYieldMetrics] = None
    quality_summary: Dict[str, Any] = Field(default_factory=dict)
    template_usage: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class GlobalGenerationResult(BaseModel):
    """Comprehensive result of the full multi-batch production dataset run."""
    dataset_version: str
    target_count: int
    candidate_target: int
    total_generated: int
    total_clean_accepted: int
    total_quality_accepted: int
    total_batch_unique: int
    global_deduped_count: int
    final_selected_count: int
    shortage_deficit: int
    replenishment_needed: bool
    yield_overall_pct: float
    shortages: List[Dict[str, Any]] = Field(default_factory=list)
    batch_results: List[BatchGenerationResult] = Field(default_factory=list)
    quality_summary: Dict[str, Any] = Field(default_factory=dict)
    candidate_dataset_file: Optional[str] = None
    candidate_dataset_sha256: Optional[str] = None
    manifest_file: Optional[str] = None
    report_files: Dict[str, str] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ============================================================================
# 2. ATOMIC FILE WRITING UTILITIES
# ============================================================================

def atomic_write_jsonl(records: Sequence[DatasetRecord], target_path: Union[str, Path]) -> Tuple[Path, str]:
    """
    Safely writes dataset records to a JSONL file using atomic rename semantics.
    Computes and returns the absolute path and SHA-256 checksum of the written file.
    """
    path = Path(target_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp_{uuid.uuid4().hex[:8]}")

    hasher = hashlib.sha256()
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records:
            line = r.model_dump_json() + "\n"
            b_line = line.encode("utf-8")
            f.write(line)
            hasher.update(b_line)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)
    return path, hasher.hexdigest()


# ============================================================================
# 3. PRODUCTION GENERATION ENGINE
# ============================================================================

class ProductionGenerationEngine:
    """
    Production-grade synthetic dataset generation and processing engine.
    Orchestrates batch synthesis, atomic file persistence, inline cleaning,
    batch-local deduplication, quality evaluation, checkpointing, global deduplication,
    and global mixing/balancing.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        templates_path: Optional[Union[str, Path]] = None,
        sources_path: Optional[Union[str, Path]] = None,
        generator_backend: Optional[SyntheticGeneratorInterface] = None,
    ):
        self.config_path = Path(config_path or "configs/dataset.yaml").resolve()
        self.templates_path = Path(templates_path or "configs/domain_templates.yaml").resolve()
        self.sources_path = Path(sources_path or "configs/sources.yaml").resolve()

        self.config = self._load_yaml(self.config_path)
        self.domain_targets: Dict[str, float] = self.config.get("domain_targets", {})
        self.difficulty_targets: Dict[str, float] = self.config.get("difficulty", {}).get("targets", {})
        self.prod_cfg: Dict[str, Any] = self.config.get("production", {})
        self.pipe_cfg: Dict[str, Any] = self.config.get("pipeline", {})
        self.quality_cfg: Dict[str, Any] = self.config.get("quality", {})

        # 1. Registries
        self.template_registry = TemplateRegistry()
        if self.templates_path.is_file():
            self.template_registry.load_manifest(self.templates_path)

        self.source_registry = SourceRegistry()
        if self.sources_path.is_file():
            self.source_registry.load_manifest(self.sources_path)

        # 2. Generator Backend
        self.generator: SyntheticGeneratorInterface = generator_backend or SampleSyntheticGenerator(
            generator_name="production_synthetic_engine",
            version="1.0.0",
        )

        # 3. Planner
        self.planner = ProductionPlanner(
            config_path=self.config_path,
            template_manifest_path=self.templates_path,
            source_manifest_path=self.sources_path,
        )

        # 4. Processing Subsystems
        cleaning_opts = self.pipe_cfg.get("cleaning", {})
        self.cleaner = DatasetCleaner(
            min_message_chars=cleaning_opts.get("min_message_chars", 10),
            max_message_chars=cleaning_opts.get("max_message_chars", 65536),
            allowed_domains=set(self.domain_targets.keys()) if self.domain_targets else None,
        )

        dedup_opts = self.pipe_cfg.get("deduplication", {})
        self.deduplicator = DatasetDeduplicator(
            enable_near_dedup=True,
            near_duplicate_threshold=dedup_opts.get("near_duplicate_threshold", 0.85),
            ngram_size=dedup_opts.get("ngram_size", 3),
        )

        self.quality_validator = QualityValidator(
            minimum_score=self.quality_cfg.get("minimum_score", 0.85),
            preferred_score=self.quality_cfg.get("preferred_score", 0.90),
            enforce_threshold=True,
            allow_unscored=True,
        )

        self.mixer = DatasetMixer(
            config_path=self.config_path,
        )


    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def plan_batches(
        self,
        target_count: Optional[int] = None,
        seed: Optional[int] = None,
        candidate_multiplier: Optional[float] = None,
        batch_size: Optional[int] = None,
        version: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> ProductionPlan:
        """Generates a complete mathematical dry-run production plan."""
        return self.planner.plan(
            target_count=target_count,
            seed=seed,
            candidate_multiplier=candidate_multiplier,
            batch_size=batch_size,
            version=version,
            output_dir=output_dir,
        )

    def _select_compatible_templates(
        self,
        domain: str,
        difficulty: str,
    ) -> List[TaskTemplate]:
        """Retrieves and filters registered templates matching domain and difficulty."""
        domain_tmpls = self.template_registry.list_by_domain(domain)
        if not domain_tmpls:
            return []

        matching = [t for t in domain_tmpls if difficulty in t.supported_difficulties]
        return matching if matching else domain_tmpls


    def generate_batch(
        self,
        batch_plan: BatchPlan,
        checkpoint_mgr: ProductionCheckpointManager,
        output_dir: Union[str, Path],
        force: bool = False,
        fail_fast: bool = False,
        dataset_version: str = "dataset-v1.0",
    ) -> BatchGenerationResult:
        """
        Executes generation and processing for a single batch with atomic writes and checkpoint updates.
        """
        out_root = Path(output_dir).resolve()
        raw_dir = out_root / "raw"
        batch_dir = out_root / "batches" / batch_plan.batch_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        batch_dir.mkdir(parents=True, exist_ok=True)

        raw_file_path = raw_dir / f"{batch_plan.batch_id}.jsonl"
        processed_file_path = batch_dir / "processed_candidates.jsonl"
        batch_report_path = batch_dir / "report.json"
        batch_meta_path = batch_dir / "metadata.json"

        # 1. Checkpoint Check & Resume Check
        existing_ckpt = checkpoint_mgr.load_checkpoint(batch_plan.batch_id)
        if not force and existing_ckpt and existing_ckpt.status == BatchStatus.COMPLETED.value:
            if processed_file_path.is_file():
                # Load existing processed records
                records: List[DatasetRecord] = []
                with open(processed_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(DatasetRecord.model_validate_json(line))
                
                # Construct result from existing checkpoint
                return BatchGenerationResult(
                    batch_id=batch_plan.batch_id,
                    batch_index=batch_plan.batch_index,
                    seed=batch_plan.seed,
                    status=BatchStatus.COMPLETED.value,
                    raw_file=str(raw_file_path) if raw_file_path.is_file() else None,
                    raw_sha256=existing_ckpt.checksum_sha256,
                    processed_file=str(processed_file_path),
                    processed_sha256=existing_ckpt.checksum_sha256,
                    generated_count=existing_ckpt.generated_count,
                    accepted_count=len(records),
                    rejected_count=existing_ckpt.rejected_count,
                    quality_summary=existing_ckpt.quality_stats,
                    started_at=existing_ckpt.started_at,
                    completed_at=existing_ckpt.completed_at,
                )

        # 2. Mark Checkpoint as GENERATING
        now_start = datetime.now(timezone.utc).isoformat()
        ckpt = BatchCheckpoint(
            batch_id=batch_plan.batch_id,
            batch_index=batch_plan.batch_index,
            status=BatchStatus.GENERATING.value,
            seed=batch_plan.seed,
            requested_count=batch_plan.candidate_target,
            started_at=now_start,
        )
        checkpoint_mgr.save_checkpoint(ckpt)

        try:
            # 3. Candidate Synthesis loop driven by batch matrix
            candidates: List[DatasetRecord] = []
            template_usage: Dict[str, TemplateUsageStats] = {}
            record_template_map: Dict[str, str] = {}

            # Build list of (domain, difficulty) generation targets
            candidate_quota_per_cell: List[Tuple[str, str]] = []
            total_candidate_quota = batch_plan.candidate_target

            # Proportional domain x difficulty distribution for candidates
            if batch_plan.matrix:
                # Apportion candidate target using the batch's matrix weights
                matrix_total = sum(sum(cells.values()) for cells in batch_plan.matrix.values())
                if matrix_total > 0:
                    for dom, cells in sorted(batch_plan.matrix.items()):
                        for diff, q in sorted(cells.items()):
                            if q > 0:
                                # Scale up proportionally to batch candidate target
                                scaled_q = int(round((q / matrix_total) * total_candidate_quota))
                                for _ in range(scaled_q):
                                    candidate_quota_per_cell.append((dom, diff))

            # Adjust count if rounding differed from candidate_target
            if len(candidate_quota_per_cell) < total_candidate_quota:
                doms = sorted(self.domain_targets.keys())
                diffs = sorted(self.difficulty_targets.keys())
                for k in range(total_candidate_quota - len(candidate_quota_per_cell)):
                    candidate_quota_per_cell.append((doms[k % len(doms)], diffs[k % len(diffs)]))
            elif len(candidate_quota_per_cell) > total_candidate_quota:
                candidate_quota_per_cell = candidate_quota_per_cell[:total_candidate_quota]

            # Deterministic RNG for this batch
            batch_rng = random.Random(batch_plan.seed)
            batch_rng.shuffle(candidate_quota_per_cell)

            # Generate candidate records
            for idx, (dom, diff) in enumerate(candidate_quota_per_cell):
                compatible_templates = self._select_compatible_templates(dom, diff)
                item_seed = batch_plan.seed + (idx * 37) + 1

                if compatible_templates:
                    # Select template in deterministic round-robin with seed offset
                    tmpl_idx = (idx + batch_plan.seed) % len(compatible_templates)
                    tmpl = compatible_templates[tmpl_idx]
                    t_id = tmpl.id
                    t_task = tmpl.task_type

                    if t_id not in template_usage:
                        template_usage[t_id] = TemplateUsageStats(
                            template_id=t_id,
                            domain=dom,
                            difficulty=diff,
                            task_type=t_task,
                        )
                    template_usage[t_id].requested += 1

                    try:
                        recs = self.generator.generate_from_template(
                            template=tmpl,
                            number_of_examples=1,
                            difficulty=diff,
                            seed=item_seed,
                            batch_id=batch_plan.batch_id,
                        )
                        record = recs[0]
                    except Exception:
                        req = GenerationRequest(
                            domain=dom,
                            topic=tmpl.topic,
                            task_type=tmpl.task_type,
                            difficulty=diff,
                            number_of_examples=1,
                            seed=item_seed,
                            generation_batch_id=batch_plan.batch_id,
                        )
                        gen_res = self.generator.generate_batch(req)
                        record = gen_res.records[0]
                else:
                    t_id = f"generic_{dom}_{diff}"
                    t_task = "explanation"
                    if t_id not in template_usage:
                        template_usage[t_id] = TemplateUsageStats(
                            template_id=t_id,
                            domain=dom,
                            difficulty=diff,
                            task_type=t_task,
                        )
                    template_usage[t_id].requested += 1

                    req = GenerationRequest(
                        domain=dom,
                        topic="general",
                        task_type=t_task,
                        difficulty=diff,
                        number_of_examples=1,
                        seed=item_seed,
                        generation_batch_id=batch_plan.batch_id,
                    )
                    gen_res = self.generator.generate_batch(req)
                    record = gen_res.records[0]

                # Attach robust immutable ProvenanceInfo deterministically
                synth_id = f"synth_{t_id}_{batch_plan.batch_id}_{item_seed}_{idx:04d}"
                det_created_at = "2026-01-01T00:00:00+00:00"
                record.metadata.provenance = ProvenanceInfo(
                    source_type=SourceType.SYNTHETIC,
                    source=f"synthetic_template:{t_id}",
                    source_id=synth_id,
                    generator="sample_synthetic_generator",
                    generator_version="1.0.0",
                    created_at=det_created_at,
                    license="MIT / Synthetic Generated",
                    source_url=None,
                )
                record.metadata.created_at = det_created_at
                record.metadata.domain = dom
                record.metadata.difficulty = diff
                record.metadata.task_type = t_task


                record_template_map[synth_id] = t_id
                template_usage[t_id].generated += 1
                candidates.append(record)

            raw_generated_count = len(candidates)

            # 4. Atomic Write of Raw Batch Records
            raw_path, raw_sha256 = atomic_write_jsonl(candidates, raw_file_path)

            # 5. Inline Processing: Cleaning & Validation
            candidate_dicts = [r.to_dict() for r in candidates]
            cleaned_records, cleaning_report = self.cleaner.clean_records(candidate_dicts)
            clean_accepted_count = len(cleaned_records)
            clean_rejected_count = len(cleaning_report.rejected_records)

            # 6. Inline Processing: Batch-Local Deduplication
            deduped_records, dedup_report = self.deduplicator.deduplicate(cleaned_records)
            exact_dups = dedup_report.exact_duplicates
            near_dups = dedup_report.near_duplicates
            dedup_accepted_count = len(deduped_records)

            # 7. Inline Processing: Quality Evaluation
            quality_records, quality_report = self.quality_validator.validate_records(deduped_records)
            final_accepted_count = len(quality_records)
            quality_rejected_count = quality_report.failed_count

            # Update template usage acceptance
            accepted_ids = {
                r.metadata.provenance.source_id for r in quality_records
                if r.metadata.provenance and r.metadata.provenance.source_id
            }
            for r in candidates:
                if r.metadata.provenance and r.metadata.provenance.source_id:
                    sid = r.metadata.provenance.source_id
                    t_key = record_template_map.get(sid, "unknown")
                    if t_key in template_usage:
                        if sid in accepted_ids:
                            template_usage[t_key].accepted += 1
                        else:
                            template_usage[t_key].rejected += 1


            # 8. Compute Batch Yield Metrics
            clean_yield = (clean_accepted_count / raw_generated_count * 100.0) if raw_generated_count > 0 else 0.0
            dedup_yield = (dedup_accepted_count / clean_accepted_count * 100.0) if clean_accepted_count > 0 else 0.0
            qual_yield = (final_accepted_count / dedup_accepted_count * 100.0) if dedup_accepted_count > 0 else 0.0
            final_yield = (final_accepted_count / raw_generated_count * 100.0) if raw_generated_count > 0 else 0.0

            yield_metrics = BatchYieldMetrics(
                requested_count=batch_plan.candidate_target,
                generated_count=raw_generated_count,
                cleaned_count=clean_accepted_count,
                deduped_count=dedup_accepted_count,
                quality_accepted_count=final_accepted_count,
                clean_rejected_count=clean_rejected_count,
                exact_duplicates=exact_dups,
                near_duplicates=near_dups,
                quality_rejected_count=quality_rejected_count,
                cleaning_yield_pct=round(clean_yield, 2),
                dedup_yield_pct=round(dedup_yield, 2),
                quality_yield_pct=round(qual_yield, 2),
                final_batch_yield_pct=round(final_yield, 2),
            )

            # 9. Compute Quality Summary
            q_scores = [r.metadata.quality_score for r in quality_records if r.metadata.quality_score is not None]
            if q_scores:
                mean_q = round(statistics.mean(q_scores), 4)
                median_q = round(statistics.median(q_scores), 4)
                min_q = round(min(q_scores), 4)
                max_q = round(max(q_scores), 4)
                pct_085 = round((sum(1 for s in q_scores if s >= 0.85) / len(q_scores)) * 100.0, 2)
                pct_090 = round((sum(1 for s in q_scores if s >= 0.90) / len(q_scores)) * 100.0, 2)
            else:
                mean_q, median_q, min_q, max_q, pct_085, pct_090 = None, None, None, None, 0.0, 0.0

            quality_summary = {
                "mean_score": mean_q,
                "median_score": median_q,
                "min_score": min_q,
                "max_score": max_q,
                "pct_ge_085": pct_085,
                "pct_ge_090": pct_090,
                "total_evaluated": len(quality_records),
            }

            # 10. Atomic Write Processed Batch Records
            proc_path, proc_sha256 = atomic_write_jsonl(quality_records, processed_file_path)

            # 11. Write Batch Metadata & Report JSON
            batch_result = BatchGenerationResult(
                batch_id=batch_plan.batch_id,
                batch_index=batch_plan.batch_index,
                seed=batch_plan.seed,
                status=BatchStatus.COMPLETED.value,
                raw_file=str(raw_path),
                raw_sha256=raw_sha256,
                processed_file=str(proc_path),
                processed_sha256=proc_sha256,
                generated_count=raw_generated_count,
                accepted_count=final_accepted_count,
                rejected_count=clean_rejected_count + exact_dups + near_dups + quality_rejected_count,
                yield_metrics=yield_metrics,
                quality_summary=quality_summary,
                template_usage={k: v.to_dict() for k, v in template_usage.items()},
                started_at=now_start,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

            with open(batch_report_path, "w", encoding="utf-8") as f:
                json.dump(batch_result.to_dict(), f, indent=2)

            batch_meta = {
                "dataset_version": dataset_version,
                "batch_id": batch_plan.batch_id,
                "batch_index": batch_plan.batch_index,
                "seed": batch_plan.seed,
                "requested": batch_plan.candidate_target,
                "generated": raw_generated_count,
                "accepted": final_accepted_count,
                "rejected": batch_result.rejected_count,
                "raw_file": str(raw_path),
                "processed_file": str(proc_path),
                "sha256": proc_sha256,
            }
            with open(batch_meta_path, "w", encoding="utf-8") as f:
                json.dump(batch_meta, f, indent=2)

            # 12. Update Checkpoint to COMPLETED
            ckpt.status = BatchStatus.COMPLETED.value
            ckpt.generated_count = raw_generated_count
            ckpt.accepted_count = final_accepted_count
            ckpt.rejected_count = batch_result.rejected_count
            ckpt.duplicate_count = exact_dups + near_dups
            ckpt.quality_stats = quality_summary
            ckpt.output_file = str(proc_path)
            ckpt.checksum_sha256 = proc_sha256
            ckpt.completed_at = batch_result.completed_at
            checkpoint_mgr.save_checkpoint(ckpt)

            return batch_result

        except Exception as e:
            checkpoint_mgr.handle_batch_failure(batch_plan.batch_id, e, fail_fast=fail_fast)

            return BatchGenerationResult(

                batch_id=batch_plan.batch_id,
                batch_index=batch_plan.batch_index,
                seed=batch_plan.seed,
                status=BatchStatus.FAILED.value,
                generated_count=0,
                accepted_count=0,
                rejected_count=0,
                error_message=str(e),
                started_at=now_start,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

    def generate_all(
        self,
        target_count: Optional[int] = None,
        seed: Optional[int] = None,
        candidate_multiplier: Optional[float] = None,
        batch_size: Optional[int] = None,
        version: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        resume: bool = True,
        retry_failed: bool = False,
        max_batches: Optional[int] = None,
        dry_run: bool = False,
        fail_fast: bool = False,
    ) -> GlobalGenerationResult:
        """
        Executes full multi-batch production generation, global deduplication,
        global stratified mixing, deficit tracking, and global report emission.
        """
        # 1. Establish Production Plan
        plan = self.planner.plan(
            target_count=target_count,
            seed=seed,
            candidate_multiplier=candidate_multiplier,
            batch_size=batch_size,
            version=version,
            output_dir=output_dir,
        )

        out_root = Path(plan.storage_layout["root"]).resolve()
        checkpoints_dir = out_root / "checkpoints"
        manifests_dir = out_root / "manifests"
        reports_dir = out_root / "reports"
        processed_dir = out_root / "processed"
        batches_dir = out_root / "batches"

        for d in [checkpoints_dir, manifests_dir, reports_dir, processed_dir, batches_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Handle Dry-Run
        if dry_run:
            json_p, md_p = plan.save_reports(reports_dir)
            manifest = self.planner.create_initial_manifest(plan)
            man_path = manifests_dir / "production_manifest.json"
            manifest.save(man_path)
            return GlobalGenerationResult(
                dataset_version=plan.version,
                target_count=plan.target_count,
                candidate_target=plan.candidate_target,
                total_generated=0,
                total_clean_accepted=0,
                total_quality_accepted=0,
                total_batch_unique=0,
                global_deduped_count=0,
                final_selected_count=0,
                shortage_deficit=plan.target_count,
                replenishment_needed=True,
                yield_overall_pct=0.0,
                manifest_file=str(man_path),
                report_files={"plan_json": str(json_p), "plan_md": str(md_p)},
            )

        # 2. Checkpoint Manager & Manifest
        checkpoint_mgr = ProductionCheckpointManager(checkpoints_dir)
        manifest_path = manifests_dir / "production_manifest.json"
        if not manifest_path.is_file() and (out_root / "production_manifest.json").is_file():
            manifest_path = out_root / "production_manifest.json"

        if manifest_path.is_file():
            manifest = ProductionManifest.load(manifest_path)
            if manifest.status == DatasetFreezeState.FROZEN.value and manifest.dataset_version == plan.version:
                raise RuntimeError(
                    f"Dataset '{manifest.dataset_version}' is FROZEN and immutable. "
                    f"Generation commands cannot modify a frozen dataset. "
                    f"Please specify a new version (e.g. dataset-v1.1) to generate new datasets."
                )
        else:
            manifest = self.planner.create_initial_manifest(plan)

        manifest.transition_state(DatasetFreezeState.GENERATING)
        manifest.save(manifest_path)

        # 3. Batch Generation Execution
        batch_results: List[BatchGenerationResult] = []
        all_candidate_records: List[DatasetRecord] = []

        batches_to_run = plan.batch_plans
        if max_batches is not None and max_batches > 0:
            batches_to_run = batches_to_run[:max_batches]

        for bp in batches_to_run:
            is_completed = checkpoint_mgr.is_batch_completed(bp.batch_id)
            ckpt = checkpoint_mgr.load_checkpoint(bp.batch_id)
            is_failed = ckpt is not None and ckpt.status == BatchStatus.FAILED.value

            force_run = False
            if is_failed and retry_failed:
                force_run = True
            elif not resume:
                force_run = True

            b_res = self.generate_batch(
                batch_plan=bp,
                checkpoint_mgr=checkpoint_mgr,
                output_dir=out_root,
                force=force_run,
                fail_fast=fail_fast,
                dataset_version=plan.version,
            )
            batch_results.append(b_res)

            # Collect processed records if batch succeeded
            if b_res.status == BatchStatus.COMPLETED.value and b_res.processed_file:
                p_file = Path(b_res.processed_file)
                if p_file.is_file():
                    with open(p_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                all_candidate_records.append(DatasetRecord.model_validate_json(line))

        # 4. Global Deduplication across all accepted candidates
        global_deduped_records, global_dedup_report = self.deduplicator.deduplicate(all_candidate_records)
        global_dedup_count = len(global_deduped_records)
        global_exact_dups = global_dedup_report.exact_duplicates
        global_near_dups = global_dedup_report.near_duplicates

        # 5. Global Mixing & Balancing targeting target_count
        if max_batches is None:
            effective_target = max(1, plan.target_count)
        else:
            effective_target = max(1, min(plan.target_count, len(global_deduped_records))) if global_deduped_records else max(1, plan.target_count)

        mix_request = MixingRequest(

            target_count=effective_target,
            strategy="proportional",
            seed=plan.seed,
            allow_oversampling=False,
            allow_undersampling=True,
            batch_id=f"global_mix_{plan.version}",
        )

        mix_result = self.mixer.mix(mix_request, candidate_records=global_deduped_records)
        final_selected_records = mix_result.records
        final_count = len(final_selected_records)

        # 6. Evaluate Candidate Yield & Automatic Replenishment Deficit
        deficit = max(0, plan.target_count - final_count) if max_batches is None else 0
        replenishment_needed = deficit > 0

        # 7. Atomic Write of Assembled Candidate Dataset
        cand_dataset_file = processed_dir / "candidate_dataset.jsonl"
        saved_cand_path, cand_sha256 = atomic_write_jsonl(final_selected_records, cand_dataset_file)

        # 8. Compute Global Telemetry & Quality Summary
        total_raw = sum(b.generated_count for b in batch_results)
        total_clean = sum(b.yield_metrics.cleaned_count for b in batch_results if b.yield_metrics)
        total_quality = sum(b.yield_metrics.quality_accepted_count for b in batch_results if b.yield_metrics)
        total_batch_unique = sum(b.accepted_count for b in batch_results)
        overall_yield = (final_count / total_raw * 100.0) if total_raw > 0 else 0.0

        all_quality_scores = [
            r.metadata.quality_score for r in final_selected_records if r.metadata.quality_score is not None
        ]
        if all_quality_scores:
            g_mean_q = round(statistics.mean(all_quality_scores), 4)
            g_median_q = round(statistics.median(all_quality_scores), 4)
            g_min_q = round(min(all_quality_scores), 4)
            g_max_q = round(max(all_quality_scores), 4)
            g_pct_085 = round((sum(1 for s in all_quality_scores if s >= 0.85) / len(all_quality_scores)) * 100.0, 2)
            g_pct_090 = round((sum(1 for s in all_quality_scores if s >= 0.90) / len(all_quality_scores)) * 100.0, 2)
        else:
            g_mean_q, g_median_q, g_min_q, g_max_q, g_pct_085, g_pct_090 = None, None, None, None, 0.0, 0.0

        global_quality_summary = {
            "mean_score": g_mean_q,
            "median_score": g_median_q,
            "min_score": g_min_q,
            "max_score": g_max_q,
            "pct_ge_085": g_pct_085,
            "pct_ge_090": g_pct_090,
            "total_evaluated": len(final_selected_records),
        }

        # 9. Generate Global Reports
        report_files: Dict[str, str] = {}

        # a. Generation Report JSON & MD
        gen_rep_json = reports_dir / "generation_report.json"
        gen_rep_md = reports_dir / "generation_report.md"
        gen_data = {
            "dataset_version": plan.version,
            "target_count": plan.target_count,
            "candidate_target": plan.candidate_target,
            "total_raw_generated": total_raw,
            "total_clean_accepted": total_clean,
            "total_quality_accepted": total_quality,
            "total_batch_unique": total_batch_unique,
            "global_deduped_count": global_dedup_count,
            "final_selected_count": final_count,
            "deficit": deficit,
            "replenishment_needed": replenishment_needed,
            "overall_yield_pct": round(overall_yield, 2),
            "completed_batches": len([b for b in batch_results if b.status == BatchStatus.COMPLETED.value]),
            "failed_batches": len([b for b in batch_results if b.status == BatchStatus.FAILED.value]),
            "global_exact_duplicates": global_exact_dups,
            "global_near_duplicates": global_near_dups,
        }
        with open(gen_rep_json, "w", encoding="utf-8") as f:
            json.dump(gen_data, f, indent=2)

        md_gen_lines = [
            f"# Production Generation Telemetry Report — `{plan.version}`",
            "",
            "## 1. Candidate Synthesis & Yield Overview",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Target Final Examples** | `{plan.target_count:,}` |",
            f"| **Candidate Target Pool** | `{plan.candidate_target:,}` |",
            f"| **Raw Generated Candidates** | `{total_raw:,}` |",
            f"| **Clean Accepted Candidates** | `{total_clean:,}` |",
            f"| **Quality Accepted Candidates** | `{total_quality:,}` |",
            f"| **Batch-Local Unique Candidates** | `{total_batch_unique:,}` |",
            f"| **Global Deduped Candidates** | `{global_dedup_count:,}` |",
            f"| **Final Selected Dataset Count** | `{final_count:,}` |",
            f"| **Shortage Deficit** | `{deficit:,}` |",
            f"| **Replenishment Needed** | `{'YES' if replenishment_needed else 'NO'}` |",
            f"| **Overall Yield** | `{overall_yield:.2f}%` |",
            "",
            "---",
            "",
            "## 2. Batch Execution Breakdown",
            "",
            "| Batch ID | Index | Status | Generated | Accepted | Yield % | SHA-256 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for b in batch_results:
            b_yield = b.yield_metrics.final_batch_yield_pct if b.yield_metrics else 0.0
            sha_trunc = (b.processed_sha256[:10] + "...") if b.processed_sha256 else "—"
            md_gen_lines.append(
                f"| `{b.batch_id}` | `{b.batch_index:03d}` | `{b.status}` | `{b.generated_count:,}` | `{b.accepted_count:,}` | `{b_yield:.2f}%` | `{sha_trunc}` |"
            )
        md_gen_lines.append("")
        with open(gen_rep_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_gen_lines))

        report_files["generation_report_json"] = str(gen_rep_json)
        report_files["generation_report_md"] = str(gen_rep_md)

        # b. Dataset Mix Report
        mix_rep_json = reports_dir / "dataset_mix_report.json"
        mix_rep_md = reports_dir / "dataset_mix_report.md"
        with open(mix_rep_json, "w", encoding="utf-8") as f:
            json.dump(mix_result.to_dict(), f, indent=2)
        with open(mix_rep_md, "w", encoding="utf-8") as f:
            f.write(mix_result.generate_markdown_report())
        report_files["dataset_mix_report_json"] = str(mix_rep_json)
        report_files["dataset_mix_report_md"] = str(mix_rep_md)

        # c. Quality Report
        qual_rep_json = reports_dir / "quality_report.json"
        with open(qual_rep_json, "w", encoding="utf-8") as f:
            json.dump(global_quality_summary, f, indent=2)
        report_files["quality_report_json"] = str(qual_rep_json)

        # d. Source Report
        src_rep_json = reports_dir / "source_report.json"
        src_counts = defaultdict(int)
        for r in final_selected_records:
            stype = r.metadata.source_type if r.metadata else "unknown"
            src_counts[stype] += 1
        with open(src_rep_json, "w", encoding="utf-8") as f:
            json.dump({"source_distribution": dict(src_counts)}, f, indent=2)
        report_files["source_report_json"] = str(src_rep_json)

        # e. Rejection Report
        rej_rep_json = reports_dir / "rejection_report.json"
        rej_data = {
            "total_clean_rejected": sum(b.yield_metrics.clean_rejected_count for b in batch_results if b.yield_metrics),
            "total_exact_duplicates": sum(b.yield_metrics.exact_duplicates for b in batch_results if b.yield_metrics) + global_exact_dups,
            "total_near_duplicates": sum(b.yield_metrics.near_duplicates for b in batch_results if b.yield_metrics) + global_near_dups,
            "total_quality_rejected": sum(b.yield_metrics.quality_rejected_count for b in batch_results if b.yield_metrics),
        }
        with open(rej_rep_json, "w", encoding="utf-8") as f:
            json.dump(rej_data, f, indent=2)
        report_files["rejection_report_json"] = str(rej_rep_json)

        # f. Combined Production Generation Report JSON
        prod_gen_rep = reports_dir / "production_generation_report.json"
        final_global_res = GlobalGenerationResult(
            dataset_version=plan.version,
            target_count=plan.target_count,
            candidate_target=plan.candidate_target,
            total_generated=total_raw,
            total_clean_accepted=total_clean,
            total_quality_accepted=total_quality,
            total_batch_unique=total_batch_unique,
            global_deduped_count=global_dedup_count,
            final_selected_count=final_count,
            shortage_deficit=deficit,
            replenishment_needed=replenishment_needed,
            yield_overall_pct=round(overall_yield, 2),
            shortages=[s.to_dict() for s in mix_result.shortages],
            batch_results=batch_results,
            quality_summary=global_quality_summary,
            candidate_dataset_file=str(saved_cand_path),
            candidate_dataset_sha256=cand_sha256,
            manifest_file=str(manifest_path),
            report_files=report_files,
        )

        with open(prod_gen_rep, "w", encoding="utf-8") as f:
            json.dump(final_global_res.to_dict(), f, indent=2)

        # 10. Update Production Manifest (transition to VALIDATING)
        manifest.transition_state(DatasetFreezeState.VALIDATING)
        manifest.actual_candidate_count = total_raw
        manifest.actual_final_count = final_count
        manifest.checksums["candidate_dataset.jsonl"] = cand_sha256
        manifest.batch_summaries = [b.to_dict() for b in batch_results]
        manifest.save(manifest_path)

        return final_global_res
