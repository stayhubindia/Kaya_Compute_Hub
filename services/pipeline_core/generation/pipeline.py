"""
Scientific Instruction Generation Pipeline (Phase 3.4).
Orchestrates:
1. Source Chunk Ingestion & Deep Analysis
2. Equation- & Table-Aware Instruction Synthesis (17 Tasks)
3. Hard Grounding Validation Gate (Zero-Hallucination)
4. Multi-Dimensional Quality Evaluation (Threshold >= 0.85)
5. Two-Stage Deduplication (Exact + Near)
6. Source-Aware Leakage-Proof Splitting (90/5/5)
7. Report and Manifest Emission with Checkpointing
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

from pydantic import BaseModel, Field

from src.dataset.deduplicator import DatasetDeduplicator, DeduplicationReport
from src.dataset.schema import DatasetRecord, ProvenanceInfo
from src.generation.answer_generator import ScientificInstructionDispatcher
from src.generation.checkpoint_manager import ChunkCheckpointManager
from src.generation.grounding_validator import GroundingValidationOutcome, ScientificGroundingValidator
from src.generation.knowledge_selector import KnowledgeSelector
from src.generation.models import (
    CandidateGenerationPolicy,
    CandidateRecord,
    ChunkAnalysis,
    ExtendedProvenance,
    KnowledgeUnit,
    ScientificGroundingStatus,
    ScientificTaskType,
)
from src.generation.prompt_builder import InstructionPromptBuilder
from src.generation.quality import InstructionQualityAuditor
from src.generation.quality_evaluator import ScientificQualityEvaluator
from src.generation.scientific_synthesizer import ScientificInstructionSynthesizer
from src.generation.source_analyzer import SourceChunkAnalyzer
from src.generation.source_aware_splitter import SourceAwareSplitResult, SourceAwareSplitter
from src.generation.statistics import GenerationStatisticsAggregator
from src.generation.task_selector import TaskSelector
from src.generation.validator import InstructionValidator
from src.ingestion.models import KnowledgeChunk


def atomic_write_jsonl(records: List[DatasetRecord], target_path: Path) -> Tuple[Path, str]:
    """Writes dataset records atomically to JSONL and returns file path & SHA256."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(f".tmp_{os.getpid()}")
    hasher = hashlib.sha256()

    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records:
            line = r.model_dump_json() + "\n"
            b_line = line.encode("utf-8")
            f.write(line)
            hasher.update(b_line)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, target_path)
    return target_path, hasher.hexdigest()


class PipelineSummary(BaseModel):
    """Execution telemetry and outcome for the generation pipeline."""
    execution_id: str
    status: str
    lifecycle: str = "READY"
    version: str = "dataset-v2.0"
    seed: int = 42
    chunks_discovered: int = 0
    chunks_processed: int = 0
    chunks_with_equations: int = 0
    chunks_with_tables: int = 0
    candidates_generated: int = 0
    candidates_accepted: int = 0
    candidates_rejected: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    unique_candidates: int = 0
    train_count: int = 0
    validation_count: int = 0
    test_count: int = 0
    quality_summary: Dict[str, Any] = Field(default_factory=dict)
    domain_distribution: Dict[str, int] = Field(default_factory=dict)
    task_distribution: Dict[str, int] = Field(default_factory=dict)
    difficulty_distribution: Dict[str, int] = Field(default_factory=dict)
    equation_stats: Dict[str, Any] = Field(default_factory=dict)
    table_stats: Dict[str, Any] = Field(default_factory=dict)
    leakage_summary: Dict[str, Any] = Field(default_factory=dict)
    manifest_path: Optional[str] = None
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None


class ScientificGenerationPipeline:
    """Full end-to-end scientific dataset generation and validation engine."""

    def __init__(
        self,
        input_chunks_path: Union[str, Path] = "data/ingested/nptel_corpus/chunks.jsonl",
        documents_path: Optional[Union[str, Path]] = "data/ingested/nptel_corpus/documents.jsonl",
        output_dir: Union[str, Path] = "data/instruction_dataset/v2.0",
        policy: Optional[CandidateGenerationPolicy] = None,
        seed: int = 42,
    ):
        p_in = Path(input_chunks_path).resolve()
        if p_in.is_dir():
            self.input_chunks_path = p_in / "chunks.jsonl"
            if not documents_path and (p_in / "documents.jsonl").is_file():
                self.documents_path = (p_in / "documents.jsonl").resolve()
            else:
                self.documents_path = Path(documents_path).resolve() if documents_path else None
        else:
            self.input_chunks_path = p_in
            self.documents_path = Path(documents_path).resolve() if documents_path else None
        self.output_dir = Path(output_dir).resolve()
        self.policy = policy or CandidateGenerationPolicy(deterministic_seed=seed)
        self.seed = seed

        # Subsystems
        self.analyzer = SourceChunkAnalyzer()
        self.synthesizer = ScientificInstructionSynthesizer(
            generator_name="scientific_instruction_synthesizer",
            version="2.0.0",
        )
        self.grounding_validator = ScientificGroundingValidator(min_grounding_overlap=0.20)
        self.quality_evaluator = ScientificQualityEvaluator(
            min_score=self.policy.min_quality_score,
            preferred_score=self.policy.preferred_quality_score,
        )
        self.deduplicator = DatasetDeduplicator(enable_near_dedup=True, near_duplicate_threshold=0.85)
        self.splitter = SourceAwareSplitter(random_seed=seed)

        # Directory layout
        self.raw_dir = self.output_dir / "raw"
        self.proc_dir = self.output_dir / "processed"
        self.splits_dir = self.output_dir / "splits"
        self.reports_dir = self.output_dir / "reports"
        self.manifests_dir = self.output_dir / "manifests"
        self.checkpoints_dir = self.output_dir / "checkpoints"

    def _load_documents_index(self) -> Dict[str, Dict[str, Any]]:
        """Indexes documents.jsonl by document_id for fast metadata lookup."""
        doc_index = {}
        if self.documents_path and self.documents_path.is_file():
            with open(self.documents_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        doc_id = d.get("document_id")
                        if doc_id:
                            doc_index[doc_id] = d
        return doc_index

    def execute_dry_run(self, max_chunks: Optional[int] = None) -> PipelineSummary:
        """Performs a read-only audit and analysis without generating records."""
        for d in [self.reports_dir, self.manifests_dir]:
            d.mkdir(parents=True, exist_ok=True)

        doc_index = self._load_documents_index()
        raw_chunks = []
        with open(self.input_chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_chunks.append(json.loads(line))

        if max_chunks:
            raw_chunks = raw_chunks[:max_chunks]

        total_chunks = len(raw_chunks)
        chunks_with_eqs = 0
        chunks_with_tbls = 0
        task_counts = Counter()
        diff_counts = Counter()
        domain_counts = Counter()

        for c_dict in raw_chunks:
            doc_id = c_dict.get("document_id")
            doc_meta = doc_index.get(doc_id)
            analysis = self.analyzer.analyze_chunk(c_dict, doc_meta)

            if analysis.equations:
                chunks_with_eqs += 1
            if analysis.tables:
                chunks_with_tbls += 1

            for t in analysis.suitable_tasks:
                task_counts[t.value] += 1
            diff_counts[analysis.natural_difficulty] += 1
            domain_counts[f"{analysis.domain}/{analysis.topic}"] += 1

        est_candidates = sum(min(3, max(1, len(c.get("text", "").split()) // 80)) for c in raw_chunks)

        summary = PipelineSummary(
            execution_id=f"dry_run_{int(datetime.now().timestamp())}",
            status="DRY_RUN_COMPLETED",
            lifecycle="DRY_RUN",
            seed=self.seed,
            chunks_discovered=total_chunks,
            chunks_processed=total_chunks,
            chunks_with_equations=chunks_with_eqs,
            chunks_with_tables=chunks_with_tbls,
            candidates_generated=0,
            candidates_accepted=0,
            candidates_rejected=0,
            quality_summary={"estimated_candidates": est_candidates},
            domain_distribution=dict(domain_counts),
            task_distribution=dict(task_counts),
            difficulty_distribution=dict(diff_counts),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        # Save dry-run reports
        with open(self.reports_dir / "generation_report.json", "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))

        return summary

    def run(
        self,
        max_chunks: Optional[int] = None,
        target_count: Optional[int] = None,
        resume: bool = True,
        retry_failed: bool = False,
    ) -> PipelineSummary:
        """Executes full scientific candidate generation, validation, quality auditing, deduplication, and splitting."""
        for d in [self.raw_dir, self.proc_dir, self.splits_dir, self.reports_dir, self.manifests_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)

        exec_id = f"gen_sci_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ckpt_mgr = ChunkCheckpointManager(self.checkpoints_dir / "generation_checkpoint.json")
        if retry_failed:
            ckpt_mgr.reset_failed()

        doc_index = self._load_documents_index()

        # Stream chunks line-by-line using a generator to prevent OOM
        def _iter_chunks():
            count = 0
            with open(self.input_chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield count, json.loads(line)
                        count += 1
                        if max_chunks and count >= max_chunks:
                            break

        if max_chunks:
            total_raw = max_chunks
        else:
            with open(self.input_chunks_path, "rb") as f:
                total_raw = sum(1 for _ in f)

        ckpt_mgr.checkpoint.total_chunks = total_raw

        all_candidates: List[DatasetRecord] = []
        accepted_candidates: List[DatasetRecord] = []
        rejected_candidates: List[Dict[str, Any]] = []

        task_counts = Counter()
        diff_counts = Counter()
        domain_counts = Counter()
        eq_chunks_count = 0
        tbl_chunks_count = 0
        quality_scores: List[float] = []

        # Generation Loop
        for c_idx, c_dict in _iter_chunks():
            if (c_idx + 1) % 250 == 0 or c_idx == 0 or c_idx == total_raw - 1:
                logging.getLogger("generate_instruction_dataset").info(
                    f"[{c_idx + 1}/{total_raw}] Generating candidates | Accepted: {len(accepted_candidates)} | Rejected: {len(rejected_candidates)}"
                )
            chunk_id = c_dict.get("chunk_id", f"chk_{c_idx:05d}")
            doc_id = c_dict.get("document_id")
            doc_meta = doc_index.get(doc_id)

            if resume and ckpt_mgr.is_chunk_completed(chunk_id):
                continue

            ckpt_mgr.mark_chunk_processing(chunk_id)

            try:
                # 1. Analyze
                analysis = self.analyzer.analyze_chunk(c_dict, doc_meta)
                if analysis.equations:
                    eq_chunks_count += 1
                if analysis.tables:
                    tbl_chunks_count += 1

                # 2. Synthesize Candidates
                chunk_seed = self.seed + (c_idx * 31) + 7
                chunk_candidates = self.synthesizer.synthesize_candidates_for_chunk(
                    analysis=analysis,
                    policy=self.policy,
                    seed=chunk_seed,
                )

                chunk_accepted = 0
                chunk_rejected = 0

                # 3. Grounding & Quality Gate
                for cand in chunk_candidates:
                    all_candidates.append(cand)
                    grounding_outcome = self.grounding_validator.validate_candidate(cand, analysis)

                    if not grounding_outcome.is_valid:
                        chunk_rejected += 1
                        rejected_candidates.append({
                            "record_id": cand.metadata.extra.get("record_id") if cand.metadata.extra else None,
                            "chunk_id": chunk_id,
                            "reason": grounding_outcome.rejection_reason,
                            "equation_status": grounding_outcome.equation_status.value,
                            "table_status": grounding_outcome.table_status.value,
                            "grounding_overlap": grounding_outcome.grounding_overlap,
                        })
                        continue

                    # 4. Quality Evaluation
                    qual_res = self.quality_evaluator.evaluate_record(cand, analysis, grounding_outcome)
                    if not qual_res.passed:
                        chunk_rejected += 1
                        rejected_candidates.append({
                            "record_id": cand.metadata.extra.get("record_id") if cand.metadata.extra else None,
                            "chunk_id": chunk_id,
                            "reason": f"Quality score below threshold: {qual_res.overall_score:.4f}",
                            "dimensions": qual_res.dimensions,
                        })
                        continue

                    # Candidate Accepted
                    chunk_accepted += 1
                    accepted_candidates.append(cand)
                    quality_scores.append(qual_res.overall_score)
                    task_counts[cand.metadata.task_type] += 1
                    diff_counts[cand.metadata.difficulty] += 1
                    domain_counts[f"{cand.metadata.domain}/{cand.metadata.topic}"] += 1

                ckpt_mgr.mark_chunk_completed(
                    chunk_id=chunk_id,
                    generated=len(chunk_candidates),
                    accepted=chunk_accepted,
                    rejected=chunk_rejected,
                )

            except Exception as e:
                ckpt_mgr.mark_chunk_failed(chunk_id, str(e))

        ckpt_mgr.save()
        # Save Raw Candidates
        atomic_write_jsonl(all_candidates, self.raw_dir / "candidates.jsonl")

        # Save Rejected Details
        with open(self.proc_dir / "rejected.jsonl", "w", encoding="utf-8") as f:
            for rj in rejected_candidates:
                f.write(json.dumps(rj) + "\n")

        # 5. Global Deduplication on Accepted Candidates
        unique_accepted, dedup_report = self.deduplicator.deduplicate(accepted_candidates)

        # Apply target count if specified
        if target_count and target_count < len(unique_accepted):
            unique_accepted = unique_accepted[:target_count]

        # Save Processed Accepted & Root combined file
        atomic_write_jsonl(unique_accepted, self.proc_dir / "accepted.jsonl")
        atomic_write_jsonl(unique_accepted, self.output_dir / "combined_candidates.jsonl")

        # 6. Source-Aware Splitting (90/5/5)
        split_result: SourceAwareSplitResult = self.splitter.split(unique_accepted)
        atomic_write_jsonl(split_result.train, self.splits_dir / "train.jsonl")
        atomic_write_jsonl(split_result.validation, self.splits_dir / "validation.jsonl")
        atomic_write_jsonl(split_result.test, self.splits_dir / "test.jsonl")

        # 7. Quality Statistics
        avg_q = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        pct_090 = (sum(1 for s in quality_scores if s >= 0.90) / len(quality_scores) * 100) if quality_scores else 0.0

        qual_summary = {
            "mean_quality_score": round(avg_q, 4),
            "pct_ge_090": round(pct_090, 2),
            "total_evaluated": len(all_candidates),
            "total_accepted": len(accepted_candidates),
            "total_rejected": len(rejected_candidates),
            "acceptance_rate": round(len(accepted_candidates) / max(1, len(all_candidates)) * 100, 2),
        }

        # Equation & Table Statistics
        eq_stats = {
            "chunks_with_equations": eq_chunks_count,
            "candidates_with_equations": sum(1 for c in unique_accepted if getattr(c.metadata, "extra", None) and getattr(c.metadata, "extra", {}).get("equation_present")),
        }
        tbl_stats = {
            "chunks_with_tables": tbl_chunks_count,
            "candidates_with_tables": sum(1 for c in unique_accepted if getattr(c.metadata, "extra", None) and getattr(c.metadata, "extra", {}).get("table_present")),
        }

        summary = PipelineSummary(
            execution_id=exec_id,
            status="COMPLETED",
            lifecycle="READY",
            seed=self.seed,
            chunks_discovered=total_raw,
            chunks_processed=ckpt_mgr.checkpoint.completed_chunks,
            chunks_with_equations=eq_chunks_count,
            chunks_with_tables=tbl_chunks_count,
            candidates_generated=len(all_candidates),
            candidates_accepted=len(accepted_candidates),
            candidates_rejected=len(rejected_candidates),
            exact_duplicates=dedup_report.exact_duplicates,
            near_duplicates=dedup_report.near_duplicates,
            unique_candidates=len(unique_accepted),
            train_count=len(split_result.train),
            validation_count=len(split_result.validation),
            test_count=len(split_result.test),
            quality_summary=qual_summary,
            domain_distribution=dict(domain_counts),
            task_distribution=dict(task_counts),
            difficulty_distribution=dict(diff_counts),
            equation_stats=eq_stats,
            table_stats=tbl_stats,
            leakage_summary=split_result.leakage_summary,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        # 8. Save Reports
        self._emit_all_reports(summary, dedup_report, split_result)

        # 9. Save Manifest
        man_path = self.manifests_dir / "dataset_manifest.json"
        manifest_data = {
            "dataset_version": "dataset-v2.0",
            "lifecycle_state": "READY",
            "execution_id": exec_id,
            "created_at": summary.started_at,
            "completed_at": summary.completed_at,
            "seed": self.seed,
            "counts": {
                "total_unique_records": len(unique_accepted),
                "train_records": len(split_result.train),
                "validation_records": len(split_result.validation),
                "test_records": len(split_result.test),
                "raw_generated": len(all_candidates),
                "rejected": len(rejected_candidates),
            },
            "quality": qual_summary,
            "leakage": split_result.leakage_summary,
            "files": {
                "train": "splits/train.jsonl",
                "validation": "splits/validation.jsonl",
                "test": "splits/test.jsonl",
                "raw_candidates": "raw/candidates.jsonl",
                "processed_accepted": "processed/accepted.jsonl",
                "processed_rejected": "processed/rejected.jsonl",
            },
        }
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        summary.manifest_path = str(man_path)
        return summary

    def _emit_all_reports(
        self,
        summary: PipelineSummary,
        dedup_report: DeduplicationReport,
        split_result: SourceAwareSplitResult,
    ) -> None:
        """Saves all 9 generation, quality, deduplication, provenance, and leakage reports."""
        # 1. Generation report (JSON & MD)
        with open(self.reports_dir / "generation_report.json", "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))

        md_gen = f"""# Phase 3.4 — Scientific Instruction Dataset Generation Report

## Summary
- **Execution ID:** `{summary.execution_id}`
- **Lifecycle State:** `{summary.lifecycle}`
- **Dataset Version:** `dataset-v2.0`
- **Deterministic Seed:** `{summary.seed}`
- **Chunks Discovered:** {summary.chunks_discovered:,}
- **Chunks Processed:** {summary.chunks_processed:,}
- **Chunks with Equations:** {summary.chunks_with_equations:,}
- **Chunks with Tables:** {summary.chunks_with_tables:,}

## Yield & Candidate Metrics
- **Candidates Generated:** {summary.candidates_generated:,}
- **Candidates Accepted:** {summary.candidates_accepted:,}
- **Candidates Rejected:** {summary.candidates_rejected:,}
- **Acceptance Rate:** {summary.quality_summary.get('acceptance_rate', 0)}%
- **Exact Duplicates:** {summary.exact_duplicates:,}
- **Near Duplicates:** {summary.near_duplicates:,}
- **Final Unique Candidates:** {summary.unique_candidates:,}

## Dataset Splits (Source-Aware Isolated)
- **Train Split (90%):** {summary.train_count:,} records
- **Validation Split (5%):** {summary.validation_count:,} records
- **Test Split (5%):** {summary.test_count:,} records
- **Cross-Split Content Leakage:** {split_result.leakage_summary.get('leakage_detected')}

## Quality Metrics
- **Mean Quality Score:** {summary.quality_summary.get('mean_quality_score')}
- **% Records >= 0.90:** {summary.quality_summary.get('pct_ge_090')}%
"""
        with open(self.reports_dir / "generation_report.md", "w", encoding="utf-8") as f:
            f.write(md_gen)

        # 2. Quality report (JSON & MD)
        with open(self.reports_dir / "quality_report.json", "w", encoding="utf-8") as f:
            json.dump(summary.quality_summary, f, indent=2)

        # 3. Provenance report
        with open(self.reports_dir / "provenance_report.json", "w", encoding="utf-8") as f:
            json.dump({
                "source": "nptel",
                "license": "CC-BY-NC-SA-4.0",
                "generator": "scientific_instruction_synthesizer",
                "version": "2.0.0",
                "domain_distribution": summary.domain_distribution,
                "task_distribution": summary.task_distribution,
            }, f, indent=2)

        # 4. Deduplication report
        with open(self.reports_dir / "deduplication_report.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(dedup_report.to_dict(), indent=2))

        # 5. Leakage report
        with open(self.reports_dir / "leakage_report.json", "w", encoding="utf-8") as f:
            json.dump(split_result.leakage_summary, f, indent=2)

        # 6. Equation report
        with open(self.reports_dir / "equation_report.json", "w", encoding="utf-8") as f:
            json.dump(summary.equation_stats, f, indent=2)

        # 7. Table report
        with open(self.reports_dir / "table_report.json", "w", encoding="utf-8") as f:
            json.dump(summary.table_stats, f, indent=2)


class InstructionDatasetPipeline:
    """Backward-compatible wrapper for Unit-based instruction generation pipeline."""

    def __init__(
        self,
        min_quality_score: float = 0.85,
        preferred_quality_score: float = 0.90,
        max_examples_per_unit: int = 3,
        enable_near_dedup: bool = True,
        near_duplicate_threshold: float = 0.85,
        random_seed: int = 42,
    ):
        self.min_quality_score = min_quality_score
        self.preferred_quality_score = preferred_quality_score
        self.max_examples_per_unit = max_examples_per_unit
        self.enable_near_dedup = enable_near_dedup
        self.near_duplicate_threshold = near_duplicate_threshold
        self.random_seed = random_seed

        self.knowledge_selector = KnowledgeSelector()
        self.task_selector = TaskSelector(deterministic_seed=random_seed)
        self.prompt_builder = InstructionPromptBuilder(deterministic_seed=random_seed)
        self.dispatcher = ScientificInstructionDispatcher()
        self.validator = InstructionValidator()
        self.quality_auditor = InstructionQualityAuditor(
            min_quality_score=min_quality_score, preferred_quality_score=preferred_quality_score
        )
        self.deduplicator = DatasetDeduplicator(
            enable_near_dedup=enable_near_dedup, near_duplicate_threshold=near_duplicate_threshold
        )
        self.stats = GenerationStatisticsAggregator()

    def run(
        self,
        input_dir_or_units: Union[str, Path, List[KnowledgeUnit]],
        output_dir: Union[str, Path],
        dry_run: bool = False,
        max_units: Optional[int] = None,
    ) -> Dict[str, Any]:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        if isinstance(input_dir_or_units, (str, Path)):
            p = Path(input_dir_or_units)
            chunks_path = p / "chunks.jsonl" if p.is_dir() else p
            chunks = []
            if chunks_path.is_file():
                with open(chunks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            chunks.append(KnowledgeChunk.model_validate(json.loads(line)))
            if max_units:
                chunks = chunks[:max_units]
            units = [KnowledgeUnit.from_knowledge_chunk(c) for c in chunks]
        else:
            units = list(input_dir_or_units)
            if max_units:
                units = units[:max_units]

        if dry_run:
            selected_units = self.knowledge_selector.select_units(units)
            return {
                "dry_run": True,
                "total_units_evaluated": len(units),
                "total_units_selected": len(selected_units),
                "selected_units_summary": [
                    {"unit_id": u.unit_id, "difficulty": u.difficulty_estimate, "types": [ct.value for ct in u.content_types]}
                    for u in selected_units[:20]
                ],
            }

        candidates: List[CandidateRecord] = []
        for unit in units:
            enriched = self.knowledge_selector.analyze_and_enrich_unit(unit)
            self.stats.record_unit_analyzed(enriched, selected=True)

            tasks = self.task_selector.select_tasks_for_unit(enriched, count=self.max_examples_per_unit)
            for idx, task_type in enumerate(tasks):
                prompt = self.prompt_builder.build_user_prompt(enriched, task_type, seed=self.random_seed + idx)
                record = self.dispatcher.dispatch_and_generate(enriched, task_type, prompt)

                grounding, math_eval, rejections = self.validator.validate_candidate(record, enriched)
                score, dims, quality_feedback = self.quality_auditor.audit_candidate(record, grounding, math_eval)
                rejections.extend(quality_feedback)

                is_acc = (score >= self.min_quality_score) and grounding.is_grounded and math_eval.is_valid and (not rejections)

                ext_prov = ExtendedProvenance(
                    source=enriched.source,
                    source_id=enriched.unit_id,
                    knowledge_document_id=enriched.document_id,
                    knowledge_section_id=enriched.section_id,
                    knowledge_chunk_id=enriched.unit_id,
                    generation_seed=self.random_seed,
                )

                cand = CandidateRecord(
                    record_id=f"cand_{enriched.unit_id}_{task_type}_{idx}",
                    record=record,
                    knowledge_unit=enriched,
                    task_type=task_type,
                    difficulty=enriched.difficulty_estimate,
                    provenance_extended=ext_prov,
                    quality_score=score,
                    quality_dimensions=dims,
                    grounding_eval=grounding,
                    math_eval=math_eval,
                    rejection_reasons=rejections,
                    is_accepted=is_acc,
                )
                candidates.append(cand)
                self.stats.record_candidate(cand)

        # Deduplication
        accepted_dataset_recs = [c.record for c in candidates if c.is_accepted]
        unique_recs, dedup_rep = self.deduplicator.deduplicate(accepted_dataset_recs)

        # Write files
        comb_path = output_path / "combined_candidates.jsonl"
        with open(comb_path, "w", encoding="utf-8") as f:
            for r in unique_recs:
                f.write(r.model_dump_json() + "\n")

        self.stats.write_reports(output_path, dedup_rep, {"combined": str(comb_path)})
        return self.stats.get_summary_dict(dedup_rep)
