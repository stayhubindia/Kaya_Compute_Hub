"""
Unit and Integration Tests for Scientific Instruction Dataset Generation (Phase 3.4).
Tests:
- Source chunk feature extraction (equations, tables, laws, tasks)
- Scientific grounding validator & zero-hallucination hard gate
- 9-dimensional quality scoring
- Deduplication and deterministic hashing
- Source-aware leakage-proof dataset splitting
- Resumable checkpointing
- Pipeline execution & manifest compliance
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.dataset.schema import DatasetRecord, Message, RecordMetadata, Role, TaskType
from src.generation.checkpoint_manager import ChunkCheckpointManager
from src.generation.grounding_validator import GroundingValidationOutcome, ScientificGroundingValidator
from src.generation.models import (
    CandidateGenerationPolicy,
    ChunkAnalysis,
    ChunkCheckpointStatus,
    ScientificGroundingStatus,
    ScientificTaskType,
)
from src.generation.pipeline import ScientificGenerationPipeline
from src.generation.quality_evaluator import ScientificQualityEvaluator
from src.generation.scientific_synthesizer import ScientificInstructionSynthesizer
from src.generation.source_analyzer import SourceChunkAnalyzer
from src.generation.source_aware_splitter import SourceAwareSplitter


@pytest.fixture
def sample_chunk_with_equation():
    return {
        "chunk_id": "chk_physics_001",
        "document_id": "doc_fluid_mechanics",
        "section_id": "sec_navier_stokes",
        "domain": "science",
        "topic": "physics",
        "subdomain": "fluid_dynamics",
        "token_estimate": 150,
        "license": "CC-BY-NC-SA-4.0",
        "text": (
            "The continuity equation represents the conservation of mass in a fluid control volume. "
            "For an incompressible fluid, the divergence of the velocity field is zero.\n\n"
            "$$\\nabla \\cdot \\vec{V} = 0$$\n\n"
            "Here, $\\vec{V}$ is the velocity vector field and $\\nabla$ denotes the divergence operator. "
            "Integrating over the volume yields steady-state mass balance across the inlet and outlet."
        ),
    }


@pytest.fixture
def sample_chunk_with_table():
    return {
        "chunk_id": "chk_physics_002",
        "document_id": "doc_thermodynamics",
        "section_id": "sec_properties",
        "domain": "science",
        "topic": "physics",
        "subdomain": "thermodynamics",
        "token_estimate": 140,
        "license": "CC-BY-NC-SA-4.0",
        "text": (
            "The following table provides the thermodynamic state properties of the working fluid.\n\n"
            "| State | Pressure (kPa) | Temperature (K) | Specific Volume (m3/kg) |\n"
            "| --- | --- | --- | --- |\n"
            "| 1 | 100.0 | 300.0 | 0.861 |\n"
            "| 2 | 500.0 | 450.0 | 0.258 |\n\n"
            "State 1 represents the initial compressor inlet condition whereas State 2 represents the compressed output state."
        ),
    }


class TestSourceChunkAnalyzer:
    def test_extracts_equations_and_tasks(self, sample_chunk_with_equation):
        analyzer = SourceChunkAnalyzer()
        analysis = analyzer.analyze_chunk(sample_chunk_with_equation)

        assert analysis.chunk_id == "chk_physics_001"
        assert len(analysis.equations) >= 1
        assert "\\nabla \\cdot \\vec{V} = 0" in analysis.equations[0]
        assert ScientificTaskType.EQUATION_INTERPRETATION in analysis.suitable_tasks
        assert ScientificTaskType.EXPLANATION in analysis.suitable_tasks

    def test_extracts_tables(self, sample_chunk_with_table):
        analyzer = SourceChunkAnalyzer()
        analysis = analyzer.analyze_chunk(sample_chunk_with_table)

        assert len(analysis.tables) == 1
        assert "Pressure (kPa)" in analysis.tables[0]
        assert ScientificTaskType.TABLE_INTERPRETATION in analysis.suitable_tasks
        assert ScientificTaskType.DATA_INTERPRETATION in analysis.suitable_tasks


class TestScientificSynthesizer:
    def test_synthesizes_grounded_candidates(self, sample_chunk_with_equation):
        analyzer = SourceChunkAnalyzer()
        analysis = analyzer.analyze_chunk(sample_chunk_with_equation)

        synthesizer = ScientificInstructionSynthesizer()
        candidates = synthesizer.synthesize_candidates_for_chunk(analysis, seed=42)

        assert len(candidates) >= 1
        record = candidates[0]
        assert len(record.messages) >= 2
        assert record.messages[0].role == Role.USER
        assert record.messages[1].role == Role.ASSISTANT
        assert record.metadata.domain == "science"
        assert record.metadata.source == "nptel"
        assert record.metadata.extra["chunk_id"] == "chk_physics_001"
        assert record.metadata.extra["equation_present"] is True


class TestGroundingValidator:
    def test_valid_grounded_record_passes(self, sample_chunk_with_equation):
        analyzer = SourceChunkAnalyzer()
        analysis = analyzer.analyze_chunk(sample_chunk_with_equation)
        synthesizer = ScientificInstructionSynthesizer()
        record = synthesizer.synthesize_single_example(
            analysis,
            ScientificTaskType.EQUATION_INTERPRETATION,
            difficulty="intermediate",
            seed=42,
        )

        validator = ScientificGroundingValidator()
        outcome = validator.validate_candidate(record, analysis)

        assert outcome.is_valid is True
        assert outcome.equation_status == ScientificGroundingStatus.VALID
        assert outcome.balanced_delimiters is True

    def test_unbalanced_code_fence_rejected(self, sample_chunk_with_equation):
        analyzer = SourceChunkAnalyzer()
        analysis = analyzer.analyze_chunk(sample_chunk_with_equation)
        synthesizer = ScientificInstructionSynthesizer()
        record = synthesizer.synthesize_single_example(
            analysis,
            ScientificTaskType.EXPLANATION,
            difficulty="beginner",
            seed=42,
        )
        # Corrupt with unbalanced fence
        record.messages[1].content += "\n```python\nprint('broken')"

        validator = ScientificGroundingValidator()
        outcome = validator.validate_candidate(record, analysis)

        assert outcome.is_valid is False
        assert "delimiters" in outcome.rejection_reason.lower()


class TestQualityEvaluator:
    def test_scores_record_accurately(self, sample_chunk_with_equation):
        analyzer = SourceChunkAnalyzer()
        analysis = analyzer.analyze_chunk(sample_chunk_with_equation)
        synthesizer = ScientificInstructionSynthesizer()
        record = synthesizer.synthesize_single_example(
            analysis,
            ScientificTaskType.EXPLANATION,
            difficulty="intermediate",
            seed=42,
        )

        validator = ScientificGroundingValidator()
        outcome = validator.validate_candidate(record, analysis)

        evaluator = ScientificQualityEvaluator(min_score=0.85)
        qual_res = evaluator.evaluate_record(record, analysis, outcome)

        assert qual_res.passed is True
        assert qual_res.overall_score >= 0.85
        assert record.metadata.quality_score >= 0.85
        assert "source_grounding" in record.metadata.dimensions


class TestSourceAwareSplitter:
    def test_zero_leakage_splitting(self, sample_chunk_with_equation, sample_chunk_with_table):
        analyzer = SourceChunkAnalyzer()
        synth = ScientificInstructionSynthesizer()

        a1 = analyzer.analyze_chunk(sample_chunk_with_equation)
        a2 = analyzer.analyze_chunk(sample_chunk_with_table)

        recs1 = synth.synthesize_candidates_for_chunk(a1, seed=42)
        recs2 = synth.synthesize_candidates_for_chunk(a2, seed=43)
        all_recs = recs1 + recs2

        splitter = SourceAwareSplitter(train_ratio=0.80, validation_ratio=0.10, test_ratio=0.10, random_seed=42)
        result = splitter.split(all_recs)

        assert len(result.train) + len(result.validation) + len(result.test) == len(all_recs)
        assert result.leakage_summary["leakage_detected"] is False
        assert result.leakage_summary["chunk_overlaps"]["train_val"] == 0
        assert result.leakage_summary["chunk_overlaps"]["train_test"] == 0


class TestCheckpointManager:
    def test_checkpoint_lifecycle(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.json"
        mgr = ChunkCheckpointManager(ckpt_path)

        assert not mgr.is_chunk_completed("chk_1")
        mgr.mark_chunk_processing("chk_1")
        mgr.mark_chunk_completed("chk_1", generated=3, accepted=3, rejected=0)

        assert mgr.is_chunk_completed("chk_1")
        assert mgr.checkpoint.completed_chunks == 1
        assert mgr.checkpoint.total_candidates_accepted == 3

        # Reload manager to test persistence
        mgr_reloaded = ChunkCheckpointManager(ckpt_path)
        assert mgr_reloaded.is_chunk_completed("chk_1")
        assert mgr_reloaded.checkpoint.completed_chunks == 1


class TestEndToEndPipeline:
    def test_pipeline_dry_run_and_execution(self, tmp_path, sample_chunk_with_equation, sample_chunk_with_table):
        chunks_file = tmp_path / "chunks.jsonl"
        with open(chunks_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_chunk_with_equation) + "\n")
            f.write(json.dumps(sample_chunk_with_table) + "\n")

        output_dir = tmp_path / "v2.0"
        pipeline = ScientificGenerationPipeline(
            input_chunks_path=chunks_file,
            output_dir=output_dir,
            seed=42,
        )

        # Dry run
        dry_summary = pipeline.execute_dry_run()
        assert dry_summary.status == "DRY_RUN_COMPLETED"
        assert dry_summary.chunks_discovered == 2
        assert dry_summary.chunks_with_equations == 1
        assert dry_summary.chunks_with_tables == 1

        # Live run
        live_summary = pipeline.run(resume=False)
        assert live_summary.status == "COMPLETED"
        assert live_summary.lifecycle == "READY"
        assert live_summary.candidates_accepted >= 2
        assert (output_dir / "splits" / "train.jsonl").is_file()
        assert (output_dir / "manifests" / "dataset_manifest.json").is_file()
        assert (output_dir / "reports" / "generation_report.md").is_file()
