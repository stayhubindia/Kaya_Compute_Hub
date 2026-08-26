"""
Data Models and Configuration for Scientific Instruction Generation (Phase 3.4).
Defines schemas for chunk analysis, knowledge units, candidate records, task eligibility,
grounding statuses, and checkpointing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field

from src.dataset.schema import DatasetRecord, ProvenanceInfo, SourceType
from src.ingestion.models import Equation, KnowledgeChunk, Section, Table


class ContentType(str, Enum):
    """Semantic content categories within scientific knowledge units."""
    CONCEPT = "concept"
    DEFINITION = "definition"
    DERIVATION = "derivation"
    CALCULATION = "calculation"
    EQUATION = "equation"
    EXPERIMENT = "experiment"
    COMPARISON = "comparison"
    METHODOLOGY = "methodology"
    CONCLUSION = "conclusion"
    TABLE_DATA = "table_data"


class ScientificTaskType(str, Enum):
    """The 17 supported scientific instruction task types."""
    EXPLANATION = "explanation"
    QUESTION_ANSWERING = "question_answering"
    CALCULATION = "calculation"
    PROBLEM_SOLVING = "problem_solving"
    DERIVATION = "derivation"
    PROOF = "proof"
    EQUATION_INTERPRETATION = "equation_interpretation"
    NUMERICAL_REASONING = "numerical_reasoning"
    CONCEPT_COMPARISON = "concept_comparison"
    APPLICATION = "application"
    MISCONCEPTION_CORRECTION = "misconception_correction"
    TABLE_INTERPRETATION = "table_interpretation"
    DATA_INTERPRETATION = "data_interpretation"
    SCIENTIFIC_REASONING = "scientific_reasoning"
    SCENARIO_ANALYSIS = "scenario_analysis"
    SUMMARIZATION = "summarization"
    MULTI_TURN = "multi_turn"


class ScientificGroundingStatus(str, Enum):
    """Grounding status for generated scientific content."""
    VALID = "VALID"
    UNCERTAIN = "UNCERTAIN"
    REJECTED = "REJECTED"


class ChunkCheckpointStatus(str, Enum):
    """Processing state for an individual chunk in resumable execution."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GroundingEvaluation(BaseModel):
    """Validation report on factual and citation grounding against source knowledge."""
    model_config = {"extra": "allow"}
    is_grounded: bool = True
    grounding_score: float = 1.0
    lexical_overlap: float = 1.0
    unmatched_claims: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_grounded": self.is_grounded,
            "grounding_score": round(self.grounding_score, 4),
            "lexical_overlap": round(self.lexical_overlap, 4),
            "unmatched_claims": self.unmatched_claims,
        }


class MathematicalValidation(BaseModel):
    """Validation report on mathematical notation and delimiter consistency."""
    model_config = {"extra": "allow"}
    is_valid: bool = True
    balanced_delimiters: bool = True
    has_hallucinated_symbols: bool = False
    equations_count: int = 0
    unmatched_symbols: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "balanced_delimiters": self.balanced_delimiters,
            "has_hallucinated_symbols": self.has_hallucinated_symbols,
            "equations_count": self.equations_count,
            "unmatched_symbols": self.unmatched_symbols,
        }


class ExtendedProvenance(BaseModel):
    """Detailed source tracking linking candidates to ingested document chunks and sections."""
    model_config = {"extra": "allow"}
    source: str = "nptel"
    source_type: str = SourceType.DOCUMENTATION.value
    source_id: str
    source_url: Optional[str] = None
    license: Optional[str] = "CC-BY-NC-SA-4.0"
    license_status: str = "KNOWN"
    internal_only: bool = False
    rights_verification_required: bool = False
    knowledge_document_id: Optional[str] = None
    knowledge_section_id: Optional[str] = None
    knowledge_chunk_id: Optional[str] = None
    generation_seed: Optional[int] = None
    generation_method: Optional[str] = None
    generator: str = "scientific_instruction_engine"
    generator_version: str = "2.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_provenance_info(self) -> ProvenanceInfo:
        return ProvenanceInfo(
            source_type=self.source_type,
            source=self.source,
            source_id=self.source_id,
            created_at=self.created_at,
            generator=self.generator,
            generator_version=self.generator_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "knowledge_document_id": self.knowledge_document_id,
            "knowledge_section_id": self.knowledge_section_id,
            "knowledge_chunk_id": self.knowledge_chunk_id,
            "generation_seed": self.generation_seed,
            "generator": self.generator,
            "generator_version": self.generator_version,
            "created_at": self.created_at,
        }


class KnowledgeUnit(BaseModel):
    """Enriched, self-contained unit of scientific knowledge extracted from chunks or sections."""
    model_config = {"extra": "allow"}
    unit_id: str
    document_id: str = "doc_default"
    section_id: Optional[str] = None
    chunk_id: Optional[str] = None
    title: str = ""
    domain: str = "science"
    subdomain: Optional[str] = None
    topic: str = "physics"
    subtopic: Optional[str] = None
    text: str = ""
    equations: List[Equation] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    token_estimate: int = 0
    mathematical_density: float = 0.0
    definition_density: float = 0.0
    content_types: List[ContentType] = Field(default_factory=list)
    difficulty_estimate: str = "intermediate"
    difficulty_rationale: Optional[str] = None
    selection_rationale: Optional[str] = None
    source: str = "nptel"
    source_type: str = SourceType.DOCUMENTATION.value
    source_url: Optional[str] = None
    source_file: Optional[str] = None
    license: Optional[str] = "CC-BY-NC-SA-4.0"
    license_status: str = "KNOWN"
    internal_only: bool = False
    rights_verification_required: bool = False
    provenance: Optional[Any] = None

    @classmethod
    def from_knowledge_chunk(cls, chunk: KnowledgeChunk, title: str = "") -> KnowledgeUnit:
        return cls(
            unit_id=f"ku_{chunk.chunk_id}",
            document_id=chunk.document_id,
            section_id=chunk.section_id,
            chunk_id=chunk.chunk_id,
            title=title,
            domain=chunk.domain or "science",
            subdomain=getattr(chunk, "subdomain", None),
            subtopic=getattr(chunk, "subtopic", None),
            topic=chunk.topic or "physics",
            text=chunk.text or "",
            equations=getattr(chunk, "equations", []) or [],
            tables=getattr(chunk, "tables", []) or [],
            token_estimate=getattr(chunk, "token_estimate", len(chunk.text.split())) if chunk.text else 0,
            source=chunk.source or "nptel",
            source_type=chunk.source_type or SourceType.DOCUMENTATION.value,
            license=getattr(chunk, "license", None) or "CC-BY-NC-SA-4.0",
            provenance=chunk.provenance,
        )

    @classmethod
    def from_section(cls, section: Section) -> KnowledgeUnit:
        return cls(
            unit_id=f"ku_sec_{section.section_id}",
            document_id=section.document_id,
            section_id=section.section_id,
            title=section.title,
            domain=getattr(section, "domain", "science"),
            subdomain=getattr(section, "subdomain", None),
            topic=getattr(section, "topic", "physics"),
            text=section.content,
            token_estimate=getattr(section, "token_estimate", len(section.content.split())),
            source=getattr(section, "source", "nptel"),
            source_type=getattr(section, "source_type", SourceType.DOCUMENTATION.value),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "document_id": self.document_id,
            "section_id": self.section_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "text": self.text,
            "token_estimate": self.token_estimate,
            "mathematical_density": round(self.mathematical_density, 4),
            "content_types": [ct.value for ct in self.content_types],
            "difficulty_estimate": self.difficulty_estimate,
            "source": self.source,
        }


class CandidateRecord(BaseModel):
    """Candidate instruction record enriched with extended provenance, grounding, and quality audit results."""
    model_config = {"extra": "allow"}
    record_id: str
    record: DatasetRecord
    knowledge_unit: KnowledgeUnit
    task_type: str
    difficulty: str
    provenance_extended: ExtendedProvenance
    quality_score: Optional[float] = None
    quality_dimensions: Optional[Dict[str, float]] = None
    grounding: Optional[GroundingEvaluation] = None
    grounding_eval: Optional[GroundingEvaluation] = None
    math_validation: Optional[MathematicalValidation] = None
    math_eval: Optional[MathematicalValidation] = None
    rejection_reasons: List[str] = Field(default_factory=list)
    is_accepted: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def model_post_init(self, __context: Any) -> None:
        if self.grounding is None and self.grounding_eval is not None:
            self.grounding = self.grounding_eval
        elif self.grounding_eval is None and self.grounding is not None:
            self.grounding_eval = self.grounding
        if self.math_validation is None and self.math_eval is not None:
            self.math_validation = self.math_eval
        elif self.math_eval is None and self.math_validation is not None:
            self.math_eval = self.math_validation

    def to_dict(self) -> Dict[str, Any]:
        g_eval = self.grounding_eval or self.grounding
        m_eval = self.math_eval or self.math_validation
        return {
            "record_id": self.record_id,
            "record": self.record.to_dict(),
            "knowledge_unit_id": self.knowledge_unit.unit_id,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "provenance_extended": self.provenance_extended.to_dict(),
            "quality_score": self.quality_score,
            "quality_dimensions": self.quality_dimensions,
            "grounding_eval": g_eval.to_dict() if g_eval else None,
            "math_eval": m_eval.to_dict() if m_eval else None,
            "rejection_reasons": self.rejection_reasons,
            "is_accepted": self.is_accepted,
            "created_at": self.created_at,
        }


class EquationGroundingInfo(BaseModel):
    """Grounding metadata for equations inside a generated candidate."""
    equation_present: bool = False
    equation_count: int = 0
    equation_ids: List[str] = Field(default_factory=list)
    status: ScientificGroundingStatus = ScientificGroundingStatus.VALID
    latex_snippets: List[str] = Field(default_factory=list)
    unmatched_symbols: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equation_present": self.equation_present,
            "equation_count": self.equation_count,
            "equation_ids": self.equation_ids,
            "status": self.status.value,
            "latex_snippets": self.latex_snippets,
            "unmatched_symbols": self.unmatched_symbols,
        }


class TableGroundingInfo(BaseModel):
    """Grounding metadata for tables inside a generated candidate."""
    table_present: bool = False
    table_count: int = 0
    table_ids: List[str] = Field(default_factory=list)
    status: ScientificGroundingStatus = ScientificGroundingStatus.VALID
    headers: List[str] = Field(default_factory=list)
    unmatched_cells: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_present": self.table_present,
            "table_count": self.table_count,
            "table_ids": self.table_ids,
            "status": self.status.value,
            "headers": self.headers,
            "unmatched_cells": self.unmatched_cells,
        }


class ChunkAnalysis(BaseModel):
    """Structural, mathematical, and semantic features extracted from a KnowledgeChunk."""
    model_config = {"extra": "allow"}
    chunk_id: str
    document_id: str
    section_id: str
    domain: str
    subdomain: Optional[str] = None
    topic: str
    title: str = ""
    text: str
    token_estimate: int
    equations: List[str] = Field(default_factory=list)
    tables: List[str] = Field(default_factory=list)
    definitions: List[Dict[str, str]] = Field(default_factory=list)
    theorems_or_laws: List[str] = Field(default_factory=list)
    has_derivation_steps: bool = False
    has_numerical_values: bool = False
    suitable_tasks: List[ScientificTaskType] = Field(default_factory=list)
    natural_difficulty: str = "intermediate"
    license: Optional[str] = "CC-BY-NC-SA-4.0"
    source_file: Optional[str] = None
    source_url: Optional[str] = None


class CandidateGenerationPolicy(BaseModel):
    """Configuration weights and limits for scientific candidate generation."""
    min_candidates_per_chunk: int = 1
    max_candidates_per_chunk: int = 5
    task_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "explanation": 0.15,
            "question_answering": 0.15,
            "calculation": 0.10,
            "problem_solving": 0.08,
            "derivation": 0.10,
            "proof": 0.05,
            "equation_interpretation": 0.10,
            "numerical_reasoning": 0.05,
            "concept_comparison": 0.05,
            "application": 0.05,
            "misconception_correction": 0.03,
            "table_interpretation": 0.03,
            "data_interpretation": 0.03,
            "scientific_reasoning": 0.08,
            "scenario_analysis": 0.05,
            "summarization": 0.05,
            "multi_turn": 0.05,
        }
    )
    difficulty_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "beginner": 0.25,
            "intermediate": 0.40,
            "advanced": 0.25,
            "expert": 0.10,
        }
    )
    min_quality_score: float = 0.85
    preferred_quality_score: float = 0.90
    deterministic_seed: int = 42


class CandidateGenerationResult(BaseModel):
    """Outcome for candidate generation across the chunk."""
    chunk_id: str
    candidates_count: int
    accepted_count: int
    rejected_count: int
    status: ChunkCheckpointStatus
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GenerationCheckpoint(BaseModel):
    """Resumable checkpoint tracking status across all chunks."""
    version: str = "dataset-v2.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    total_candidates_generated: int = 0
    total_candidates_accepted: int = 0
    total_candidates_rejected: int = 0
    chunk_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
