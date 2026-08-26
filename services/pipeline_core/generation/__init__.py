"""
Phase 3.4 — Source-Grounded Scientific Instruction Synthesis Engine.
Transforms ingested scientific knowledge chunks into high-fidelity conversational instruction datasets.
"""

from src.generation.answer_generator import ScientificInstructionDispatcher
from src.generation.checkpoint import GenerationCheckpointManager
from src.generation.checkpoint_manager import ChunkCheckpointManager
from src.generation.grounding_validator import GroundingValidationOutcome, ScientificGroundingValidator
from src.generation.knowledge_selector import KnowledgeSelector
from src.generation.models import (
    CandidateGenerationPolicy,
    CandidateGenerationResult,
    CandidateRecord,
    ChunkAnalysis,
    ChunkCheckpointStatus,
    ContentType,
    EquationGroundingInfo,
    ExtendedProvenance,
    GenerationCheckpoint,
    GroundingEvaluation,
    KnowledgeUnit,
    MathematicalValidation,
    ScientificGroundingStatus,
    ScientificTaskType,
    TableGroundingInfo,
)
from src.generation.pipeline import InstructionDatasetPipeline, ScientificGenerationPipeline
from src.generation.prompt_builder import InstructionPromptBuilder, ScientificPromptBuilder
from src.generation.quality import InstructionQualityAuditor
from src.generation.quality_evaluator import ScientificQualityEvaluation, ScientificQualityEvaluator
from src.generation.scientific_synthesizer import ScientificInstructionSynthesizer
from src.generation.source_analyzer import SourceChunkAnalyzer
from src.generation.source_aware_splitter import SourceAwareSplitResult, SourceAwareSplitter
from src.generation.statistics import GenerationStatisticsAggregator
from src.generation.task_selector import TaskSelector
from src.generation.validator import InstructionValidator

__all__ = [
    "ScientificTaskType",
    "ScientificGroundingStatus",
    "EquationGroundingInfo",
    "TableGroundingInfo",
    "ChunkAnalysis",
    "CandidateGenerationPolicy",
    "CandidateGenerationResult",
    "ChunkCheckpointStatus",
    "GenerationCheckpoint",
    "ScientificGenerationPipeline",
    "InstructionDatasetPipeline",
    "CandidateRecord",
    "ContentType",
    "ExtendedProvenance",
    "GroundingEvaluation",
    "KnowledgeUnit",
    "MathematicalValidation",
    "KnowledgeSelector",
    "TaskSelector",
    "ScientificPromptBuilder",
    "InstructionPromptBuilder",
    "ScientificInstructionDispatcher",
    "InstructionValidator",
    "InstructionQualityAuditor",
    "GenerationStatisticsAggregator",
    "ScientificGroundingValidator",
    "GroundingValidationOutcome",
    "ScientificQualityEvaluator",
    "ScientificQualityEvaluation",
    "ScientificInstructionSynthesizer",
    "SourceChunkAnalyzer",
    "SourceAwareSplitter",
    "SourceAwareSplitResult",
    "GenerationCheckpointManager",
]
