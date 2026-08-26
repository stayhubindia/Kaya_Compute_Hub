"""
Scientific Quality Evaluator (Phase 3.4).
Computes multi-dimensional quality scores across 9 dimensions:
- correctness
- source_grounding (hard gate)
- relevance
- clarity
- completeness
- technical_accuracy
- reasoning_quality
- equation_fidelity
- table_fidelity
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.dataset.schema import DatasetRecord, Role
from src.generation.grounding_validator import GroundingValidationOutcome
from src.generation.models import ChunkAnalysis


class ScientificQualityEvaluation(BaseModel):
    """Multi-dimensional quality score result for an individual candidate."""
    passed: bool
    overall_score: float
    dimensions: Dict[str, float] = Field(default_factory=dict)
    feedback: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "overall_score": round(self.overall_score, 4),
            "dimensions": {k: round(v, 4) for k, v in self.dimensions.items()},
            "feedback": self.feedback,
        }


class ScientificQualityEvaluator:
    """Evaluates scientific instruction candidates across 9 dimensional metrics."""

    DIMENSION_WEIGHTS = {
        "correctness": 0.20,
        "source_grounding": 0.25,
        "relevance": 0.10,
        "clarity": 0.10,
        "completeness": 0.10,
        "technical_accuracy": 0.10,
        "reasoning_quality": 0.05,
        "equation_fidelity": 0.05,
        "table_fidelity": 0.05,
    }

    def __init__(self, min_score: float = 0.85, preferred_score: float = 0.90):
        self.min_score = min_score
        self.preferred_score = preferred_score

    def evaluate_record(
        self,
        record: DatasetRecord,
        analysis: ChunkAnalysis,
        grounding_outcome: GroundingValidationOutcome,
    ) -> ScientificQualityEvaluation:
        """Computes deterministic 9-dimensional quality scores."""
        feedback: List[str] = []
        dims: Dict[str, float] = {}

        assistant_msgs = [m.content for m in record.messages if m.role == Role.ASSISTANT]
        user_msgs = [m.content for m in record.messages if m.role == Role.USER]
        full_ans = " ".join(assistant_msgs)
        full_usr = " ".join(user_msgs)

        # 1. Source Grounding (Hard Gate)
        grounding_score = min(1.0, max(0.0, grounding_outcome.grounding_overlap + 0.35))
        if not grounding_outcome.is_valid:
            grounding_score = 0.0
            feedback.append(f"Grounding failure: {grounding_outcome.rejection_reason}")
        dims["source_grounding"] = grounding_score

        # 2. Correctness
        correctness = 0.95 if grounding_outcome.balanced_delimiters and grounding_score >= 0.80 else 0.70
        dims["correctness"] = correctness

        # 3. Relevance (User query alignment)
        relevance = 0.95 if len(full_usr.split()) >= 8 else 0.85
        dims["relevance"] = relevance

        # 4. Clarity (Formatting, paragraph structure)
        clarity = 0.95 if "###" in full_ans or "\n\n" in full_ans else 0.85
        dims["clarity"] = clarity

        # 5. Completeness (Sufficient length and depth)
        ans_words = len(full_ans.split())
        completeness = 0.95 if ans_words >= 60 else (0.85 if ans_words >= 30 else 0.70)
        dims["completeness"] = completeness

        # 6. Technical Accuracy
        tech_acc = 0.95 if not grounding_outcome.unmatched_symbols else 0.60
        dims["technical_accuracy"] = tech_acc

        # 7. Reasoning Quality
        has_reasoning_markers = any(
            w in full_ans.lower() for w in ["because", "therefore", "thus", "hence", "leads to", "consequently", "step"]
        )
        dims["reasoning_quality"] = 0.95 if has_reasoning_markers else 0.88

        # 8. Equation Fidelity
        if "$$" in full_ans:
            dims["equation_fidelity"] = 0.98 if grounding_outcome.balanced_delimiters else 0.40
        else:
            dims["equation_fidelity"] = 1.0

        # 9. Table Fidelity
        if "| --- |" in full_ans:
            dims["table_fidelity"] = 0.98 if not grounding_outcome.unmatched_cells else 0.40
        else:
            dims["table_fidelity"] = 1.0

        # Weighted score computation
        overall = sum(dims[k] * self.DIMENSION_WEIGHTS[k] for k in self.DIMENSION_WEIGHTS)

        passed = overall >= self.min_score and dims["source_grounding"] >= 0.80 and grounding_outcome.is_valid

        # Synchronize metadata
        record.metadata.quality_score = round(overall, 4)
        record.metadata.dimensions = {k: round(v, 4) for k, v in dims.items()}

        return ScientificQualityEvaluation(
            passed=passed,
            overall_score=overall,
            dimensions=dims,
            feedback=feedback,
        )
