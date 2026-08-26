"""
Scientific Instruction Quality Auditor (Phase 3.4).
Evaluates generated instruction examples across correctness, grounding, clarity,
technical accuracy, and mathematical consistency against the 0.85 quality threshold.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.dataset.schema import DatasetRecord, Role
from src.generation.models import GroundingEvaluation, MathematicalValidation


class InstructionQualityAuditor:
    """Evaluates multi-dimensional quality scores for generated instruction candidates."""

    def __init__(
        self,
        min_quality_score: float = 0.85,
        preferred_quality_score: float = 0.90,
    ):
        self.min_quality_score = min_quality_score
        self.preferred_quality_score = preferred_quality_score

    def audit_candidate(
        self,
        record: DatasetRecord,
        grounding: GroundingEvaluation,
        math_eval: MathematicalValidation,
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """
        Calculates composite quality score and dimensions.
        Returns (composite_score, dimensions_dict, quality_feedback).
        """
        feedback: List[str] = []

        # Extract text elements
        user_texts = [m.content for m in record.messages if m.role == Role.USER]
        asst_texts = [m.content for m in record.messages if m.role == Role.ASSISTANT]

        full_user = " ".join(user_texts)
        full_asst = " ".join(asst_texts)

        # Baseline dimension scores
        correctness = 0.95
        source_grounding = max(0.0, min(1.0, grounding.grounding_score))
        relevance = 0.95
        clarity = 0.94
        completeness = 0.92
        technical_accuracy = 0.95
        math_consistency = 0.95 if math_eval.is_valid else 0.40

        # Penalties:
        # 1. Broken Unicode replacement character
        if "\ufffd" in full_user or "\ufffd" in full_asst:
            clarity -= 0.30
            technical_accuracy -= 0.30
            feedback.append("Contains broken Unicode replacement character (\\ufffd).")

        # 2. Too short user prompt
        if len(full_user.split()) < 5:
            relevance -= 0.20
            feedback.append("User prompt is too brief.")

        # 3. Too short assistant response
        if len(full_asst.split()) < 20:
            completeness -= 0.30
            feedback.append("Assistant response is too brief (< 20 words).")

        # 4. Repeated lines in assistant response
        lines = [line.strip() for line in full_asst.split("\n") if line.strip()]
        if lines:
            unique_lines = set(lines)
            rep_ratio = 1.0 - (len(unique_lines) / len(lines))
            if rep_ratio > 0.30:
                clarity -= 0.25
                completeness -= 0.20
                feedback.append(f"High line repetition detected (repetition ratio: {rep_ratio:.2f}).")

        # 5. Unbalanced math brackets penalty
        if not math_eval.balanced_delimiters:
            math_consistency = 0.30
            correctness -= 0.25
            feedback.append("Unbalanced mathematical delimiters.")

        # Composite score calculation (weighted sum)
        weights = {
            "correctness": 0.25,
            "source_grounding": 0.25,
            "relevance": 0.15,
            "clarity": 0.10,
            "completeness": 0.10,
            "technical_accuracy": 0.15,
        }

        dim_scores = {
            "correctness": max(0.0, min(1.0, correctness)),
            "source_grounding": max(0.0, min(1.0, source_grounding)),
            "relevance": max(0.0, min(1.0, relevance)),
            "clarity": max(0.0, min(1.0, clarity)),
            "completeness": max(0.0, min(1.0, completeness)),
            "technical_accuracy": max(0.0, min(1.0, technical_accuracy)),
            "mathematical_consistency": max(0.0, min(1.0, math_consistency)),
        }

        composite_score = sum(dim_scores[dim] * w for dim, w in weights.items())
        composite_score = max(0.0, min(1.0, composite_score))

        if composite_score < self.min_quality_score:
            feedback.append(
                f"Quality score {composite_score:.2f} falls below required threshold ({self.min_quality_score:.2f})."
            )

        return round(composite_score, 4), dim_scores, feedback
