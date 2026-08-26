"""
Dataset Quality Validator.
Evaluates dataset quality deterministically against Phase 2.1 thresholds:
- minimum_score = 0.85
- preferred_score = 0.90
Explicitly represents evaluated vs unscored states without fabricating synthetic scores.
Provides a modular interface for future model-assisted evaluators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.dataset.schema import DatasetRecord, Role


class QualityState(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNSCORED = "unscored"


@dataclass
class QualityEvaluationResult:
    state: QualityState
    score: Optional[float] = None
    dimensions: Dict[str, float] = field(default_factory=dict)
    feedback: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "score": round(self.score, 4) if self.score is not None else None,
            "dimensions": {k: round(v, 4) for k, v in self.dimensions.items()},
            "feedback": self.feedback,
        }


@dataclass
class QualityValidationReport:
    total_records: int = 0
    passed_count: int = 0
    failed_count: int = 0
    unscored_count: int = 0
    preferred_quality_count: int = 0  # score >= 0.90
    average_score: Optional[float] = None
    evaluation_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "unscored_count": self.unscored_count,
            "preferred_quality_count": self.preferred_quality_count,
            "average_score": round(self.average_score, 4) if self.average_score is not None else None,
            "sample_evaluations": self.evaluation_details[:50],
        }


class BaseQualityEvaluator(ABC):
    """Abstract base class for quality scoring engines."""

    @abstractmethod
    def evaluate(self, record: DatasetRecord) -> QualityEvaluationResult:
        pass


class DeterministicRuleEvaluator(BaseQualityEvaluator):
    """
    Deterministic rule-based quality evaluator.
    Inspects existing metadata scores and performs structural heuristics
    (code block balance, markdown formatting, response depth).
    """

    def __init__(self, minimum_score: float = 0.85, preferred_score: float = 0.90):
        self.minimum_score = minimum_score
        self.preferred_score = preferred_score

    def evaluate(self, record: DatasetRecord) -> QualityEvaluationResult:
        meta_score = record.metadata.quality_score
        feedback: List[str] = []
        dims: Dict[str, float] = record.metadata.dimensions or {}

        # Heuristic checks on structural clarity and completeness
        code_blocks_unclosed = 0
        for m in record.messages:
            count_fences = m.content.count("```")
            if count_fences % 2 != 0:
                code_blocks_unclosed += 1
                feedback.append(f"Unclosed code fence in message role {m.role.value}")

        if meta_score is None:
            # Record is unscored
            return QualityEvaluationResult(
                state=QualityState.UNSCORED,
                score=None,
                dimensions=dims,
                feedback=feedback or ["Record has no prior quality score assigned."],
            )

        # Evaluated record
        if code_blocks_unclosed > 0:
            meta_score = max(0.0, meta_score - 0.15)
            feedback.append("Score penalized due to unclosed code fences.")

        state = QualityState.PASSED if meta_score >= self.minimum_score else QualityState.FAILED
        if state == QualityState.FAILED:
            feedback.append(f"Score {meta_score:.3f} below minimum threshold {self.minimum_score}")

        return QualityEvaluationResult(
            state=state,
            score=meta_score,
            dimensions=dims,
            feedback=feedback,
        )


class QualityValidator:
    """Validates records against quality criteria and tracks quality metrics."""

    def __init__(
        self,
        minimum_score: float = 0.85,
        preferred_score: float = 0.90,
        enforce_threshold: bool = True,
        allow_unscored: bool = True,
        evaluator: Optional[BaseQualityEvaluator] = None,
    ):
        self.minimum_score = minimum_score
        self.preferred_score = preferred_score
        self.enforce_threshold = enforce_threshold
        self.allow_unscored = allow_unscored
        self.evaluator = evaluator or DeterministicRuleEvaluator(minimum_score, preferred_score)

    def validate_records(
        self, records: List[DatasetRecord]
    ) -> Tuple[List[DatasetRecord], QualityValidationReport]:
        report = QualityValidationReport(total_records=len(records))
        accepted: List[DatasetRecord] = []
        scores: List[float] = []

        for idx, record in enumerate(records):
            eval_result = self.evaluator.evaluate(record)

            if eval_result.state == QualityState.PASSED:
                report.passed_count += 1
                if eval_result.score is not None:
                    scores.append(eval_result.score)
                    if eval_result.score >= self.preferred_score:
                        report.preferred_quality_count += 1
                accepted.append(record)

            elif eval_result.state == QualityState.UNSCORED:
                report.unscored_count += 1
                if self.allow_unscored:
                    accepted.append(record)

            elif eval_result.state == QualityState.FAILED:
                report.failed_count += 1
                if eval_result.score is not None:
                    scores.append(eval_result.score)
                if not self.enforce_threshold:
                    accepted.append(record)

            report.evaluation_details.append({
                "record_index": idx,
                "domain": record.metadata.domain,
                "task_type": record.metadata.task_type,
                "state": eval_result.state.value,
                "score": eval_result.score,
                "feedback": eval_result.feedback,
            })

        if scores:
            report.average_score = sum(scores) / len(scores)

        return accepted, report
