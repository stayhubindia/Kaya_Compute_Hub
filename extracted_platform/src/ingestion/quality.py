"""
Knowledge Quality Validator and Anomaly Detector (Phase 3.3).
Audits extracted text, documents, and chunks for corruption, boilerplate, broken Unicode,
and structural degradation, assigning quality scores and audit feedback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.ingestion.models import (
    IngestionDocument,
    KnowledgeChunk,
    QualityStatus,
)


@dataclass
class QualityAuditResult:
    quality_status: QualityStatus
    quality_score: float
    feedback: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_status": self.quality_status.value,
            "quality_score": round(self.quality_score, 4),
            "feedback": self.feedback,
        }


class IngestionQualityValidator:
    """Evaluates content quality of ingested documents and chunks."""

    def __init__(
        self,
        min_doc_chars: int = 100,
        min_chunk_chars: int = 50,
        max_repeated_line_ratio: float = 0.25,
        min_quality_score: float = 0.85,
    ):
        self.min_doc_chars = min_doc_chars
        self.min_chunk_chars = min_chunk_chars
        self.max_repeated_line_ratio = max_repeated_line_ratio
        self.min_quality_score = min_quality_score

    def audit_text(self, text: str, is_chunk: bool = False) -> QualityAuditResult:
        """Audits a block of text for quality and corruption anomalies."""
        feedback: List[str] = []
        score = 1.0
        min_chars = self.min_chunk_chars if is_chunk else self.min_doc_chars

        if not text or not text.strip():
            return QualityAuditResult(
                quality_status=QualityStatus.REJECTED,
                quality_score=0.0,
                feedback=["Empty text content."],
            )

        clean_text = text.strip()
        length = len(clean_text)

        # Check minimum length
        if length < min_chars:
            score -= 0.3
            feedback.append(f"Content length ({length} chars) is below minimum threshold ({min_chars} chars).")

        # Check broken Unicode replacement characters (\ufffd)
        replacement_count = clean_text.count("\ufffd")
        if replacement_count > 0:
            penalty = min(0.4, replacement_count * 0.05)
            score -= penalty
            feedback.append(f"Detected {replacement_count} broken Unicode replacement characters.")

        # Check repeated lines
        lines = [line.strip() for line in clean_text.split("\n") if line.strip()]
        if len(lines) >= 4:
            unique_lines = set(lines)
            repeat_ratio = 1.0 - (len(unique_lines) / len(lines))
            if repeat_ratio > self.max_repeated_line_ratio:
                score -= 0.25
                feedback.append(f"High repeated line ratio ({repeat_ratio:.2%}).")

        # Check non-printable / high ASCII symbol ratio
        printable_chars = sum(1 for c in clean_text if c.isprintable() or c in "\n\t\r")
        non_printable_ratio = 1.0 - (printable_chars / max(1, length))
        if non_printable_ratio > 0.05:
            score -= 0.4
            feedback.append(f"High non-printable character ratio ({non_printable_ratio:.2%}).")

        # Bound score between 0.0 and 1.0
        score = max(0.0, min(1.0, score))

        if score >= self.min_quality_score:
            status = QualityStatus.PASSED
        elif score >= 0.50:
            status = QualityStatus.WARNING
        else:
            status = QualityStatus.REJECTED

        return QualityAuditResult(
            quality_status=status,
            quality_score=score,
            feedback=feedback,
        )

    def audit_document(self, doc: IngestionDocument) -> QualityAuditResult:
        """Evaluates overall document quality."""
        full_text = doc.get_full_text()
        return self.audit_text(full_text, is_chunk=False)

    def audit_chunk(self, chunk: KnowledgeChunk) -> QualityAuditResult:
        """Evaluates chunk-level quality and updates the chunk score."""
        res = self.audit_text(chunk.text, is_chunk=True)
        chunk.quality_score = res.quality_score
        return res
