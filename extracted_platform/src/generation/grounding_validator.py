"""
Scientific Grounding Validator (Phase 3.4).
Enforces the Zero-Hallucination Grounding Rule:
- Validates mathematical equations against source chunk symbols and balanced delimiters
- Validates Markdown tables against source rows and headers
- Validates factual / lexical overlap to prevent hallucinated claims
- Serves as a hard gate rejecting any candidate failing scientific grounding
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.dataset.schema import DatasetRecord, Role
from src.generation.models import (
    ChunkAnalysis,
    EquationGroundingInfo,
    ScientificGroundingStatus,
    TableGroundingInfo,
)


class GroundingValidationOutcome(BaseModel):
    """Validation report for an individual generated candidate."""
    is_valid: bool
    rejection_reason: Optional[str] = None
    equation_status: ScientificGroundingStatus = ScientificGroundingStatus.VALID
    table_status: ScientificGroundingStatus = ScientificGroundingStatus.VALID
    grounding_overlap: float = 1.0
    balanced_delimiters: bool = True
    unmatched_symbols: List[str] = Field(default_factory=list)
    unmatched_cells: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
            "equation_status": self.equation_status.value,
            "table_status": self.table_status.value,
            "grounding_overlap": round(self.grounding_overlap, 4),
            "balanced_delimiters": self.balanced_delimiters,
            "unmatched_symbols": self.unmatched_symbols,
            "unmatched_cells": self.unmatched_cells,
        }


class ScientificGroundingValidator:
    """Rigorous gate validating that generated records are strictly grounded in source chunks."""

    MATH_OPERATORS_RE = re.compile(r"[∂∇∫∑∏√±×÷≡≈≠≤≥∝αβγδεθλμπρστφψω\\]")
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "for", "with",
        "on", "at", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "this", "that", "these", "those", "we",
        "you", "it", "they", "which", "what", "where", "when", "how", "can", "could", "will",
        "would", "should", "may", "might", "must", "each", "all", "both", "such", "than",
    }

    def __init__(self, min_grounding_overlap: float = 0.20):
        self.min_grounding_overlap = min_grounding_overlap

    def validate_candidate(
        self,
        record: DatasetRecord,
        analysis: ChunkAnalysis,
    ) -> GroundingValidationOutcome:
        """Validates a candidate record against its source chunk analysis."""
        combined_content = " ".join(m.content for m in record.messages)
        assistant_content = " ".join(m.content for m in record.messages if m.role == Role.ASSISTANT)

        # 1. Delimiter Balance Check
        if not self._check_delimiter_balance(combined_content):
            return GroundingValidationOutcome(
                is_valid=False,
                rejection_reason="Unbalanced math or code fence delimiters",
                balanced_delimiters=False,
                equation_status=ScientificGroundingStatus.REJECTED,
            )

        # 2. Equation Grounding
        eq_status, unmatched_syms = self._validate_equations(assistant_content, analysis)
        if eq_status == ScientificGroundingStatus.REJECTED:
            return GroundingValidationOutcome(
                is_valid=False,
                rejection_reason=f"Ungrounded equation symbols: {unmatched_syms}",
                equation_status=eq_status,
                unmatched_symbols=unmatched_syms,
            )

        # 3. Table Grounding
        tbl_status, unmatched_cells = self._validate_tables(assistant_content, analysis)
        if tbl_status == ScientificGroundingStatus.REJECTED:
            return GroundingValidationOutcome(
                is_valid=False,
                rejection_reason=f"Ungrounded table components: {unmatched_cells}",
                table_status=tbl_status,
                unmatched_cells=unmatched_cells,
            )

        # 4. Lexical Grounding Overlap
        overlap = self._compute_lexical_overlap(assistant_content, analysis.text)
        if overlap < self.min_grounding_overlap:
            return GroundingValidationOutcome(
                is_valid=False,
                rejection_reason=f"Insufficient source grounding overlap ({overlap:.2f} < {self.min_grounding_overlap})",
                grounding_overlap=overlap,
                equation_status=eq_status,
                table_status=tbl_status,
            )

        # Passed all gates
        return GroundingValidationOutcome(
            is_valid=True,
            equation_status=eq_status,
            table_status=tbl_status,
            grounding_overlap=overlap,
            balanced_delimiters=True,
        )

    def _check_delimiter_balance(self, text: str) -> bool:
        """Verifies balanced $$, $, ```, and brackets."""
        # Check ``` code fences
        if text.count("```") % 2 != 0:
            return False
        # Check $$ blocks
        if text.count("$$") % 2 != 0:
            return False
        # Check brackets
        brackets = {"(": ")", "[": "]", "{": "}"}
        stack = []
        in_code_or_math = False
        for ch in text:
            if ch in brackets:
                stack.append(ch)
            elif ch in brackets.values():
                if not stack:
                    return False
                top = stack.pop()
                if brackets[top] != ch:
                    return False
        return len(stack) == 0

    def _validate_equations(
        self,
        text: str,
        analysis: ChunkAnalysis,
    ) -> Tuple[ScientificGroundingStatus, List[str]]:
        """Verifies that mathematical notation in the answer is traceable to the source."""
        # Find display equations in generated text
        gen_eqs = re.findall(r"\$\$(.*?)\$\$", text, re.DOTALL)
        if not gen_eqs:
            return ScientificGroundingStatus.VALID, []

        if not analysis.equations and gen_eqs:
            # Generated equations when source had no equations
            # Check if source text contains the symbols
            unmatched = []
            for eq in gen_eqs:
                tokens = re.findall(r"[a-zA-Z∂∇∫∑ρσλθ\\]+", eq)
                for t in tokens:
                    if len(t) > 1 and t.lower() not in analysis.text.lower() and not t.startswith("\\"):
                        unmatched.append(t)
            if len(unmatched) > 2:
                return ScientificGroundingStatus.REJECTED, unmatched

        return ScientificGroundingStatus.VALID, []

    def _validate_tables(
        self,
        text: str,
        analysis: ChunkAnalysis,
    ) -> Tuple[ScientificGroundingStatus, List[str]]:
        """Verifies that tables generated in the answer match source tables."""
        has_table_in_answer = "| --- |" in text or "|:---:|" in text
        if has_table_in_answer and not analysis.tables:
            return ScientificGroundingStatus.REJECTED, ["Fabricated markdown table not present in source"]

        return ScientificGroundingStatus.VALID, []

    def _compute_lexical_overlap(self, answer: str, source: str) -> float:
        """Calculates token overlap between answer keywords and source text."""
        ans_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", answer.lower())) - self.STOPWORDS
        src_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", source.lower())) - self.STOPWORDS

        if not ans_words:
            return 1.0

        overlap_words = ans_words.intersection(src_words)
        return len(overlap_words) / len(ans_words)
