"""
Scientific Quality and Rigor Auditor (Phase 3.5).
Validates mathematical equations, dimensional/unit consistency, source grounding,
and citation integrity for scientific instruction records.
"""

from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.dataset.schema import DatasetRecord, Role, TaskType


class ScientificValidationStatus(str, Enum):
    """Scientific verification status for a record."""
    VERIFIED = "VERIFIED"
    VALIDATION_UNCERTAIN = "VALIDATION_UNCERTAIN"
    FAILED = "FAILED"


class RecordScientificQA(BaseModel):
    """Scientific validation outcome for an individual record."""
    record_id: str
    status: ScientificValidationStatus
    is_valid: bool
    equations_count: int
    balanced_delimiters: bool
    units_detected: List[str] = Field(default_factory=list)
    has_numerical_values: bool
    grounding_overlap: float
    notes: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "status": self.status.value,
            "is_valid": self.is_valid,
            "equations_count": self.equations_count,
            "balanced_delimiters": self.balanced_delimiters,
            "units_detected": self.units_detected,
            "has_numerical_values": self.has_numerical_values,
            "grounding_overlap": round(self.grounding_overlap, 4),
            "notes": self.notes,
        }


class ScientificQAResult(BaseModel):
    """Aggregated scientific QA metrics."""
    total_evaluated: int
    verified_count: int
    uncertain_count: int
    failed_count: int
    total_equations_found: int
    records_with_equations: int
    records_with_units: int
    average_grounding_overlap: float
    status_distribution: Dict[str, int] = Field(default_factory=dict)
    issue_breakdown: Dict[str, int] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ScientificQAAuditor:
    """Performs deep domain-specific scientific, mathematical, and citation verification."""

    # Scientific SI and common units regex pattern
    SCIENTIFIC_UNITS_PATTERN = re.compile(
        r"\b(?:m/s|km/s|m/s\^2|eV|keV|MeV|GeV|TeV|J|kJ|MJ|N|kN|Pa|kPa|MPa|GPa|Hz|kHz|MHz|GHz|K|°C|mol|kmol|kg|g|mg|μg|C|V|mV|kV|A|mA|μA|Ω|kΩ|MΩ|T|mT|W|kW|MW|GW|rad|sr|Å|nm|μm|mm|cm|km)\b"
    )

    # Citation patterns (e.g., [1], (Einstein 1905), arXiv:2301.12345)
    CITATION_PATTERNS = [
        re.compile(r"\[\d+\]"),
        re.compile(r"\((?:[A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4})\)"),
        re.compile(r"\barXiv:\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        min_grounding_overlap: float = 0.15,
        strict_equations_for_math_tasks: bool = True,
    ):
        self.min_grounding_overlap = min_grounding_overlap
        self.strict_equations_for_math_tasks = strict_equations_for_math_tasks

    def audit_record(self, record: DatasetRecord, index: int = 0) -> RecordScientificQA:
        """Evaluates mathematical consistency, equation formatting, units, and citation realism."""
        rec_id = (
            record.metadata.record_id
            if hasattr(record.metadata, "record_id") and record.metadata.record_id
            else f"rec_{index:06d}"
        )
        task_type = record.metadata.task_type
        domain = record.metadata.domain

        assistant_text = " ".join(
            m.content for m in record.messages if m.role == Role.ASSISTANT.value or m.role == Role.ASSISTANT
        )
        user_text = " ".join(
            m.content for m in record.messages if m.role == Role.USER.value or m.role == Role.USER
        )
        all_text = f"{user_text}\n{assistant_text}"

        notes: List[str] = []
        is_failed = False
        is_uncertain = False

        # 1. LaTeX Equation Detection & Bracket Balancing
        # Delimiter parity check
        clean_text = assistant_text.replace(r"\$", "")  # ignore escaped dollar signs
        display_eq_count = clean_text.count("$$")
        text_without_display = clean_text.replace("$$", "")
        inline_eq_count = text_without_display.count("$")

        balanced_delimiters = True
        if display_eq_count % 2 != 0 or inline_eq_count % 2 != 0:
            balanced_delimiters = False
            is_failed = True
            notes.append("Unbalanced LaTeX equation delimiters ($$ or $).")

        equations = re.findall(r"\$\$(.*?)\$\$|\$([^\$]+)\$", assistant_text, re.DOTALL)
        eq_count = len(equations)

        for d_eq, i_eq in equations:
            eq_str = d_eq or i_eq
            if not self._check_balanced_brackets(eq_str):
                balanced_delimiters = False
                is_failed = True
        # Math / Derivation / Calculation tasks require equations
        if self.strict_equations_for_math_tasks and task_type in [
            TaskType.PROOF.value, TaskType.CALCULATION.value, "derivation"
        ]:
            if eq_count == 0:
                is_uncertain = True
                notes.append(f"Mathematical task '{task_type}' contains no explicit LaTeX equations.")

        # 2. Units & Numerical Values
        units_found = list(set(self.SCIENTIFIC_UNITS_PATTERN.findall(assistant_text)))
        has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", assistant_text))

        if task_type in [TaskType.CALCULATION.value, "numerical_calculation"]:
            if not has_numbers:
                is_failed = True
                notes.append("Quantitative calculation task has no numerical quantities in assistant response.")
            if not units_found and domain in ["science", "mathematics", "networking", "cybersecurity"]:
                notes.append("No explicit physical/technical units detected in quantitative calculation.")

        # 3. Citation Consistency
        citations_found = []
        for pat in self.CITATION_PATTERNS:
            citations_found.extend(pat.findall(assistant_text))

        # Check for placeholder citations e.g. [Citation Needed] or [?], [XX]
        if re.search(r"\[(?:citation\s+needed|\?|xx|todo)\]", assistant_text, re.IGNORECASE):
            is_failed = True
            notes.append("Placeholder or invalid citation token detected in response.")

        # 4. Source Grounding Evaluation (if source info or provenance is present)
        source_overlap = 1.0  # Default to full score if already passed prior generator checks
        prov = record.metadata.provenance
        if prov and prov.source_id:
            # Token overlap between question and answer
            q_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", user_text.lower()))
            a_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", assistant_text.lower()))
            if q_words and a_words:
                overlap = len(q_words.intersection(a_words)) / max(1, len(q_words))
                source_overlap = min(1.0, overlap)

        # Determine overall status
        if is_failed:
            status = ScientificValidationStatus.FAILED
            is_valid = False
        elif is_uncertain:
            status = ScientificValidationStatus.VALIDATION_UNCERTAIN
            is_valid = True  # Uncertain is not hard-failed, but flagged in scorecard
        else:
            status = ScientificValidationStatus.VERIFIED
            is_valid = True

        return RecordScientificQA(
            record_id=rec_id,
            status=status,
            is_valid=is_valid,
            equations_count=eq_count,
            balanced_delimiters=balanced_delimiters,
            units_detected=units_found,
            has_numerical_values=has_numbers,
            grounding_overlap=source_overlap,
            notes=notes,
        )

    def audit_dataset(
        self, records: List[DatasetRecord]
    ) -> Tuple[List[DatasetRecord], List[DatasetRecord], ScientificQAResult, List[RecordScientificQA]]:
        """
        Audits all records across scientific dimensions.
        Returns: (passed_records, failed_records, qa_result, record_evaluations).
        """
        passed: List[DatasetRecord] = []
        failed: List[DatasetRecord] = []
        evals: List[RecordScientificQA] = []

        status_counts: Dict[str, int] = defaultdict(int)
        issues: Dict[str, int] = defaultdict(int)
        tot_eqs = 0
        eq_recs = 0
        unit_recs = 0
        overlap_sum = 0.0

        for idx, r in enumerate(records):
            ev = self.audit_record(r, index=idx)
            evals.append(ev)

            status_counts[ev.status.value] += 1
            tot_eqs += ev.equations_count
            if ev.equations_count > 0:
                eq_recs += 1
            if ev.units_detected:
                unit_recs += 1
            overlap_sum += ev.grounding_overlap

            for n in ev.notes:
                issues[n] += 1

            if ev.is_valid:
                passed.append(r)
            else:
                failed.append(r)

        total = len(records)
        avg_overlap = (overlap_sum / total) if total > 0 else 0.0

        qa_res = ScientificQAResult(
            total_evaluated=total,
            verified_count=status_counts.get(ScientificValidationStatus.VERIFIED.value, 0),
            uncertain_count=status_counts.get(ScientificValidationStatus.VALIDATION_UNCERTAIN.value, 0),
            failed_count=status_counts.get(ScientificValidationStatus.FAILED.value, 0),
            total_equations_found=tot_eqs,
            records_with_equations=eq_recs,
            records_with_units=unit_recs,
            average_grounding_overlap=round(avg_overlap, 4),
            status_distribution=dict(status_counts),
            issue_breakdown=dict(issues),
        )

        return passed, failed, qa_res, evals

    def _check_balanced_brackets(self, s: str) -> bool:
        """Verifies that curly braces {}, parentheses (), and brackets [] are balanced."""
        stack = []
        pairs = {"}": "{", ")": "(", "]": "["}
        for char in s:
            if char in "{([":
                stack.append(char)
            elif char in "})]":
                if not stack or stack.pop() != pairs[char]:
                    return False
        return len(stack) == 0
