"""
Dataset Cleaner.
Applies rigorous rule-based cleaning, format validation, and safety filters.
Guarantees full provenance and reason tracking for every rejected record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError
from src.dataset.schema import DatasetRecord, Role


class RejectionReason(str, Enum):
    EMPTY_RECORD = "empty_record"
    EMPTY_MESSAGE = "empty_message"
    EXCESSIVELY_SHORT = "excessively_short"
    EXCESSIVELY_LONG = "excessively_long"
    CORRUPTED_UNICODE = "corrupted_unicode"
    FORMATTING_ARTIFACTS = "formatting_artifacts"
    INVALID_ROLE_SEQUENCE = "invalid_role_sequence"
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"
    INVALID_DOMAIN = "invalid_domain"
    INVALID_TASK_TYPE = "invalid_task_type"
    INVALID_DIFFICULTY = "invalid_difficulty"
    FAILED_QUALITY_CHECK = "failed_quality_check"


@dataclass
class RejectedRecord:
    reason: RejectionReason
    details: str
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    raw_preview: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason.value,
            "details": self.details,
            "source_file": self.source_file,
            "line_number": self.line_number,
            "raw_preview": self.raw_preview[:200] if self.raw_preview else None,
        }


@dataclass
class CleaningReport:
    total_inspected: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    rejections_by_reason: Dict[str, int] = field(default_factory=dict)
    rejected_records: List[RejectedRecord] = field(default_factory=list)

    def add_rejection(self, rejection: RejectedRecord):
        self.rejected_count += 1
        self.rejected_records.append(rejection)
        reason_key = rejection.reason.value
        self.rejections_by_reason[reason_key] = self.rejections_by_reason.get(reason_key, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_inspected": self.total_inspected,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "rejections_by_reason": self.rejections_by_reason,
            "rejections_summary": [r.to_dict() for r in self.rejected_records[:50]],  # Cap detailed list preview
        }


class DatasetCleaner:
    """Filters, cleans, and validates normalized record dictionaries into DatasetRecord objects."""

    # Unwanted token/formatting artifacts
    ARTIFACT_PATTERNS = [
        re.compile(r"<\|im_start\|>", re.IGNORECASE),
        re.compile(r"<\|im_end\|>", re.IGNORECASE),
        re.compile(r"<\|endoftext\|>", re.IGNORECASE),
        re.compile(r"<unk>", re.IGNORECASE),
        re.compile(r"\[INST\]|\[/INST\]"),
        re.compile(r"<s>|</s>"),
    ]

    def __init__(
        self,
        min_message_chars: int = 10,
        max_message_chars: int = 65536,
        allowed_domains: Optional[Set[str]] = None,
        allowed_task_types: Optional[Set[str]] = None,
        check_artifacts: bool = True,
    ):
        self.min_message_chars = min_message_chars
        self.max_message_chars = max_message_chars
        self.allowed_domains = allowed_domains
        self.allowed_task_types = allowed_task_types
        self.check_artifacts = check_artifacts

    def clean_records(
        self, normalized_items: List[Union[Dict[str, Any], DatasetRecord]]
    ) -> Tuple[List[DatasetRecord], CleaningReport]:
        report = CleaningReport(total_inspected=len(normalized_items))
        accepted: List[DatasetRecord] = []

        for item in normalized_items:
            if isinstance(item, DatasetRecord):
                item_dict = item.to_dict()
                raw_rec = None
            else:
                item_dict = item
                raw_rec = item.get("_raw_record") if isinstance(item, dict) else None

            source_file = raw_rec.source_file if raw_rec else None
            line_number = raw_rec.line_number if raw_rec else None
            raw_text = raw_rec.raw_text if raw_rec else str(item)

            messages = item_dict.get("messages", [])
            metadata = item_dict.get("metadata", {})

            # 1. Empty Record Check
            if not messages:
                report.add_rejection(
                    RejectedRecord(
                        reason=RejectionReason.EMPTY_RECORD,
                        details="No conversational messages found in record.",
                        source_file=source_file,
                        line_number=line_number,
                        raw_preview=raw_text,
                    )
                )
                continue

            # 2. Per-Message Checks (Empty, Length, Corrupted Unicode, Artifacts)
            has_error = False
            for idx, msg in enumerate(messages):
                content = msg.get("content", "")
                role = msg.get("role", "")

                if not content or not content.strip():
                    report.add_rejection(
                        RejectedRecord(
                            reason=RejectionReason.EMPTY_MESSAGE,
                            details=f"Message at index {idx} (role '{role}') has empty content.",
                            source_file=source_file,
                            line_number=line_number,
                            raw_preview=raw_text,
                        )
                    )
                    has_error = True
                    break

                if len(content.strip()) < self.min_message_chars:
                    report.add_rejection(
                        RejectedRecord(
                            reason=RejectionReason.EXCESSIVELY_SHORT,
                            details=f"Message at index {idx} has length {len(content.strip())} < {self.min_message_chars} chars.",
                            source_file=source_file,
                            line_number=line_number,
                            raw_preview=raw_text,
                        )
                    )
                    has_error = True
                    break

                if len(content) > self.max_message_chars:
                    report.add_rejection(
                        RejectedRecord(
                            reason=RejectionReason.EXCESSIVELY_LONG,
                            details=f"Message at index {idx} has length {len(content)} > {self.max_message_chars} chars.",
                            source_file=source_file,
                            line_number=line_number,
                            raw_preview=raw_text,
                        )
                    )
                    has_error = True
                    break

                # Check for corrupted unicode characters (replacement char \ufffd, null bytes)
                if "\ufffd" in content or "\x00" in content:
                    report.add_rejection(
                        RejectedRecord(
                            reason=RejectionReason.CORRUPTED_UNICODE,
                            details=f"Message at index {idx} contains corrupted Unicode replacement character or null byte.",
                            source_file=source_file,
                            line_number=line_number,
                            raw_preview=raw_text,
                        )
                    )
                    has_error = True
                    break

                # Check for formatting artifacts / template leaks
                if self.check_artifacts:
                    for pattern in self.ARTIFACT_PATTERNS:
                        if pattern.search(content):
                            report.add_rejection(
                                RejectedRecord(
                                    reason=RejectionReason.FORMATTING_ARTIFACTS,
                                    details=f"Message at index {idx} contains template artifact matching '{pattern.pattern}'.",
                                    source_file=source_file,
                                    line_number=line_number,
                                    raw_preview=raw_text,
                                )
                            )
                            has_error = True
                            break
                    if has_error:
                        break

            if has_error:
                continue

            # 3. Domain & Task Type Taxonomy Verification
            domain = metadata.get("domain")
            if self.allowed_domains and domain not in self.allowed_domains:
                report.add_rejection(
                    RejectedRecord(
                        reason=RejectionReason.INVALID_DOMAIN,
                        details=f"Domain '{domain}' is not in configured domain taxonomy.",
                        source_file=source_file,
                        line_number=line_number,
                        raw_preview=raw_text,
                    )
                )
                continue

            task_type = metadata.get("task_type")
            if self.allowed_task_types and task_type not in self.allowed_task_types:
                report.add_rejection(
                    RejectedRecord(
                        reason=RejectionReason.INVALID_TASK_TYPE,
                        details=f"Task type '{task_type}' is not in configured task types.",
                        source_file=source_file,
                        line_number=line_number,
                        raw_preview=raw_text,
                    )
                )
                continue

            # 4. Schema Construction & Validation
            try:
                record = DatasetRecord.from_dict({
                    "messages": messages,
                    "metadata": metadata,
                })
                accepted.append(record)
                report.accepted_count += 1
            except (ValidationError, ValueError) as ve:
                report.add_rejection(
                    RejectedRecord(
                        reason=RejectionReason.SCHEMA_VALIDATION_ERROR,
                        details=f"Schema validation failed: {str(ve)}",
                        source_file=source_file,
                        line_number=line_number,
                        raw_preview=raw_text,
                    )
                )

        return accepted, report
