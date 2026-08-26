"""
Rights and License Auditor Subsystem (Phase 3.5).
Classifies and verifies license metadata, enforces release policies, isolates unverified records
into quarantine, and generates structured rights audit reports.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from src.dataset.schema import DatasetRecord, SourceType


class LicenseStatus(str, Enum):
    """Classification of record licensing rights."""
    LICENSE_VERIFIED = "LICENSE_VERIFIED"
    LICENSE_UNKNOWN = "LICENSE_UNKNOWN"
    RIGHTS_REVIEW_REQUIRED = "RIGHTS_REVIEW_REQUIRED"
    INTERNAL_ONLY = "INTERNAL_ONLY"


class RecordRightsClassification(BaseModel):
    """Audit outcome for an individual record's licensing."""
    record_id: str
    source_type: str
    source_name: str
    source_id: str
    license_declared: str
    status: LicenseStatus
    is_releasable: bool
    review_notes: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "source_id": self.source_id,
            "license_declared": self.license_declared,
            "status": self.status.value,
            "is_releasable": self.is_releasable,
            "review_notes": self.review_notes,
        }


class RightsAuditResult(BaseModel):
    """Aggregated rights and licensing audit metrics."""
    total_records: int
    verified_count: int
    unknown_count: int
    review_required_count: int
    internal_only_count: int
    releasable_count: int
    quarantined_count: int
    license_distribution: Dict[str, int] = Field(default_factory=dict)
    status_distribution: Dict[str, int] = Field(default_factory=dict)
    rejection_reasons: Dict[str, int] = Field(default_factory=dict)
    quarantine_record_ids: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class RightsAuditor:
    """Audits rights, licenses, and legal compliance across scientific dataset records."""

    # Recognized standard open licenses
    VERIFIED_LICENSES: Set[str] = {
        "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "NPTEL_EDUCATIONAL",
        "CC0", "CC0-1.0", "MIT", "APACHE-2.0",
        "PUBLIC_DOMAIN", "BSD-3-CLAUSE", "OPEN_DATA_COMMONS",
    }

    # Educational / Non-commercial licenses
    INTERNAL_LICENSES: Set[str] = {
        "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "NPTEL_EDUCATIONAL", "ARXIV_NONCOMMERCIAL",
        "ACADEMIC_FREE_LICENSE", "PROPRIETARY_INTERNAL",
    }

    def __init__(
        self,
        allowed_licenses: Optional[List[str]] = None,
        allow_unknown_license: bool = True,
        allow_internal_only: bool = True,
        quarantine_unauthorized: bool = True,
    ):
        self.allowed_licenses = set(allowed_licenses) if allowed_licenses else None
        self.allow_unknown_license = allow_unknown_license
        self.allow_internal_only = allow_internal_only
        self.quarantine_unauthorized = quarantine_unauthorized

    def classify_record(self, record: DatasetRecord, index: int = 0) -> RecordRightsClassification:
        """Evaluates rights compliance for a single dataset record."""
        rec_id = (
            record.metadata.record_id
            if hasattr(record.metadata, "record_id") and record.metadata.record_id
            else f"rec_{index:06d}"
        )
        prov = record.metadata.provenance
        source_type = record.metadata.source_type or (prov.source_type if prov else "unknown")
        source_name = record.metadata.source or (prov.source if prov else "unknown")
        source_id = prov.source_id if prov else f"idx_{index}"
        declared_license = (prov.license if prov and prov.license else "UNKNOWN").upper().strip()

        review_notes: List[str] = []
        is_releasable = True

        # Classify status
        if declared_license in self.VERIFIED_LICENSES:
            status = LicenseStatus.LICENSE_VERIFIED
        elif declared_license in self.INTERNAL_LICENSES:
            status = LicenseStatus.INTERNAL_ONLY
            if not self.allow_internal_only:
                is_releasable = False
                review_notes.append(f"Internal-only license '{declared_license}' not permitted by release policy.")
        elif declared_license in ["UNKNOWN", "UNSPECIFIED", "NONE", ""]:
            status = LicenseStatus.LICENSE_UNKNOWN
            review_notes.append("License metadata is unknown/unspecified.")
            if not self.allow_unknown_license:
                is_releasable = False
                review_notes.append("Unknown license prohibited by strict release policy.")
        else:
            # Custom or ambiguous license
            status = LicenseStatus.RIGHTS_REVIEW_REQUIRED
            review_notes.append(f"Ambiguous or unrecognized license string: '{declared_license}'")
            if not self.allow_unknown_license:
                is_releasable = False

        # If explicit whitelist configured, enforce it
        if self.allowed_licenses:
            allowed_upper = {lic.upper() for lic in self.allowed_licenses}
            if declared_license not in allowed_upper:
                is_releasable = False
                review_notes.append(f"Declared license '{declared_license}' not in allowed policy whitelist.")

        return RecordRightsClassification(
            record_id=rec_id,
            source_type=source_type,
            source_name=source_name,
            source_id=source_id,
            license_declared=declared_license,
            status=status,
            is_releasable=is_releasable,
            review_notes=review_notes,
        )

    def audit_dataset(
        self, records: List[DatasetRecord]
    ) -> Tuple[List[DatasetRecord], List[DatasetRecord], RightsAuditResult, List[RecordRightsClassification]]:
        """
        Audits all records, splitting into accepted releasable records and quarantined records.
        Returns: (accepted_records, quarantined_records, audit_result, record_classifications).
        """
        accepted: List[DatasetRecord] = []
        quarantined: List[DatasetRecord] = []
        classifications: List[RecordRightsClassification] = []

        lic_counts: Dict[str, int] = defaultdict(int)
        status_counts: Dict[str, int] = defaultdict(int)
        rej_reasons: Dict[str, int] = defaultdict(int)
        quarantine_ids: List[str] = []

        for idx, rec in enumerate(records):
            cls_res = self.classify_record(rec, index=idx)
            classifications.append(cls_res)

            lic_counts[cls_res.license_declared] += 1
            status_counts[cls_res.status.value] += 1

            if cls_res.is_releasable:
                accepted.append(rec)
            else:
                quarantined.append(rec)
                quarantine_ids.append(cls_res.record_id)
                for note in cls_res.review_notes:
                    rej_reasons[note] += 1

        total = len(records)
        verified_cnt = status_counts.get(LicenseStatus.LICENSE_VERIFIED.value, 0)
        unknown_cnt = status_counts.get(LicenseStatus.LICENSE_UNKNOWN.value, 0)
        review_cnt = status_counts.get(LicenseStatus.RIGHTS_REVIEW_REQUIRED.value, 0)
        internal_cnt = status_counts.get(LicenseStatus.INTERNAL_ONLY.value, 0)

        result = RightsAuditResult(
            total_records=total,
            verified_count=verified_cnt,
            unknown_count=unknown_cnt,
            review_required_count=review_cnt,
            internal_only_count=internal_cnt,
            releasable_count=len(accepted),
            quarantined_count=len(quarantined),
            license_distribution=dict(lic_counts),
            status_distribution=dict(status_counts),
            rejection_reasons=dict(rej_reasons),
            quarantine_record_ids=quarantine_ids,
        )

        return accepted, quarantined, result, classifications

    def save_reports(
        self,
        audit_result: RightsAuditResult,
        quarantined_records: List[DatasetRecord],
        classifications: List[RecordRightsClassification],
        output_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """Saves reports/rights_audit.json, reports/rights_audit.md, and quarantine jsonl."""
        out_path = Path(output_dir).resolve()
        reports_dir = out_path / "reports"
        quarantine_dir = out_path / "quarantine"

        reports_dir.mkdir(parents=True, exist_ok=True)
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        json_file = reports_dir / "rights_audit.json"
        md_file = reports_dir / "rights_audit.md"
        quarantine_file = quarantine_dir / "rights_review_required.jsonl"

        # 1. JSON Report
        report_data = audit_result.to_dict()
        report_data["generated_at"] = datetime.now(timezone.utc).isoformat()
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 2. Markdown Report
        md_content = self._generate_markdown(audit_result, classifications)
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        # 3. Quarantine JSONL
        if quarantined_records:
            with open(quarantine_file, "w", encoding="utf-8") as f:
                for r in quarantined_records:
                    f.write(r.model_dump_json() + "\n")

        return {
            "json_report": json_file,
            "md_report": md_file,
            "quarantine_file": quarantine_file,
        }

    def _generate_markdown(
        self, audit_result: RightsAuditResult, classifications: List[RecordRightsClassification]
    ) -> str:
        """Builds Markdown report for rights and license audit."""
        lines: List[str] = [
            "# Rights & Licensing Audit Report",
            "",
            f"**Audit Timestamp**: {datetime.now(timezone.utc).isoformat()}  ",
            f"**Total Records Evaluated**: `{audit_result.total_records:,}`  ",
            f"**Releasable Records**: `{audit_result.releasable_count:,}` ({(audit_result.releasable_count / max(1, audit_result.total_records)):.1%})  ",
            f"**Quarantined Records**: `{audit_result.quarantined_count:,}`  ",
            "",
            "## 1. Licensing Status Distribution",
            "",
            "| Rights Status | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ]

        total = max(1, audit_result.total_records)
        for st, count in sorted(audit_result.status_distribution.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| **`{st}`** | `{count:,}` | `{count / total:.1%}` |")

        lines.extend([
            "",
            "## 2. Declared License Types",
            "",
            "| License | Occurrences | Percentage |",
            "| :--- | :--- | :--- |",
        ])

        for lic, count in sorted(audit_result.license_distribution.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| `{lic}` | `{count:,}` | `{count / total:.1%}` |")

        if audit_result.rejection_reasons:
            lines.extend([
                "",
                "## 3. Quarantine Reasons & Gating Warnings",
                "",
                "| Reason | Records Affected |",
                "| :--- | :--- |",
            ])
            for reason, count in sorted(audit_result.rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {reason} | `{count:,}` |")

        lines.append("")
        return "\n".join(lines)
