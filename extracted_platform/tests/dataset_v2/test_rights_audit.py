"""
Tests for Rights & Licensing Auditor Subsystem (Phase 3.5).
"""

import pytest
from src.dataset.rights_audit import LicenseStatus, RightsAuditor
from src.dataset.schema import DatasetRecord, Message, ProvenanceInfo, RecordMetadata, Role, SourceType


def make_record(rec_id: str, license_name: str, source_type: str = SourceType.DOCUMENTATION.value) -> DatasetRecord:
    return DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Explain quantum entanglement."),
            Message(role=Role.ASSISTANT, content="Quantum entanglement is a physical phenomenon..."),
        ],
        metadata=RecordMetadata(
            domain="science",
            topic="quantum_physics",
            difficulty="intermediate",
            task_type="conceptual_explanation",
            source_type=source_type,
            source="nptel_physics",
            provenance=ProvenanceInfo(
                source_type=source_type,
                source="nptel_physics",
                source_id=f"doc_{rec_id}",
                license=license_name,
            ),
        ),
    )


def test_rights_classification_open_license():
    auditor = RightsAuditor()
    rec = make_record("001", "CC-BY-4.0")
    res = auditor.classify_record(rec)
    assert res.status == LicenseStatus.LICENSE_VERIFIED
    assert res.is_releasable is True


def test_rights_classification_internal_license():
    auditor = RightsAuditor(allow_internal_only=True)
    rec = make_record("002", "NPTEL_EDUCATIONAL")
    res = auditor.classify_record(rec)
    assert res.status == LicenseStatus.INTERNAL_ONLY
    assert res.is_releasable is True


def test_rights_classification_unknown_license():
    auditor_allow = RightsAuditor(allow_unknown_license=True)
    auditor_deny = RightsAuditor(allow_unknown_license=False)

    rec = make_record("003", "UNKNOWN")
    res1 = auditor_allow.classify_record(rec)
    assert res1.status == LicenseStatus.LICENSE_UNKNOWN
    assert res1.is_releasable is True

    res2 = auditor_deny.classify_record(rec)
    assert res2.is_releasable is False


def test_audit_dataset_quarantine(tmp_path):
    auditor = RightsAuditor(
        allowed_licenses=["CC-BY-4.0", "MIT"],
        allow_unknown_license=False,
    )
    records = [
        make_record("001", "CC-BY-4.0"),
        make_record("002", "MIT"),
        make_record("003", "PROPRIETARY_UNAUTHORIZED"),
    ]

    accepted, quarantined, result, classifications = auditor.audit_dataset(records)
    assert len(accepted) == 2
    assert len(quarantined) == 1
    assert result.quarantined_count == 1
    assert result.releasable_count == 2

    # Save reports
    saved = auditor.save_reports(result, quarantined, classifications, tmp_path)
    assert saved["json_report"].exists()
    assert saved["md_report"].exists()
    assert saved["quarantine_file"].exists()
