"""
Unit tests for License and Provenance Handlers (Phase 3.3).
"""

import pytest
from src.ingestion.license import LicenseHandler
from src.ingestion.models import LicenseStatus
from src.ingestion.provenance import ProvenanceTracker


def test_license_permissive_evaluation():
    handler = LicenseHandler()
    res = handler.evaluate_license(
        "This research paper is distributed under a Creative Commons Attribution 4.0 International license."
    )
    assert res.license_name == "CC-BY-4.0"
    assert res.license_status == LicenseStatus.KNOWN
    assert res.internal_only is False


def test_license_nptel_restricted_evaluation():
    handler = LicenseHandler()
    res = handler.evaluate_license(
        "NPTEL course notes distributed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
        source="nptel",
    )
    assert res.license_name == "CC-BY-NC-SA-4.0"
    assert res.license_status == LicenseStatus.KNOWN
    assert res.internal_only is True


def test_license_unknown_fallback():
    handler = LicenseHandler()
    res = handler.evaluate_license("Arbitrary document with no license mention.")
    assert res.license_name == "UNKNOWN"
    assert res.license_status == LicenseStatus.UNKNOWN
    assert res.internal_only is True


def test_provenance_tracker_integration():
    tracker = ProvenanceTracker()
    tracker.register_default_sources()

    prov = tracker.create_provenance(
        source="nptel-physics-v1",
        document_id="doc_nptel_001",
    )
    assert prov.source == "nptel-physics-v1"
    assert prov.source_id == "doc_nptel_001"
    assert prov.license == "CC-BY-NC-SA-4.0"
    assert prov.generator == "knowledge_ingestion_engine"
