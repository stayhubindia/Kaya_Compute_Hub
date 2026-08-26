"""
Unit tests for Ingestion Quality Validator (Phase 3.3).
"""

import pytest
from src.ingestion.models import QualityStatus
from src.ingestion.quality import IngestionQualityValidator


def test_quality_empty_text():
    validator = IngestionQualityValidator()
    res = validator.audit_text("")
    assert res.quality_status == QualityStatus.REJECTED
    assert res.quality_score == 0.0


def test_quality_broken_unicode_detection():
    validator = IngestionQualityValidator()
    text = "Valid physics text containing corrupted \ufffd\ufffd\ufffd characters." * 5
    res = validator.audit_text(text)
    assert any("broken Unicode" in f for f in res.feedback)
    assert res.quality_score < 1.0


def test_quality_repeated_lines_penalty():
    validator = IngestionQualityValidator()
    text = "Repeated line of boilerplate\n" * 10
    res = validator.audit_text(text)
    assert any("repeated line ratio" in f for f in res.feedback)
    assert res.quality_score < 0.85


def test_quality_high_quality_scientific_text():
    validator = IngestionQualityValidator()
    text = """
    In quantum mechanics, the Hamiltonian operator represents the total energy of the physical system.
    For a conservative system, the energy eigenvalues satisfy the time-independent Schrodinger equation.
    The boundary conditions dictate the discrete quantization of permissible energy states.
    """
    res = validator.audit_text(text)
    assert res.quality_status == QualityStatus.PASSED
    assert res.quality_score >= 0.85
