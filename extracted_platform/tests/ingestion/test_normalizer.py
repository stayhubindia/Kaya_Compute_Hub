"""
Unit tests for Document Normalizer (Phase 3.3).
"""

import pytest
from src.ingestion.document_normalizer import DocumentNormalizer


def test_unicode_and_whitespace_normalization():
    normalizer = DocumentNormalizer()
    raw = "Quantum\u00a0Mechanics\u200b\n\n\n\nand   Statistical   Physics"
    normalized = normalizer.normalize(raw)
    assert normalized == "Quantum Mechanics\n\nand Statistical Physics"


def test_dehyphenation():
    normalizer = DocumentNormalizer()
    raw = "The thermo-\ndynamics of the sys-\n tem was analyzed under quan-\n tum conditions."
    normalized = normalizer.dehyphenate_text(raw)
    assert "thermodynamics" in normalized
    assert "system" in normalized
    assert "quantum" in normalized


def test_page_number_artifact_stripping():
    normalizer = DocumentNormalizer()
    raw = """
    First paragraph of physics content.
    14
    Second paragraph following page number.
    Page 15 of 50
    Third paragraph.
    [16]
    Fourth paragraph.
    """
    cleaned = normalizer.strip_page_number_artifacts(raw)
    assert "14" not in cleaned.splitlines()
    assert "Page 15 of 50" not in cleaned
    assert "[16]" not in cleaned
    assert "First paragraph of physics content." in cleaned
    assert "Fourth paragraph." in cleaned


def test_running_headers_and_arxiv_stripping():
    normalizer = DocumentNormalizer()
    raw = """
    arXiv:2301.04567v2 [astro-ph.HE] 12 Feb 2023
    We observe high-energy gamma-ray emissions from the pulsar.
    NPTEL – Chemical Engineering – Advanced Thermodynamics
    The Gibbs free energy is minimized at equilibrium.
    """
    cleaned = normalizer.strip_running_headers_and_footers(raw)
    assert "arXiv:2301.04567v2" not in cleaned
    assert "NPTEL – Chemical Engineering" not in cleaned
    assert "We observe high-energy gamma-ray emissions" in cleaned
    assert "The Gibbs free energy is minimized at equilibrium." in cleaned
