"""
Unit tests for HTML Document Extraction Engine (Phase 3.3).
"""

import pytest
from src.ingestion.html_extractor import HTMLExtractor
from src.ingestion.models import ExtractionStatus


def test_html_boilerplate_removal_and_extraction():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>General Relativity and Gravitational Waves</title>
        <meta name="author" content="Albert Einstein">
        <meta name="citation_author" content="Arthur Eddington">
        <link rel="canonical" href="https://arxiv.org/abs/2301.99999">
    </head>
    <body>
        <nav><a href="/home">Home</a> | <a href="/menu">Menu</a></nav>
        <header><h1>Site Header Banner</h1></header>
        <script>alert("Untrusted JS");</script>
        <style>body { color: red; }</style>

        <h1>1. Introduction to Curvature</h1>
        <p>Spacetime tells matter how to move; matter tells spacetime how to curve.</p>

        <h2>1.1 Einstein Field Equations</h2>
        <p>The field equations relate spacetime geometry to energy-momentum distribution.</p>

        <table>
            <caption>Physical Constants</caption>
            <tr><th>Constant</th><th>Value</th><th>Units</th></tr>
            <tr><td>Speed of Light</td><td>299792458</td><td>m/s</td></tr>
            <tr><td>Gravitational Constant</td><td>6.674e-11</td><td>m^3 kg^-1 s^-2</td></tr>
        </table>

        <footer>Copyright 2026 Physics Archive</footer>
    </body>
    </html>
    """

    extractor = HTMLExtractor()
    doc, telemetry = extractor.extract_html(html_content, source="arxiv")

    assert doc is not None
    assert telemetry.extraction_status == ExtractionStatus.SUCCESS
    assert doc.metadata.title == "General Relativity and Gravitational Waves"
    assert "Albert Einstein" in doc.metadata.authors
    assert doc.metadata.canonical_url == "https://arxiv.org/abs/2301.99999"

    # Ensure boilerplate was stripped
    full_text = doc.get_full_text()
    assert "Site Header Banner" not in full_text
    assert "Untrusted JS" not in full_text
    assert "Copyright 2026 Physics Archive" not in full_text

    # Ensure content was preserved
    assert "Spacetime tells matter how to move" in full_text
    assert "Einstein Field Equations" in full_text
    assert "| Speed of Light | 299792458 | m/s |" in full_text
    assert telemetry.tables_detected == 1


def test_html_math_tag_conversion():
    html_content = """
    <html>
    <body>
        <h2>Quantum Mechanics</h2>
        <p>The energy of a photon is given by
           <math alttext="E = h \\nu"><annotation encoding="application/x-tex">E = h \\nu</annotation></math>.
        </p>
    </body>
    </html>
    """

    extractor = HTMLExtractor()
    doc, telemetry = extractor.extract_html(html_content, source="nptel")

    assert doc is not None
    full_text = doc.get_full_text()
    assert "E = h \\nu" in full_text or "E = h" in full_text


def test_html_corrupted_empty_handling():
    extractor = HTMLExtractor()
    doc, telemetry = extractor.extract_html("   ", source="generic")

    assert doc is not None
    assert telemetry.extraction_status == ExtractionStatus.FAILED
    assert telemetry.characters_extracted == 0
