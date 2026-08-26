"""
Integration tests for Knowledge Ingestion Pipeline (Phase 3.3).
"""

import json
from pathlib import Path
import pytest

from src.ingestion.pipeline import KnowledgeIngestionPipeline


def test_pipeline_end_to_end_execution(tmp_path):
    input_dir = tmp_path / "raw_docs"
    input_dir.mkdir()
    output_dir = tmp_path / "knowledge_output"

    # Create sample HTML file
    sample_html = """
    <html>
    <head><title>Quantum Electrodynamics Lecture Notes</title></head>
    <body>
        <h1>1. Introduction to Gauge Invariance</h1>
        <p>Gauge invariance is a fundamental symmetry of quantum field theory.</p>
        <p>The Lagrangian density is invariant under local U(1) gauge transformations.</p>
        <table>
            <tr><th>Field</th><th>Symmetry</th></tr>
            <tr><td>Photon</td><td>U(1)</td></tr>
        </table>
    </body>
    </html>
    """
    (input_dir / "lecture_1.html").write_text(sample_html, encoding="utf-8")

    # Create sample JSON file
    sample_json = {
        "title": "Black Hole Thermodynamics",
        "authors": ["J. Bekenstein"],
        "abstract": "The horizon area of a black hole behaves as thermodynamic entropy.",
        "text": "The generalized second law states that total entropy never decreases.",
        "categories": ["astro-ph.HE", "gr-qc"],
    }
    (input_dir / "paper_1.json").write_text(json.dumps(sample_json), encoding="utf-8")

    pipeline = KnowledgeIngestionPipeline(
        output_dir=output_dir,
        source="arxiv",
        resume=False,
        force=True,
    )

    stats = pipeline.run(input_dir)

    assert stats.documents_discovered == 2
    assert stats.documents_successful >= 1
    assert stats.total_chunks >= 2

    # Verify generated output files
    docs_file = output_dir / "documents.jsonl"
    sections_file = output_dir / "sections.jsonl"
    chunks_file = output_dir / "chunks.jsonl"
    manifest_file = output_dir / "manifest.json"
    report_json = output_dir / "reports" / "ingestion_report.json"
    report_md = output_dir / "reports" / "ingestion_report.md"

    assert docs_file.is_file()
    assert sections_file.is_file()
    assert chunks_file.is_file()
    assert manifest_file.is_file()
    assert report_json.is_file()
    assert report_md.is_file()

    # Verify manifest contents
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["source"] == "arxiv"
    assert manifest_data["counts"]["documents_discovered"] == 2
    assert "science" in manifest_data["distributions"]["domains"]
